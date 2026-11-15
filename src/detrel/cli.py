"""Command-line entry points. Subcommands are added as milestones land."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone
from typing import Any

import torch

from detrel import data, evaluate, metrics, model, plots
from detrel.config import load_config, resolve_path
from detrel.corruptions import CORRUPTION_NAMES

logger = logging.getLogger("detrel")

SPLITS = ("calibration", "evaluation")


def prepare_data(cfg: dict[str, Any]) -> tuple[dict[str, Any], data.Split]:
    """Download annotations and subset images, then create or verify the split manifest.

    Args:
        cfg: Validated config.

    Returns:
        The parsed annotations and the calibration/evaluation split.
    """
    ann_path = data.ensure_annotations(cfg)
    coco = data.load_annotations(ann_path)
    split = data.make_split(cfg, coco, data.sha256_file(ann_path))
    data.ensure_images(cfg, coco, split.calibration + split.evaluation)
    logger.info(
        "Data ready: %d calibration + %d evaluation images",
        len(split.calibration),
        len(split.evaluation),
    )
    return coco, split


class Workspace:
    """Loaded annotations, split, and a lazily built detector shared across conditions."""

    def __init__(self, cfg: dict[str, Any]):
        """Prepare data and resolve the device; the model is built on first cache miss."""
        self.cfg = cfg
        self.coco, self.split = prepare_data(cfg)
        self.image_dir = resolve_path(cfg["paths"]["data_dir"]) / cfg["paths"]["images"]
        self.device = model.resolve_device(cfg["model"]["device"])
        self._detector: torch.nn.Module | None = None
        self.computed: list[str] = []  # conditions actually inferred (not from cache)

    def image_ids(self, split: str) -> list[int]:
        """Image ids of ``calibration`` or ``evaluation``."""
        return getattr(self.split, split)

    def detector(self) -> torch.nn.Module:
        """The detector on the target device (built once)."""
        if self._detector is None:
            logger.info("Loading %s on %s", self.cfg["model"]["name"], self.device)
            self._detector = model.build_detector(self.cfg).to(self.device)
        return self._detector

    def detections(
        self, split: str, corruption: str | None, severity: int
    ) -> list[model.ImageDetections]:
        """Detections for one split and condition, from cache when valid."""
        cfg, ids = self.cfg, self.image_ids(split)
        key = model.condition_key(corruption, severity)
        path = resolve_path(cfg["paths"]["cache_dir"]) / split / f"{key}.pt"
        meta = model.cache_metadata(cfg, split, ids, corruption, severity)

        def compute() -> list[model.ImageDetections]:
            self.computed.append(f"{split}/{key}")
            logger.info("Running %s on %s split", key, split)
            subset = data.CocoSubset(self.coco, self.image_dir, ids)
            ds = model.CorruptedImages(subset, corruption, severity, cfg["seed"])
            m = cfg["model"]
            return model.run_inference(
                self.detector(), ds, m["batch_size"], m["num_workers"], self.device, m["amp"]
            )

        return model.cached_detections(path, meta, compute)


def _map_table(
    ws: Workspace, split: str, dets: list[model.ImageDetections]
) -> dict[str, dict[str, float]]:
    d = ws.cfg["data"]
    ids = ws.image_ids(split)
    rows = {"driving (8 classes)": list(d["driving_classes"].values())}
    rows |= {f"  group: {g}": cats for g, cats in d["class_groups"].items()}
    rows["all 80 classes"] = [c["id"] for c in ws.coco["categories"]]
    logger.info("%-26s %8s %8s", f"[{split}]", "mAP50", "mAP")
    table = {}
    for name, cats in rows.items():
        table[name.strip()] = r = metrics.coco_map(ws.coco, dets, ids, cats)
        logger.info("%-26s %8.3f %8.3f", name, r["mAP50"], r["mAP"])
    return table


def baseline(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Clean inference on both splits (cached) and an mAP table.

    Args:
        cfg: Validated config.
        args: Parsed CLI arguments (unused).

    Returns:
        ``{split: {row_name: {"mAP": ..., "mAP50": ...}}}``.
    """
    del args
    ws = Workspace(cfg)
    tables = {split: _map_table(ws, split, ws.detections(split, None, 0)) for split in SPLITS}
    logger.info(
        "Reference: torchvision reports %.1f mAP (all classes, full val2017); the all-class "
        "row above should be in the same range.",
        cfg["model"]["reference_map_val2017"],
    )
    return tables


def infer(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Run (or load) one split x condition; useful for timing.

    Args:
        cfg: Validated config.
        args: Parsed CLI arguments with ``split``, ``corruption``, ``severity``.
    """
    ws = Workspace(cfg)
    dets = ws.detections(args.split, args.corruption, args.severity)
    _map_table(ws, args.split, dets)


def conditions(cfg: dict[str, Any]) -> list[tuple[str | None, int]]:
    """Clean followed by every configured corruption x severity."""
    c = cfg["corruptions"]
    return [(None, 0)] + [(name, sev) for name in c["names"] for sev in c["severities"]]


def sweep(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Full sweep: inference for every condition (cached), metrics, CSV outputs.

    Writes ``summary.csv``, ``reliability.csv`` and ``run_info.json`` (paths from config).

    Args:
        cfg: Validated config.
        args: Parsed CLI arguments (unused).

    Returns:
        The summary rows.
    """
    del args
    t_start = time.perf_counter()
    ws = Workspace(cfg)
    t_data = time.perf_counter()

    ws.detections("calibration", None, 0)  # needed for temperature scaling (step 9)
    dets = {}
    for corruption, severity in conditions(cfg):
        dets[model.condition_key(corruption, severity)] = ws.detections(
            "evaluation", corruption, severity
        )
    t_infer = time.perf_counter()

    # Temperature scaling: fit on the clean calibration split only (brief §6.4).
    cal_ids = ws.image_ids(cfg["calibration"]["fit_split"])
    fit = evaluate.fit_on_split(
        cfg,
        ws.coco,
        data.ground_truth(ws.coco, cal_ids),
        ws.detections(cfg["calibration"]["fit_split"], None, 0),
    )
    c = cfg["calibration"]
    calib_info = {
        "temperature": fit.temperature,
        "nll_before": fit.nll_before,
        "nll_after": fit.nll_after,
        "n_detections": fit.n,
        "n_tp": fit.n_tp,
        "at_bound": fit.at_bound,
        "method": c["method"],
        "fit_split": c["fit_split"],
        "fit_scope": c["fit_scope"],
        "fit_score_threshold": c["fit_score_threshold"],
    }
    calib_path = resolve_path(cfg["paths"]["calibration_json"])
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    calib_path.write_text(json.dumps(calib_info, indent=1) + "\n", encoding="utf-8")

    ids = ws.image_ids("evaluation")
    gt = data.ground_truth(ws.coco, ids)
    boot = cfg["metrics"]["bootstrap"]
    weights = metrics.bootstrap_weights(len(ids), boot["n_resamples"], cfg["seed"])
    summary, reliability, clean = [], [], None
    for corruption, severity in conditions(cfg):
        key = model.condition_key(corruption, severity)
        res = evaluate.evaluate_condition(
            cfg,
            ws.coco,
            gt,
            dets[key],
            ids,
            weights,
            corruption or model.CLEAN,
            severity,
            temperature=fit.temperature,
        )
        clean = clean or res
        summary += evaluate.summary_rows(cfg, res, clean)
        reliability += evaluate.reliability_rows(res)
        logger.info("Metrics done: %s", key)
    t_metrics = time.perf_counter()

    paths = cfg["paths"]
    evaluate.write_csv(resolve_path(paths["summary_csv"]), summary)
    evaluate.write_csv(resolve_path(paths["reliability_csv"]), reliability)
    info = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": {
            "data": round(t_data - t_start, 1),
            "inference": round(t_infer - t_data, 1),
            "metrics": round(t_metrics - t_infer, 1),
            "total": round(t_metrics - t_start, 1),
        },
        "conditions_inferred": ws.computed,
        "conditions_total": 1 + len(dets),
        "n_eval_images": len(ids),
        "bootstrap_resamples": boot["n_resamples"],
        "device": torch.cuda.get_device_name(0) if ws.device.type == "cuda" else "cpu",
        "amp": cfg["model"]["amp"],
        "temperature": fit.temperature,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "numpy": __import__("numpy").__version__,
        },
    }
    path = resolve_path(paths["run_info"])
    path.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    logger.info(
        "Sweep finished in %.1f min (data %.0fs, inference %.0fs for %d conditions, metrics %.0fs)",
        info["seconds"]["total"] / 60,
        info["seconds"]["data"],
        info["seconds"]["inference"],
        len(ws.computed),
        info["seconds"]["metrics"],
    )
    return summary


def make_plots(cfg: dict[str, Any], args: argparse.Namespace) -> list[Any]:
    """Render figures from the committed CSVs (no GPU, no data needed).

    Args:
        cfg: Validated config.
        args: Parsed CLI arguments (unused).

    Returns:
        Paths of the written figures.
    """
    del args
    paths = cfg["paths"]
    run_info, calib = resolve_path(paths["run_info"]), resolve_path(paths["calibration_json"])
    n_images = (
        json.loads(run_info.read_text(encoding="utf-8")).get("n_eval_images")
        if run_info.exists()
        else None
    )
    temperature = (
        json.loads(calib.read_text(encoding="utf-8"))["temperature"] if calib.exists() else None
    )
    return plots.make_figures(
        cfg,
        resolve_path(paths["summary_csv"]),
        resolve_path(paths["reliability_csv"]),
        resolve_path(paths["figures_dir"]),
        n_images,
        temperature,
    )


def run_all(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Everything, in order: sweep (data, inference, metrics) then figures."""
    sweep(cfg, args)
    make_plots(cfg, args)


def main(argv: list[str] | None = None) -> int:
    """Run the ``detrel`` command-line interface.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="detrel")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data", help="Download data and create/verify the split manifest.")
    sub.add_parser("baseline", help="Clean inference on both splits and an mAP table.")
    sub.add_parser("sweep", help="All conditions: inference (cached), metrics, summary.csv.")
    sub.add_parser("plots", help="Figures from the result CSVs (no GPU or data needed).")
    sub.add_parser("run-all", help="sweep, then plots.")
    p_infer = sub.add_parser("infer", help="Run one split x condition (cached).")
    p_infer.add_argument("--split", choices=SPLITS, default="evaluation")
    p_infer.add_argument("--corruption", choices=CORRUPTION_NAMES, default=None)
    p_infer.add_argument("--severity", type=int, choices=range(0, 6), default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    if args.command == "prepare-data":
        prepare_data(cfg)
    elif args.command == "baseline":
        baseline(cfg, args)
    elif args.command == "sweep":
        sweep(cfg, args)
    elif args.command == "plots":
        make_plots(cfg, args)
    elif args.command == "run-all":
        run_all(cfg, args)
    else:
        infer(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
