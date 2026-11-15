"""Command-line entry points. Subcommands are added as milestones land."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import torch

from detrel import data, metrics, model
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
    else:
        infer(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
