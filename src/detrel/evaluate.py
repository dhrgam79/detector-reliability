"""Turn one condition's detections into result rows (accuracy + calibration, with CIs).

Every condition is evaluated with the *same* bootstrap image weights, so differences to the
clean condition are paired: ``delta = metric(condition) - metric(clean)`` per resample.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from detrel.calibration import FitResult, apply_temperature, fit_temperature
from detrel.data import GroundTruth
from detrel.matching import match_detections
from detrel.metrics import (
    MapEvaluator,
    coco_map,
    dece,
    dece_resamples,
    image_index,
    percentile_ci,
    select_detections,
    summarize_ap,
)
from detrel.model import ImageDetections

ALL_CLASSES = "all80"
DRIVING = "driving"


def scopes(cfg: dict[str, Any], coco: dict[str, Any]) -> dict[str, list[int]]:
    """Category sets reported separately: driving (8), each group, and all COCO classes."""
    d = cfg["data"]
    out = {DRIVING: sorted(d["driving_classes"].values())}
    out |= {g: sorted(c) for g, c in d["class_groups"].items()}
    out[ALL_CLASSES] = sorted(c["id"] for c in coco["categories"])
    return out


def metric_names(cfg: dict[str, Any], calibrated: bool = False) -> list[str]:
    """Metric keys: ``mAP``, ``mAP50``, ``dece_t<thr>`` and, if calibrated, ``dece_cal_t<thr>``."""
    thresholds = cfg["metrics"]["dece_score_thresholds"]
    names = ["mAP", "mAP50"] + [f"dece_t{t:g}" for t in thresholds]
    if calibrated:
        names += [f"dece_cal_t{t:g}" for t in thresholds]
    return names


def fit_on_split(
    cfg: dict[str, Any],
    coco: dict[str, Any],
    gt: Mapping[int, GroundTruth],
    detections: Sequence[ImageDetections],
) -> FitResult:
    """Fit the temperature on one split's detections (scope and threshold from config).

    Uses the same matching as D-ECE, so the TP/FP labels the fit sees are exactly the ones
    D-ECE is later computed with.
    """
    c = cfg["calibration"]
    pooled = match_detections(detections, gt, cfg["matching"]["iou_threshold"])
    sel = select_detections(pooled, c["fit_score_threshold"], scopes(cfg, coco)[c["fit_scope"]])
    lo, hi = c["temperature_bounds"]
    return fit_temperature(sel.score, sel.tp, (lo, hi), c["logit_eps"])


@dataclass
class ConditionResult:
    """Point estimates and bootstrap samples of one condition.

    Attributes:
        corruption: Corruption name or ``clean``.
        severity: 0 for clean.
        points: ``{(scope, metric): value}``.
        samples: ``{(scope, metric): (R,) bootstrap samples}`` (absent = no CI, e.g. all-80 mAP).
        counts: ``{(scope, key): int}`` with ``n_gt`` and ``n_det_t<thr>``.
        reliability: ``{(scope, threshold, calibrated): DeceResult}``.
    """

    corruption: str
    severity: int
    points: dict[tuple[str, str], float] = field(default_factory=dict)
    samples: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    counts: dict[tuple[str, str], int] = field(default_factory=dict)
    reliability: dict[tuple[str, float], Any] = field(default_factory=dict)


def evaluate_condition(
    cfg: dict[str, Any],
    coco: dict[str, Any],
    gt: Mapping[int, GroundTruth],
    detections: Sequence[ImageDetections],
    image_ids: Sequence[int],
    weights: np.ndarray,
    corruption: str,
    severity: int,
    temperature: float | None = None,
) -> ConditionResult:
    """Compute every reported metric for one condition.

    Args:
        cfg: Validated config.
        coco: Ground-truth annotations.
        gt: Ground truth per image (from :func:`detrel.data.ground_truth`).
        detections: Detections for ``image_ids``.
        image_ids: Evaluation images, in the order the weight columns refer to.
        weights: ``(R, n_images)`` bootstrap weights shared by all conditions.
        corruption: Corruption name or ``clean``.
        severity: Severity (0 = clean).
        temperature: If given, also compute D-ECE after temperature scaling, on the same
            detections (selected by their original score; scaling preserves TP/FP labels).

    Returns:
        The condition's :class:`ConditionResult`.
    """
    res = ConditionResult(corruption, severity)
    sc = scopes(cfg, coco)
    n_bins = cfg["metrics"]["dece_bins"]

    # Accuracy. Bootstrap only over the driving classes (see DECISIONS D51).
    ev = MapEvaluator(coco, detections, image_ids, sc[DRIVING])
    ap_point = ev.per_category_ap()
    ap_samples = [ev.per_category_ap(w) for w in weights]
    for name, cats in sc.items():
        if name == ALL_CLASSES:
            m = coco_map(coco, detections, image_ids, cats)
            res.points |= {(name, "mAP"): m["mAP"], (name, "mAP50"): m["mAP50"]}
            continue
        point = summarize_ap(ap_point, ev.category_ids, cats)
        draws = [summarize_ap(a, ev.category_ids, cats) for a in ap_samples]
        for k in ("mAP", "mAP50"):
            res.points[(name, k)] = point[k]
            res.samples[(name, k)] = np.array([d[k] for d in draws])

    # Calibration.
    pooled = match_detections(detections, gt, cfg["matching"]["iou_threshold"])
    gt_labels = np.concatenate(
        [gt[i].labels[~gt[i].iscrowd] for i in image_ids] or [np.zeros(0, np.int64)]
    )
    for name, cats in sc.items():
        res.counts[(name, "n_gt")] = int(np.isin(gt_labels, cats).sum())
        for thr in cfg["metrics"]["dece_score_thresholds"]:
            sel = select_detections(pooled, thr, cats)
            key = f"dece_t{thr:g}"
            r = dece(sel.score, sel.tp, n_bins)
            res.points[(name, key)] = r.value
            res.samples[(name, key)] = dece_resamples(
                sel.score, sel.tp, image_index(sel, image_ids), n_bins, weights
            )
            res.counts[(name, f"n_det_t{thr:g}")] = len(sel)
            res.reliability[(name, thr, False)] = r
            if temperature is not None:
                cal = apply_temperature(sel.score, temperature, cfg["calibration"]["logit_eps"])
                rc = dece(cal, sel.tp, n_bins)
                res.points[(name, f"dece_cal_t{thr:g}")] = rc.value
                res.samples[(name, f"dece_cal_t{thr:g}")] = dece_resamples(
                    cal, sel.tp, image_index(sel, image_ids), n_bins, weights
                )
                res.reliability[(name, thr, True)] = rc
    return res


def summary_rows(
    cfg: dict[str, Any], result: ConditionResult, clean: ConditionResult
) -> list[dict[str, Any]]:
    """Rows for ``summary.csv``: one per scope, with CIs and paired deltas vs clean."""
    level = cfg["metrics"]["bootstrap"]["confidence_level"]
    min_n = cfg["data"]["min_instances_for_claim"]
    scope_names = list(dict.fromkeys(s for s, _ in result.points))
    rows = []
    for s in scope_names:
        row: dict[str, Any] = {
            "corruption": result.corruption,
            "severity": result.severity,
            "scope": s,
            "n_gt": result.counts[(s, "n_gt")],
            "sufficient": result.counts[(s, "n_gt")] >= min_n,
        }
        calibrated = any(k[1].startswith("dece_cal_") for k in result.points)
        for m in metric_names(cfg, calibrated):
            point, samples = result.points[(s, m)], result.samples.get((s, m))
            ci = percentile_ci(point, samples, level) if samples is not None else None
            row[m] = point
            row[f"{m}_lo"] = ci.low if ci else math.nan
            row[f"{m}_hi"] = ci.high if ci else math.nan
            d_point = point - clean.points[(s, m)]
            base = clean.samples.get((s, m))
            d_ci = (
                percentile_ci(d_point, samples - base, level) if ci and base is not None else None
            )
            row[f"d_{m}"] = d_point
            row[f"d_{m}_lo"] = d_ci.low if d_ci else math.nan
            row[f"d_{m}_hi"] = d_ci.high if d_ci else math.nan
        if calibrated:
            # Improvement from scaling within this condition (before - after), paired CI.
            for t in cfg["metrics"]["dece_score_thresholds"]:
                b, a = f"dece_t{t:g}", f"dece_cal_t{t:g}"
                gain = percentile_ci(
                    result.points[(s, b)] - result.points[(s, a)],
                    result.samples[(s, b)] - result.samples[(s, a)],
                    level,
                )
                row[f"cal_gain_t{t:g}"] = gain.value
                row[f"cal_gain_t{t:g}_lo"] = gain.low
                row[f"cal_gain_t{t:g}_hi"] = gain.high
        for t in cfg["metrics"]["dece_score_thresholds"]:
            row[f"n_det_t{t:g}"] = result.counts[(s, f"n_det_t{t:g}")]
        rows.append(row)
    return rows


def reliability_rows(result: ConditionResult) -> list[dict[str, Any]]:
    """Rows for ``reliability.csv``: per scope, threshold and confidence bin."""
    rows = []
    for (scope, thr, calibrated), r in result.reliability.items():
        for k in range(len(r.count)):
            rows.append(
                {
                    "corruption": result.corruption,
                    "severity": result.severity,
                    "scope": scope,
                    "score_threshold": thr,
                    "calibrated": calibrated,
                    "bin": k,
                    "bin_low": r.bin_edges[k],
                    "bin_high": r.bin_edges[k + 1],
                    "count": r.count[k],
                    "precision": r.precision[k],
                    "confidence": r.confidence[k],
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows (dicts with identical keys) to CSV; floats with 6 significant digits."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: Any) -> Any:
        return f"{v:.6g}" if isinstance(v, float | np.floating) else v

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})
