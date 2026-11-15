"""Accuracy (COCO mAP) and calibration (D-ECE) metrics, with image-level bootstrap support.

Bootstrap resamples are represented as **image weights** (how many times each image was drawn),
so every statistic here accepts an optional weight vector instead of materialising copies.
Using the same weights for every test condition makes condition comparisons *paired*.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detrel.matching import PooledMatches
from detrel.model import ImageDetections

logger = logging.getLogger(__name__)

_EPS = np.spacing(1)  # same epsilon pycocotools uses in precision


# ============================================================================ COCO mAP


def _coco_from_dict(coco: dict[str, Any]) -> COCO:
    # pycocotools stores the matched GT *id* per detection and treats 0 as "unmatched", so a
    # GT annotation with id 0 silently becomes unmatchable and mAP is wrong. Real COCO ids
    # start at 1; refuse anything else rather than return a corrupted number.
    if any(a["id"] <= 0 for a in coco["annotations"]):
        raise ValueError("annotation ids must be positive (pycocotools treats id 0 as unmatched)")
    gt = COCO()
    gt.dataset = coco
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def _to_coco_results(
    detections: Sequence[ImageDetections], image_ids: set[int], category_ids: set[int]
) -> list[dict[str, Any]]:
    return [
        {
            "image_id": d.image_id,
            "category_id": int(label),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        }
        for d in detections
        if d.image_id in image_ids
        for (x1, y1, x2, y2), label, score in zip(d.boxes, d.labels, d.scores, strict=True)
        if int(label) in category_ids
    ]


def coco_map(
    coco: dict[str, Any] | COCO,
    detections: Sequence[ImageDetections],
    image_ids: Iterable[int],
    category_ids: Iterable[int],
) -> dict[str, float]:
    """COCO box mAP restricted to given images and categories (official pycocotools path).

    Classes without GT in ``image_ids`` are excluded from the mean (pycocotools convention).

    Args:
        coco: Ground truth, as a parsed instances dict or a ``COCO`` object.
        detections: Detections; those for other images or categories are ignored.
        image_ids: Images to evaluate.
        category_ids: Categories to average over.

    Returns:
        ``{"mAP": mAP@[.5:.95], "mAP50": mAP@0.5}`` in ``[0, 1]``; ``0.0`` if there are no
        detections, ``nan`` if no requested category has ground truth.
    """
    gt = coco if isinstance(coco, COCO) else _coco_from_dict(coco)
    img_ids, cat_ids = sorted(set(image_ids)), sorted(set(category_ids))
    results = _to_coco_results(detections, set(img_ids), set(cat_ids))
    if not results:
        has_gt = bool(gt.getAnnIds(imgIds=img_ids, catIds=cat_ids, iscrowd=False))
        value = 0.0 if has_gt else float("nan")
        return {"mAP": value, "mAP50": value}

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.imgIds = img_ids
        ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # pycocotools reports -1 when no category has GT; that is "undefined", not zero.
    def defined(x: float) -> float:
        return float(x) if x >= 0 else float("nan")

    return {"mAP": defined(ev.stats[0]), "mAP50": defined(ev.stats[1])}


@dataclass(frozen=True)
class _CategoryData:
    """Score-sorted detections of one category with their per-IoU-threshold outcomes."""

    image_index: np.ndarray  # (D,) index into the evaluator's image list
    tps: np.ndarray  # (T, D) bool
    fps: np.ndarray  # (T, D) bool
    n_gt: np.ndarray  # (n_images,) non-ignored GT count per image


class MapEvaluator:
    """COCO mAP that can be recomputed cheaply under image weights (for bootstrapping).

    pycocotools' per-image matching (``evaluate``) runs once; ``compute`` then reimplements
    ``accumulate`` (101-point interpolated precision, all IoU thresholds, area "all",
    maxDets 100) with weights. With unit weights it reproduces pycocotools exactly; with
    integer weights it equals pycocotools on a dataset where images are duplicated.
    """

    def __init__(
        self,
        coco: dict[str, Any] | COCO,
        detections: Sequence[ImageDetections],
        image_ids: Sequence[int],
        category_ids: Iterable[int],
    ):
        """Run pycocotools matching once.

        Args:
            coco: Ground truth.
            detections: Detections for (at least) ``image_ids``.
            image_ids: Images, in the order bootstrap weight vectors refer to.
            category_ids: Categories to evaluate (per-category AP is kept).
        """
        gt = coco if isinstance(coco, COCO) else _coco_from_dict(coco)
        self.image_ids = list(image_ids)
        self.category_ids = sorted(set(category_ids))
        n_img = len(self.image_ids)
        results = _to_coco_results(detections, set(self.image_ids), set(self.category_ids))

        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(results) if results else COCO(), iouType="bbox")
            ev.params.imgIds = self.image_ids
            ev.params.catIds = self.category_ids
            ev.params.areaRng = [ev.params.areaRng[0]]  # "all" only
            ev.params.areaRngLbl = ["all"]
            ev.params.maxDets = [100]
            if results:
                ev.evaluate()
        self.iou_thresholds = ev.params.iouThrs
        self.recall_thresholds = ev.params.recThrs

        self._cats: list[_CategoryData] = []
        for k, cat in enumerate(self.category_ids):
            n_gt = np.zeros(n_img)
            scores, img_idx, matched, ignored = [], [], [], []
            for i, img in enumerate(self.image_ids):
                e = ev.evalImgs[k * n_img + i] if results else None
                if e is None:  # no detections and no GT for this (image, category) in COCOeval
                    if not results:
                        n_gt[i] = len(gt.getAnnIds(imgIds=[img], catIds=[cat], iscrowd=False))
                    continue
                n_gt[i] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
                s = np.asarray(e["dtScores"][:100], dtype=np.float64)
                scores.append(s)
                img_idx.append(np.full(len(s), i, dtype=np.int64))
                matched.append(np.asarray(e["dtMatches"])[:, :100] > 0)
                ignored.append(np.asarray(e["dtIgnore"])[:, :100].astype(bool))
            n_t = len(self.iou_thresholds)
            if scores:
                s = np.concatenate(scores)
                order = np.argsort(-s, kind="mergesort")  # identical to pycocotools
                m = np.concatenate(matched, axis=1)[:, order]
                ig = np.concatenate(ignored, axis=1)[:, order]
                idx = np.concatenate(img_idx)[order]
            else:
                m = ig = np.zeros((n_t, 0), dtype=bool)
                idx = np.zeros(0, dtype=np.int64)
            self._cats.append(_CategoryData(idx, m & ~ig, ~m & ~ig, n_gt))

    def per_category_ap(self, weights: np.ndarray | None = None) -> np.ndarray:
        """AP per category and IoU threshold.

        Args:
            weights: Image weights ``(n_images,)``; ``None`` means all ones.

        Returns:
            ``(n_categories, n_iou_thresholds)``; ``nan`` rows for categories without
            (weighted) non-ignored GT.
        """
        w_img = np.ones(len(self.image_ids)) if weights is None else np.asarray(weights, float)
        out = np.full((len(self._cats), len(self.iou_thresholds)), np.nan)
        for k, c in enumerate(self._cats):
            npig = float(w_img @ c.n_gt)
            if npig <= 0:
                continue
            w = w_img[c.image_index]
            nd = len(w)
            if nd == 0:
                out[k] = 0.0
                continue
            tp = np.cumsum(c.tps * w, axis=1)
            fp = np.cumsum(c.fps * w, axis=1)
            rc = tp / npig
            pr = tp / (fp + tp + _EPS)
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]  # precision envelope
            for t in range(pr.shape[0]):
                inds = np.searchsorted(rc[t], self.recall_thresholds, side="left")
                q = np.where(inds < nd, pr[t, np.minimum(inds, nd - 1)], 0.0)
                out[k, t] = q.mean()
        return out

    def compute(
        self, weights: np.ndarray | None = None, category_ids: Iterable[int] | None = None
    ) -> dict[str, float]:
        """Compute mAP over (a subset of) the evaluator's categories.

        Args:
            weights: Image weights; ``None`` means all ones.
            category_ids: Subset to average over; ``None`` uses all.

        Returns:
            ``{"mAP": ..., "mAP50": ...}``; ``nan`` if no category has GT.
        """
        return summarize_ap(self.per_category_ap(weights), self.category_ids, category_ids)


def summarize_ap(
    ap: np.ndarray, all_category_ids: Sequence[int], category_ids: Iterable[int] | None = None
) -> dict[str, float]:
    """Average a per-category AP matrix (from :meth:`MapEvaluator.per_category_ap`)."""
    if category_ids is not None:
        wanted = set(category_ids)
        ap = ap[[i for i, c in enumerate(all_category_ids) if c in wanted]]
    valid = ~np.isnan(ap[:, 0])
    if not valid.any():
        return {"mAP": float("nan"), "mAP50": float("nan")}
    return {"mAP": float(ap[valid].mean()), "mAP50": float(ap[valid, 0].mean())}


# ============================================================================== D-ECE


@dataclass(frozen=True)
class DeceResult:
    """D-ECE value and the per-bin statistics behind it (for reliability diagrams).

    Attributes:
        value: D-ECE in ``[0, 1]``; ``nan`` if there are no detections.
        bin_edges: ``(K + 1,)`` equal-width edges over ``[0, 1]``.
        count: ``(K,)`` (weighted) detections per bin.
        precision: ``(K,)`` fraction of TP per bin; ``nan`` for empty bins.
        confidence: ``(K,)`` mean score per bin; ``nan`` for empty bins.
    """

    value: float
    bin_edges: np.ndarray
    count: np.ndarray
    precision: np.ndarray
    confidence: np.ndarray


def dece(
    scores: np.ndarray,
    tp: np.ndarray,
    n_bins: int,
    weights: np.ndarray | None = None,
) -> DeceResult:
    """Detection Expected Calibration Error (Küppers et al. 2020, confidence-only).

    ``D-ECE = sum_k (n_k / N) * |prec(k) - conf(k)|`` over ``n_bins`` equal-width confidence
    bins on ``[0, 1]`` (a score of exactly 1.0 falls in the last bin). Equivalently
    ``sum_k |sum_{i in k} w_i (tp_i - s_i)| / sum_i w_i``, which is how it is computed.

    Args:
        scores: ``(D,)`` confidences in ``[0, 1]``.
        tp: ``(D,)`` bool, True positive labels.
        n_bins: Number of bins K.
        weights: Optional ``(D,)`` non-negative per-detection weights (bootstrap).

    Returns:
        :class:`DeceResult`.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(tp, dtype=np.float64)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip((s * n_bins).astype(np.int64), 0, n_bins - 1)
    count = np.bincount(b, weights=w, minlength=n_bins)
    sum_tp = np.bincount(b, weights=w * y, minlength=n_bins)
    sum_s = np.bincount(b, weights=w * s, minlength=n_bins)
    total = count.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(count > 0, sum_tp / count, np.nan)
        confidence = np.where(count > 0, sum_s / count, np.nan)
    value = float(np.abs(sum_tp - sum_s).sum() / total) if total > 0 else float("nan")
    return DeceResult(value, edges, count, precision, confidence)


def dece_resamples(
    scores: np.ndarray,
    tp: np.ndarray,
    image_idx: np.ndarray,
    n_bins: int,
    weights: np.ndarray,
) -> np.ndarray:
    """D-ECE under many image-weight resamples at once.

    Per-image, per-bin sums of ``tp`` and ``score`` are computed once; each resample is then a
    matrix product, so 1000 resamples cost about as much as one plain D-ECE.

    Args:
        scores: ``(D,)`` confidences.
        tp: ``(D,)`` TP labels.
        image_idx: ``(D,)`` index of each detection's image into the weight columns.
        n_bins: Number of bins K.
        weights: ``(R, n_images)`` image weights.

    Returns:
        ``(R,)`` D-ECE values; ``nan`` where a resample contains no detections.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(tp, dtype=np.float64)
    n_images = weights.shape[1]
    b = np.clip((s * n_bins).astype(np.int64), 0, n_bins - 1)
    flat = np.asarray(image_idx, dtype=np.int64) * n_bins + b
    size = n_images * n_bins
    diff = np.bincount(flat, weights=y - s, minlength=size).reshape(n_images, n_bins)
    count = np.bincount(np.asarray(image_idx, dtype=np.int64), minlength=n_images).astype(float)
    w = np.asarray(weights, dtype=np.float64)
    total = w @ count
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, np.abs(w @ diff).sum(axis=1) / total, np.nan)


def select_detections(
    pooled: PooledMatches,
    score_threshold: float = 0.0,
    class_ids: Iterable[int] | None = None,
) -> PooledMatches:
    """Keep pooled detections with ``score >= score_threshold`` and label in ``class_ids``."""
    keep = pooled.score >= score_threshold
    if class_ids is not None:
        keep &= np.isin(pooled.label, np.array(sorted(set(class_ids))))
    return PooledMatches(
        pooled.image_id[keep], pooled.label[keep], pooled.score[keep], pooled.tp[keep]
    )


def image_index(pooled: PooledMatches, image_ids: Sequence[int]) -> np.ndarray:
    """Position of each pooled detection's image in ``image_ids`` (for weight lookup)."""
    ids = np.asarray(image_ids)
    order = np.argsort(ids)
    pos = np.clip(np.searchsorted(ids, pooled.image_id, sorter=order), 0, max(len(ids) - 1, 0))
    if len(pooled) and (len(ids) == 0 or not np.array_equal(ids[order[pos]], pooled.image_id)):
        raise KeyError("pooled detections reference images outside image_ids")
    return order[pos]


# ========================================================================== bootstrap


def bootstrap_weights(n_images: int, n_resamples: int, seed: int) -> np.ndarray:
    """Image-level bootstrap resamples as integer weights.

    Each row is a multinomial draw of ``n_images`` images with replacement. Depends only on
    ``(n_images, n_resamples, seed)``, so reusing it across conditions pairs the comparison.

    Returns:
        ``int64`` ``(n_resamples, n_images)``; each row sums to ``n_images``.
    """
    rng = np.random.RandomState(seed)
    return rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=n_resamples)


@dataclass(frozen=True)
class Estimate:
    """Point estimate with a percentile bootstrap confidence interval."""

    value: float
    low: float
    high: float


def bootstrap_ci(
    statistic: Callable[[np.ndarray | None], float],
    weights: np.ndarray,
    confidence_level: float,
) -> Estimate:
    """Percentile bootstrap CI of ``statistic``.

    Args:
        statistic: Function of an image-weight vector (``None`` = unweighted point estimate).
        weights: ``(R, n_images)`` resamples from :func:`bootstrap_weights`.
        confidence_level: E.g. 0.95.

    Returns:
        :class:`Estimate`. Resamples where the statistic is undefined (``nan``) are dropped.
    """
    samples = np.array([statistic(w) for w in weights], dtype=np.float64)
    return percentile_ci(statistic(None), samples, confidence_level)


def percentile_ci(point: float, samples: np.ndarray, confidence_level: float) -> Estimate:
    """Percentile CI from precomputed bootstrap samples (``nan`` samples are dropped)."""
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return Estimate(float(point), float("nan"), float("nan"))
    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return Estimate(float(point), float(lo), float(hi))
