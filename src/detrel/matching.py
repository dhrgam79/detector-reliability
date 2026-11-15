"""Box IoU and greedy detection-to-ground-truth matching (TP / FP / ignored labels).

Matching follows the COCO protocol at a single IoU threshold, so D-ECE and mAP judge the
same detections the same way:

* detections are processed in descending score order (stable for ties);
* a detection may match only GT of its own class with IoU >= threshold;
* among eligible GT it takes the unmatched **non-crowd** box with the highest IoU (TP);
* if no non-crowd GT qualifies, it may match a **crowd** region, using
  intersection-over-detection-area as the overlap; it is then *ignored* (neither TP nor FP).
  Crowd regions can absorb any number of detections;
* everything else is a false positive.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from detrel.data import GroundTruth
from detrel.model import ImageDetections


@dataclass(frozen=True)
class ImageMatch:
    """Matching result for one image, aligned with its detections.

    Attributes:
        tp: ``bool`` ``(N,)``; True for true positives.
        ignored: ``bool`` ``(N,)``; True for detections matched to a crowd region.
        gt_index: ``int64`` ``(N,)``; matched GT index (non-crowd or crowd), ``-1`` if none.
    """

    tp: np.ndarray
    ignored: np.ndarray
    gt_index: np.ndarray


@dataclass(frozen=True)
class PooledMatches:
    """Per-detection labels pooled over images (non-ignored detections only).

    Attributes:
        image_id: ``int64`` ``(D,)``; source image of each detection (for bootstrapping).
        label: ``int64`` ``(D,)``; predicted COCO category id.
        score: ``float64`` ``(D,)``; confidence.
        tp: ``bool`` ``(D,)``; True positive (1) or false positive (0).
    """

    image_id: np.ndarray
    label: np.ndarray
    score: np.ndarray
    tp: np.ndarray

    def __len__(self) -> int:
        """Number of pooled detections."""
        return len(self.score)


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of ``xyxy`` boxes; degenerate boxes have area 0."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def box_iou(a: np.ndarray, b: np.ndarray, b_iscrowd: np.ndarray | None = None) -> np.ndarray:
    """Pairwise overlap between boxes ``a`` (N, 4) and ``b`` (M, 4), ``xyxy``.

    Args:
        a: Detection boxes.
        b: Ground-truth boxes.
        b_iscrowd: Optional ``bool`` (M,). For crowd columns the overlap is
            ``intersection / area(a)`` (COCO convention), otherwise standard IoU.

    Returns:
        ``float64`` array (N, M). Pairs with zero denominator get 0.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a, area_b = box_area(a)[:, None], box_area(b)[None, :]
    denom = area_a + area_b - inter
    if b_iscrowd is not None and len(b):
        denom = np.where(np.asarray(b_iscrowd, dtype=bool)[None, :], area_a, denom)
    return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)


def match_image(
    det_boxes: np.ndarray,
    det_labels: np.ndarray,
    det_scores: np.ndarray,
    gt: GroundTruth,
    iou_threshold: float,
) -> ImageMatch:
    """Greedily match one image's detections to its ground truth.

    Args:
        det_boxes: ``(N, 4)`` ``xyxy``.
        det_labels: ``(N,)`` category ids.
        det_scores: ``(N,)`` confidences.
        gt: Ground truth of the image.
        iou_threshold: Minimum overlap for a match (inclusive).

    Returns:
        :class:`ImageMatch` aligned with the input detection order.
    """
    n = len(det_scores)
    tp = np.zeros(n, dtype=bool)
    ignored = np.zeros(n, dtype=bool)
    gt_index = np.full(n, -1, dtype=np.int64)
    if n == 0 or len(gt.labels) == 0:
        return ImageMatch(tp, ignored, gt_index)

    overlap = box_iou(det_boxes, gt.boxes, gt.iscrowd)
    same_class = np.asarray(det_labels)[:, None] == gt.labels[None, :]
    overlap = np.where(same_class & (overlap >= iou_threshold), overlap, -1.0)

    taken = np.zeros(len(gt.labels), dtype=bool)
    for d in np.argsort(-np.asarray(det_scores), kind="stable"):
        regular = np.where(~gt.iscrowd & ~taken, overlap[d], -1.0)
        g = int(np.argmax(regular))
        if regular[g] >= 0:
            taken[g] = True
            tp[d], gt_index[d] = True, g
            continue
        crowd = np.where(gt.iscrowd, overlap[d], -1.0)
        g = int(np.argmax(crowd))
        if crowd[g] >= 0:
            ignored[d], gt_index[d] = True, g
    return ImageMatch(tp, ignored, gt_index)


def match_detections(
    detections: Sequence[ImageDetections],
    ground_truth: Mapping[int, GroundTruth],
    iou_threshold: float,
    class_ids: Iterable[int] | None = None,
) -> PooledMatches:
    """Match every image and pool non-ignored detections.

    Restricting to ``class_ids`` happens after matching; because matching is per class, this
    is identical to matching with only those classes present.

    Args:
        detections: Per-image detections.
        ground_truth: Mapping image id -> ground truth; every detection image must be present.
        iou_threshold: Minimum overlap for a match.
        class_ids: Keep only detections predicted as these classes; ``None`` keeps all.

    Returns:
        Pooled per-detection arrays.
    """
    keep_classes = None if class_ids is None else np.array(sorted(set(class_ids)))
    parts: list[tuple[np.ndarray, ...]] = []
    for det in detections:
        m = match_image(
            det.boxes, det.labels, det.scores, ground_truth[det.image_id], iou_threshold
        )
        keep = ~m.ignored
        if keep_classes is not None:
            keep &= np.isin(det.labels, keep_classes)
        parts.append(
            (
                np.full(int(keep.sum()), det.image_id, dtype=np.int64),
                det.labels[keep].astype(np.int64),
                det.scores[keep].astype(np.float64),
                m.tp[keep],
            )
        )
    if not parts:
        empty = np.zeros(0)
        return PooledMatches(
            empty.astype(np.int64), empty.astype(np.int64), empty, empty.astype(bool)
        )
    image_id, label, score, tp = (np.concatenate(cols) for cols in zip(*parts, strict=True))
    return PooledMatches(image_id=image_id, label=label, score=score, tp=tp)
