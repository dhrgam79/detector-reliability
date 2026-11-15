"""Accuracy metrics (COCO mAP). D-ECE is added in milestone step 6."""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterable, Sequence
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detrel.model import ImageDetections

logger = logging.getLogger(__name__)


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


def coco_map(
    coco: dict[str, Any] | COCO,
    detections: Sequence[ImageDetections],
    image_ids: Iterable[int],
    category_ids: Iterable[int],
) -> dict[str, float]:
    """COCO box mAP restricted to given images and categories.

    Uses ``pycocotools`` unchanged, so crowd regions, area ranges and the 101-point
    interpolation follow the official protocol. Classes without GT in ``image_ids`` are
    excluded from the mean (pycocotools convention).

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
    wanted_imgs, wanted_cats = set(img_ids), set(cat_ids)

    results = [
        {
            "image_id": d.image_id,
            "category_id": int(label),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        }
        for d in detections
        if d.image_id in wanted_imgs
        for (x1, y1, x2, y2), label, score in zip(d.boxes, d.labels, d.scores, strict=True)
        if int(label) in wanted_cats
    ]
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
