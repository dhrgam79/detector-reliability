import math

import numpy as np
import pytest

from detrel.metrics import coco_map
from detrel.model import ImageDetections

CAR, PERSON = 3, 1


def gt_coco():
    """Two images; image 1 has a car at (10,10)-(50,50), image 2 a car and a crowd person."""
    images = [{"id": 1, "width": 100, "height": 100}, {"id": 2, "width": 100, "height": 100}]
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": CAR,
            "bbox": [10, 10, 40, 40],
            "area": 1600,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": CAR,
            "bbox": [20, 20, 40, 40],
            "area": 1600,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": PERSON,
            "bbox": [60, 0, 30, 90],
            "area": 2700,
            "iscrowd": 1,
        },
    ]
    cats = [{"id": CAR, "name": "car"}, {"id": PERSON, "name": "person"}]
    return {"images": images, "annotations": anns, "categories": cats}


def det(image_id, boxes, labels, scores):
    return ImageDetections(
        image_id=image_id,
        boxes=np.array(boxes, np.float32).reshape(-1, 4),
        labels=np.array(labels, np.int64),
        scores=np.array(scores, np.float32),
    )


PERFECT = [
    det(1, [[10, 10, 50, 50]], [CAR], [0.9]),
    det(2, [[20, 20, 60, 60]], [CAR], [0.8]),
]


def test_perfect_detections_give_map_one():
    r = coco_map(gt_coco(), PERFECT, [1, 2], [CAR])
    assert r == pytest.approx({"mAP": 1.0, "mAP50": 1.0})


def test_low_score_false_positive_ranked_last_keeps_ap_one():
    dets = [PERFECT[0], det(2, [[20, 20, 60, 60], [70, 70, 90, 90]], [CAR, CAR], [0.8, 0.1])]
    assert coco_map(gt_coco(), dets, [1, 2], [CAR])["mAP"] == pytest.approx(1.0)


def test_loose_box_passes_iou50_but_not_strict_map():
    # IoU with GT = 1600/2500 = 0.64: TP at 0.5 but FP at >= 0.65.
    dets = [det(1, [[10, 10, 60, 60]], [CAR], [0.9]), PERFECT[1]]
    r = coco_map(gt_coco(), dets, [1, 2], [CAR])
    assert r["mAP50"] == pytest.approx(1.0)
    assert r["mAP"] < 1.0


def test_wrong_class_is_not_a_match():
    dets = [det(1, [[10, 10, 50, 50]], [PERSON], [0.9]), PERFECT[1]]
    r = coco_map(gt_coco(), dets, [1, 2], [CAR])
    assert r["mAP50"] == pytest.approx(0.5, abs=0.01)  # one of two cars found


def test_detection_on_crowd_region_is_ignored():
    # pycocotools ignores detections matched to crowd GT: person class has no non-crowd GT.
    dets = PERFECT + [det(2, [[60, 0, 90, 90]], [PERSON], [0.95])]
    assert coco_map(gt_coco(), dets, [1, 2], [CAR, PERSON])["mAP"] == pytest.approx(1.0)


def test_restricting_images_and_categories():
    assert coco_map(gt_coco(), PERFECT[:1], [1], [CAR])["mAP"] == pytest.approx(1.0)


def test_no_detections_is_zero_and_no_gt_is_nan():
    assert coco_map(gt_coco(), [], [1, 2], [CAR]) == {"mAP": 0.0, "mAP50": 0.0}
    assert math.isnan(coco_map(gt_coco(), [], [1, 2], [PERSON])["mAP"])  # crowd-only GT
