import contextlib
import io

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detrel.data import GroundTruth
from detrel.matching import box_area, box_iou, match_detections, match_image
from detrel.model import ImageDetections

CAR, PERSON = 3, 1


def gt(boxes, labels, crowd=None):
    boxes = np.array(boxes, np.float32).reshape(-1, 4)
    crowd = np.zeros(len(boxes), bool) if crowd is None else np.array(crowd, bool)
    return GroundTruth(boxes=boxes, labels=np.array(labels, np.int64), iscrowd=crowd)


def match(det_boxes, det_labels, det_scores, g, thr=0.5):
    return match_image(
        np.array(det_boxes, np.float32).reshape(-1, 4),
        np.array(det_labels, np.int64),
        np.array(det_scores, np.float32),
        g,
        thr,
    )


# ------------------------------------------------------------------------------- IoU


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),
        ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),
        ([0, 0, 10, 10], [10, 0, 20, 10], 0.0),  # touching edges
        ([0, 0, 10, 10], [5, 0, 15, 10], 50 / 150),  # half overlap
        ([0, 0, 10, 10], [0, 0, 5, 10], 0.5),  # containment
        ([0, 0, 10, 10], [2, 2, 12, 12], 64 / 136),
        ([0, 0, 0, 10], [0, 0, 10, 10], 0.0),  # degenerate box
    ],
)
def test_box_iou_known_values(a, b, expected):
    assert box_iou(np.array([a]), np.array([b]))[0, 0] == pytest.approx(expected)
    assert box_iou(np.array([b]), np.array([a]))[0, 0] == pytest.approx(expected)  # symmetric


def test_box_iou_shapes_and_empty():
    a = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [5, 5, 6, 6]])
    b = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
    assert box_iou(a, b).shape == (3, 2)
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 2)
    assert box_iou(a, np.zeros((0, 4))).shape == (3, 0)


def test_crowd_overlap_is_intersection_over_detection_area():
    det = np.array([[10, 10, 20, 20]])  # area 100, fully inside the crowd region
    crowd = np.array([[0, 0, 100, 100]])  # area 10000
    assert box_iou(det, crowd)[0, 0] == pytest.approx(0.01)
    assert box_iou(det, crowd, np.array([True]))[0, 0] == pytest.approx(1.0)


def test_box_area():
    np.testing.assert_allclose(box_area(np.array([[0, 0, 2, 3], [5, 5, 4, 9]])), [6, 0])


# -------------------------------------------------------------------------- matching


def test_single_true_positive():
    m = match([[0, 0, 10, 10]], [CAR], [0.9], gt([[0, 0, 10, 10]], [CAR]))
    assert m.tp.tolist() == [True] and m.ignored.tolist() == [False] and m.gt_index.tolist() == [0]


def test_threshold_is_inclusive():
    g = gt([[0, 0, 10, 10]], [CAR])
    assert match([[0, 0, 5, 10]], [CAR], [0.9], g).tp.tolist() == [True]  # IoU exactly 0.5
    assert match([[0, 0, 4.9, 10]], [CAR], [0.9], g).tp.tolist() == [False]


def test_wrong_class_is_false_positive():
    m = match([[0, 0, 10, 10]], [PERSON], [0.9], gt([[0, 0, 10, 10]], [CAR]))
    assert m.tp.tolist() == [False] and m.gt_index.tolist() == [-1]


def test_duplicate_detection_higher_score_wins():
    m = match([[0, 0, 10, 10], [0, 0, 10, 10]], [CAR, CAR], [0.3, 0.8], gt([[0, 0, 10, 10]], [CAR]))
    assert m.tp.tolist() == [False, True]  # input order preserved; the 0.8 detection is the TP


def test_greedy_by_score_not_by_iou():
    # The higher-score detection takes the GT even though the other overlaps it better.
    g = gt([[0, 0, 10, 10]], [CAR])
    m = match([[0, 0, 10, 6], [0, 0, 10, 10]], [CAR, CAR], [0.9, 0.5], g)
    assert m.tp.tolist() == [True, False]


def test_detection_takes_best_iou_among_unmatched_gt():
    g = gt([[0, 0, 10, 10], [2, 0, 12, 10]], [CAR, CAR])
    m = match([[2, 0, 12, 10]], [CAR], [0.9], g)
    assert m.gt_index.tolist() == [1]


def test_two_detections_two_gts():
    g = gt([[0, 0, 10, 10], [50, 50, 60, 60]], [CAR, CAR])
    m = match([[50, 50, 60, 60], [0, 0, 10, 10]], [CAR, CAR], [0.6, 0.7], g)
    assert m.tp.tolist() == [True, True] and m.gt_index.tolist() == [1, 0]


def test_tie_scores_are_deterministic():
    g = gt([[0, 0, 10, 10]], [CAR])
    m = match([[0, 0, 10, 10], [0, 0, 10, 10]], [CAR, CAR], [0.5, 0.5], g)
    assert m.tp.tolist() == [True, False]  # stable: first in input order wins


def test_crowd_only_match_is_ignored_and_reusable():
    g = gt([[0, 0, 100, 100]], [PERSON], crowd=[True])
    m = match([[10, 10, 20, 20], [30, 30, 40, 40]], [PERSON, PERSON], [0.9, 0.8], g)
    assert m.ignored.tolist() == [True, True] and not m.tp.any()


def test_regular_gt_preferred_over_crowd():
    g = gt([[0, 0, 100, 100], [10, 10, 20, 20]], [PERSON, PERSON], crowd=[True, False])
    m = match([[10, 10, 20, 20], [11, 11, 21, 21]], [PERSON, PERSON], [0.9, 0.8], g)
    assert m.tp.tolist() == [True, False]
    assert m.ignored.tolist() == [False, True]  # second falls back to the crowd region


def test_no_gt_all_false_positive_and_no_detections_empty():
    m = match([[0, 0, 1, 1]], [CAR], [0.9], gt([], []))
    assert m.tp.tolist() == [False] and m.ignored.tolist() == [False]
    assert len(match([], [], [], gt([[0, 0, 1, 1]], [CAR])).tp) == 0


# ---------------------------------------------------------------------------- pooling


def det(image_id, boxes, labels, scores):
    return ImageDetections(
        image_id=image_id,
        boxes=np.array(boxes, np.float32).reshape(-1, 4),
        labels=np.array(labels, np.int64),
        scores=np.array(scores, np.float32),
    )


def test_match_detections_pools_and_filters():
    gts = {
        1: gt([[0, 0, 10, 10], [0, 0, 100, 100]], [CAR, PERSON], crowd=[False, True]),
        2: gt([[0, 0, 10, 10]], [PERSON]),
    }
    dets = [
        det(
            1,
            [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]],
            [CAR, CAR, PERSON],
            [0.9, 0.4, 0.7],
        ),
        det(2, [[0, 0, 10, 10]], [PERSON], [0.6]),
    ]
    pooled = match_detections(dets, gts, 0.5)
    # image 1: car TP, car FP, person ignored (inside crowd); image 2: person TP
    assert pooled.image_id.tolist() == [1, 1, 2]
    assert pooled.tp.tolist() == [True, False, True]
    np.testing.assert_allclose(pooled.score, [0.9, 0.4, 0.6], rtol=1e-6)

    only_person = match_detections(dets, gts, 0.5, class_ids=[PERSON])
    assert only_person.label.tolist() == [PERSON] and only_person.tp.tolist() == [True]
    assert len(match_detections([], gts, 0.5)) == 0


# ----------------------------------------------- agreement with pycocotools (reference)


def _random_scene(rng, n_images=40):
    images, anns, dets, aid = (
        [],
        [],
        [],
        1,
    )  # COCO ann ids start at 1 (pycocotools treats 0 as "unmatched")
    for img_id in range(1, n_images + 1):
        images.append({"id": img_id, "width": 200, "height": 200})
        n_gt = rng.randint(0, 6)
        gt_boxes = []
        for _ in range(n_gt):
            x, y = rng.uniform(0, 150, 2)
            w, h = rng.uniform(10, 50, 2)
            cat = int(rng.choice([CAR, PERSON]))
            crowd = int(rng.rand() < 0.15)
            anns.append(
                {
                    "id": aid,
                    "image_id": img_id,
                    "category_id": cat,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": crowd,
                }
            )
            gt_boxes.append((x, y, w, h, cat))
            aid += 1
        boxes, labels, scores = [], [], []
        for x, y, w, h, cat in gt_boxes:  # jittered copies of GT, some duplicated
            for _ in range(rng.randint(0, 3)):
                j = rng.normal(0, 6, 4)
                x1, x2 = sorted((x + j[0], x + w + j[2]))  # detectors emit valid xyxy
                y1, y2 = sorted((y + j[1], y + h + j[3]))
                boxes.append([x1, y1, x2, y2])
                labels.append(cat if rng.rand() < 0.9 else int(rng.choice([CAR, PERSON])))
                scores.append(rng.rand())
        for _ in range(rng.randint(0, 4)):  # pure false positives
            x, y = rng.uniform(0, 150, 2)
            boxes.append([x, y, x + rng.uniform(10, 50), y + rng.uniform(10, 50)])
            labels.append(int(rng.choice([CAR, PERSON])))
            scores.append(rng.rand())
        dets.append(det(img_id, boxes, labels, scores))
    coco = {"images": images, "annotations": anns, "categories": [{"id": CAR}, {"id": PERSON}]}
    return coco, dets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_agree_with_pycocotools(seed):
    coco, dets = _random_scene(np.random.RandomState(seed))

    # Reference: COCOeval per-image matching at IoU 0.5, area range "all".
    results, ref = [], {}
    for d in dets:
        for b, lab, s in zip(d.boxes, d.labels, d.scores, strict=True):
            results.append(
                {
                    "image_id": d.image_id,
                    "category_id": int(lab),
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                    "score": float(s),
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = coco
        g.createIndex()
        ev = COCOeval(g, g.loadRes(results), iouType="bbox")
        ev.params.maxDets = [1000]
        ev.evaluate()
    for e in ev.evalImgs:
        if e is None or e["aRng"] != ev.params.areaRng[0]:
            continue
        for k, dt_id in enumerate(e["dtIds"]):
            ref[dt_id] = (
                "ignored" if e["dtIgnore"][0, k] else "tp" if e["dtMatches"][0, k] > 0 else "fp"
            )

    # Ours, mapped to the same detection ids (loadRes assigns ids 1..N in input order).
    gts = {
        img["id"]: GroundTruth(
            np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, bool)
        )
        for img in coco["images"]
    }
    from detrel.data import ground_truth

    gts |= ground_truth(coco, [img["id"] for img in coco["images"]])
    ours, dt_id = {}, 1
    for d in dets:
        m = match_image(d.boxes, d.labels, d.scores, gts[d.image_id], 0.5)
        for k in range(len(d.scores)):
            ours[dt_id] = "ignored" if m.ignored[k] else "tp" if m.tp[k] else "fp"
            dt_id += 1

    assert len(ref) == len(ours) > 100
    disagreements = {k: (ours[k], ref[k]) for k in ref if ours[k] != ref[k]}
    assert not disagreements
    counts = {s: list(ours.values()).count(s) for s in ("tp", "fp", "ignored")}
    assert all(v > 5 for v in counts.values())  # the scene exercises every outcome
