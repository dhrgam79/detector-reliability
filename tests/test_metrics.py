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


def test_annotation_id_zero_is_rejected():
    coco = gt_coco()
    coco["annotations"][0]["id"] = 0
    with pytest.raises(ValueError, match="positive"):
        coco_map(coco, PERFECT, [1, 2], [CAR])


# ============================================================================ D-ECE

from detrel.matching import PooledMatches  # noqa: E402
from detrel.metrics import (  # noqa: E402
    MapEvaluator,
    bootstrap_ci,
    bootstrap_weights,
    dece,
    image_index,
    select_detections,
)


def test_dece_perfectly_calibrated_exact():
    # In each bin, the TP fraction equals the confidence exactly.
    scores, tp = [], []
    for c in np.arange(0.05, 1.0, 0.1):
        n_tp = int(round(c * 1000))
        scores += [c] * 1000
        tp += [True] * n_tp + [False] * (1000 - n_tp)
    assert dece(np.array(scores), np.array(tp), 10).value == pytest.approx(0.0, abs=1e-12)


def test_dece_perfectly_calibrated_stochastic():
    rng = np.random.RandomState(0)
    s = rng.uniform(0, 1, 200_000)
    assert dece(s, rng.uniform(0, 1, s.size) < s, 10).value < 0.01


def test_dece_all_tp_at_half_confidence_is_half():
    assert dece(np.full(50, 0.5), np.ones(50, bool), 10).value == pytest.approx(0.5)


def test_dece_overconfident_false_positives():
    assert dece(np.full(10, 0.9), np.zeros(10, bool), 10).value == pytest.approx(0.9)


def test_dece_hand_computed_two_bins():
    # bin [0.2,0.3): scores .25,.25 with 1 TP -> |0.5-0.25| * 2/4
    # bin [0.8,0.9): scores .85,.85 with 2 TP -> |1.0-0.85| * 2/4
    r = dece(np.array([0.25, 0.25, 0.85, 0.85]), np.array([1, 0, 1, 1], bool), 10)
    assert r.value == pytest.approx(0.5 * 0.25 + 0.5 * 0.15)
    assert r.count.tolist() == [0, 0, 2, 0, 0, 0, 0, 0, 2, 0]
    assert r.precision[2] == pytest.approx(0.5) and r.confidence[8] == pytest.approx(0.85)
    assert np.isnan(r.precision[0]) and np.isnan(r.confidence[0])


def test_dece_bin_edges_and_extremes():
    r = dece(np.array([0.0, 0.1, 0.0999, 1.0]), np.array([0, 0, 0, 1], bool), 10)
    assert r.count.tolist() == [2, 1, 0, 0, 0, 0, 0, 0, 0, 1]  # 1.0 lands in the last bin
    np.testing.assert_allclose(r.bin_edges, np.linspace(0, 1, 11))


def test_dece_integer_weights_equal_repetition():
    rng = np.random.RandomState(1)
    s, y, w = rng.rand(50), rng.rand(50) < 0.5, rng.randint(0, 4, 50)
    expected = dece(np.repeat(s, w), np.repeat(y, w), 10).value
    assert dece(s, y, 10, weights=w).value == pytest.approx(expected)


def test_dece_empty_is_nan():
    assert math.isnan(dece(np.zeros(0), np.zeros(0, bool), 10).value)


def pooled(image_id, label, score, tp):
    return PooledMatches(
        np.array(image_id, np.int64),
        np.array(label, np.int64),
        np.array(score, float),
        np.array(tp, bool),
    )


def test_select_detections_threshold_and_classes():
    p = pooled([1, 1, 2, 2], [CAR, PERSON, CAR, CAR], [0.2, 0.5, 0.3, 0.9], [0, 1, 1, 1])
    s = select_detections(p, 0.3, [CAR])
    assert s.score.tolist() == [0.3, 0.9] and s.image_id.tolist() == [2, 2]  # threshold inclusive


def test_image_index():
    p = pooled([30, 10, 30], [CAR] * 3, [0.5] * 3, [1, 1, 1])
    assert image_index(p, [10, 20, 30]).tolist() == [2, 0, 2]
    with pytest.raises(KeyError):
        image_index(p, [10, 20])


# ======================================================================= MapEvaluator

from tests.test_matching import _random_scene  # noqa: E402

RAND_CATS = [CAR, PERSON]


@pytest.mark.parametrize("seed", [0, 1])
def test_map_evaluator_matches_pycocotools(seed):
    coco, dets = _random_scene(np.random.RandomState(seed))
    ids = [img["id"] for img in coco["images"]]
    ev = MapEvaluator(coco, dets, ids, RAND_CATS)
    for cats in (RAND_CATS, [CAR], [PERSON]):
        ref = coco_map(coco, dets, ids, cats)
        ours = ev.compute(category_ids=cats)
        assert ours["mAP"] == pytest.approx(ref["mAP"], abs=1e-12)
        assert ours["mAP50"] == pytest.approx(ref["mAP50"], abs=1e-12)


def _duplicate(coco, dets, weights):
    """Physically repeat image i weights[i] times (new image and annotation ids)."""
    images, anns, out_dets, next_img, next_ann = [], [], [], 1, 1
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    det_by_img = {d.image_id: d for d in dets}
    for img, w in zip(coco["images"], weights, strict=True):
        for _ in range(int(w)):
            images.append({**img, "id": next_img})
            for a in by_img.get(img["id"], []):
                anns.append({**a, "id": next_ann, "image_id": next_img})
                next_ann += 1
            d = det_by_img[img["id"]]
            out_dets.append(ImageDetections(next_img, d.boxes, d.labels, d.scores))
            next_img += 1
    return {"images": images, "annotations": anns, "categories": coco["categories"]}, out_dets


def test_map_evaluator_weights_equal_duplicated_dataset():
    coco, dets = _random_scene(np.random.RandomState(3))
    ids = [img["id"] for img in coco["images"]]
    w = bootstrap_weights(len(ids), 1, seed=5)[0]
    dup_coco, dup_dets = _duplicate(coco, dets, w)
    ref = coco_map(dup_coco, dup_dets, [i["id"] for i in dup_coco["images"]], RAND_CATS)
    ours = MapEvaluator(coco, dets, ids, RAND_CATS).compute(weights=w)
    assert ours["mAP"] == pytest.approx(ref["mAP"], abs=1e-12)
    assert ours["mAP50"] == pytest.approx(ref["mAP50"], abs=1e-12)


def test_map_evaluator_no_detections_and_no_gt():
    coco = gt_coco()
    ev = MapEvaluator(coco, [], [1, 2], [CAR, PERSON])
    assert ev.compute(category_ids=[CAR]) == {"mAP": 0.0, "mAP50": 0.0}
    assert math.isnan(ev.compute(category_ids=[PERSON])["mAP"])  # crowd-only GT


# ========================================================================== bootstrap


def test_bootstrap_weights_shape_sums_and_determinism():
    w = bootstrap_weights(7, 50, seed=1)
    assert w.shape == (50, 7) and (w.sum(axis=1) == 7).all()
    np.testing.assert_array_equal(w, bootstrap_weights(7, 50, seed=1))
    assert not np.array_equal(w, bootstrap_weights(7, 50, seed=2))


def test_bootstrap_ci_of_mean_matches_normal_theory():
    rng = np.random.RandomState(0)
    x = rng.normal(10.0, 2.0, 400)

    def mean(w):
        return float(x.mean() if w is None else (w @ x) / w.sum())

    est = bootstrap_ci(mean, bootstrap_weights(len(x), 2000, seed=0), 0.95)
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert est.value == pytest.approx(x.mean())
    assert est.low < est.value < est.high
    assert (est.high - est.low) / 2 == pytest.approx(half, rel=0.1)


def test_bootstrap_ci_drops_undefined_resamples():
    w = np.array([[1, 0], [0, 1], [2, 0], [0, 2], [1, 1]])
    values = {
        (1, 0): float("nan"),
        (0, 1): float("nan"),
        (2, 0): float("nan"),
        (0, 2): 2.0,
        (1, 1): 3.0,
    }

    def stat(weights):
        return 1.0 if weights is None else values[tuple(weights)]

    est = bootstrap_ci(stat, w, 0.5)
    assert est.value == 1.0 and 2.0 <= est.low <= est.high <= 3.0


def test_dece_resamples_equal_per_resample_dece():
    from detrel.metrics import dece_resamples

    rng = np.random.RandomState(4)
    n_img, d = 30, 400
    s, y = rng.rand(d), rng.rand(d) < 0.4
    idx = rng.randint(0, n_img, d)
    w = bootstrap_weights(n_img, 25, seed=9)
    w[0] = 0
    w[0, 0] = n_img  # a resample of one image, possibly without detections
    fast = dece_resamples(s, y, idx, 10, w)
    for r in range(len(w)):
        slow = dece(s, y, 10, weights=w[r][idx]).value
        assert (math.isnan(fast[r]) and math.isnan(slow)) or fast[r] == pytest.approx(
            slow, abs=1e-12
        )


def test_percentile_ci():
    from detrel.metrics import percentile_ci

    est = percentile_ci(0.5, np.r_[np.linspace(0, 1, 101), np.nan], 0.9)
    assert est.value == 0.5 and est.low == pytest.approx(0.05) and est.high == pytest.approx(0.95)
    assert math.isnan(percentile_ci(0.1, np.array([np.nan]), 0.95).low)
