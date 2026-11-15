import copy
import math

import numpy as np
import pytest

from detrel import evaluate
from detrel.config import load_config
from detrel.data import ground_truth
from detrel.metrics import bootstrap_weights
from detrel.model import ImageDetections

CAR, PERSON = 3, 1
N = 40


def scene():
    images = [{"id": i, "width": 100, "height": 100} for i in range(1, N + 1)]
    anns, aid = [], 1
    for i in range(1, N + 1):
        for cat, box in ((CAR, [10, 10, 30, 30]), (PERSON, [60, 10, 20, 50])):
            anns.append(
                {
                    "id": aid,
                    "image_id": i,
                    "category_id": cat,
                    "bbox": box,
                    "area": 1.0,
                    "iscrowd": 0,
                }
            )
            aid += 1
    cats = [{"id": CAR}, {"id": PERSON}]
    return {"images": images, "annotations": anns, "categories": cats}


CAR_BOX, PERSON_BOX = [10, 10, 40, 40], [60, 10, 80, 60]


def detections(degraded: bool):
    """Clean: both objects found at 0.9. Degraded: half the cars missed, an FP at 0.8 instead."""
    out = []
    for i in range(1, N + 1):
        boxes, labels, scores = [PERSON_BOX], [PERSON], [0.9]
        if degraded and i % 2 == 0:
            boxes.append([0, 70, 20, 90])
            labels.append(CAR)
            scores.append(0.8)
        else:
            boxes.append(CAR_BOX)
            labels.append(CAR)
            scores.append(0.9)
        out.append(
            ImageDetections(
                i, np.array(boxes, np.float32), np.array(labels), np.array(scores, np.float32)
            )
        )
    return out


@pytest.fixture
def setup():
    cfg = copy.deepcopy(load_config())
    cfg["metrics"]["bootstrap"]["n_resamples"] = 200
    coco = scene()
    ids = list(range(1, N + 1))
    return cfg, coco, ground_truth(coco, ids), ids, bootstrap_weights(N, 200, seed=0)


def test_scopes_and_metric_names(setup):
    cfg, coco, *_ = setup
    sc = evaluate.scopes(cfg, coco)
    assert list(sc) == ["driving", "vru", "vehicle", "infrastructure", "all80"]
    assert sc["all80"] == [PERSON, CAR]
    assert evaluate.metric_names(cfg) == ["mAP", "mAP50", "dece_t0.05", "dece_t0.3"]


def test_clean_condition_values(setup):
    cfg, coco, gt, ids, w = setup
    res = evaluate.evaluate_condition(cfg, coco, gt, detections(False), ids, w, "clean", 0)
    assert res.points[("vehicle", "mAP50")] == pytest.approx(1.0)
    assert res.points[("vru", "mAP50")] == pytest.approx(1.0)
    # Every detection is a TP at confidence 0.9 -> D-ECE = |1 - 0.9| = 0.1
    assert res.points[("driving", "dece_t0.05")] == pytest.approx(0.1)
    assert res.counts[("vehicle", "n_gt")] == N and res.counts[("driving", "n_det_t0.3")] == 2 * N
    assert math.isnan(res.points[("infrastructure", "mAP")])  # no traffic lights / stop signs
    assert ("all80", "mAP") not in res.samples  # no bootstrap for all-80 mAP (D51)
    assert ("all80", "dece_t0.3") in res.samples


def test_summary_rows_paired_deltas(setup):
    cfg, coco, gt, ids, w = setup
    clean = evaluate.evaluate_condition(cfg, coco, gt, detections(False), ids, w, "clean", 0)
    bad = evaluate.evaluate_condition(cfg, coco, gt, detections(True), ids, w, "fog", 5)
    rows = {r["scope"]: r for r in evaluate.summary_rows(cfg, bad, clean)}
    v = rows["vehicle"]
    assert v["corruption"] == "fog" and v["severity"] == 5
    assert v["d_mAP50"] < 0 and v["d_mAP50_hi"] < 0  # paired CI excludes zero
    assert v["mAP50_lo"] <= v["mAP50"] <= v["mAP50_hi"]
    assert rows["driving"]["d_dece_t0.05"] > 0  # overconfident FPs worsen calibration
    assert rows["vru"]["d_mAP50"] == pytest.approx(0.0)  # persons unchanged
    assert v["sufficient"] is False  # 40 GT < min_instances_for_claim
    assert math.isnan(rows["all80"]["mAP_lo"]) and math.isnan(rows["all80"]["d_mAP_lo"])
    assert not math.isnan(rows["all80"]["d_mAP"])

    self_rows = {r["scope"]: r for r in evaluate.summary_rows(cfg, clean, clean)}
    assert self_rows["vehicle"]["d_mAP"] == 0.0 and self_rows["vehicle"]["d_mAP_lo"] == 0.0


def test_reliability_rows(setup):
    cfg, coco, gt, ids, w = setup
    res = evaluate.evaluate_condition(cfg, coco, gt, detections(False), ids, w, "clean", 0)
    rows = evaluate.reliability_rows(res)
    assert len(rows) == 5 * 2 * cfg["metrics"]["dece_bins"]  # scopes x thresholds x bins
    used = [
        r
        for r in rows
        if r["scope"] == "driving" and r["score_threshold"] == 0.05 and r["count"] > 0
    ]
    # float32(0.9) = 0.89999998 falls in [0.8, 0.9): exactly one non-empty bin.
    assert len(used) == 1 and used[0]["bin"] == 8
    assert used[0]["count"] == 2 * N and used[0]["precision"] == pytest.approx(1.0)


def test_write_csv(tmp_path):
    p = tmp_path / "x" / "a.csv"
    evaluate.write_csv(
        p, [{"a": 1, "b": 0.123456789, "c": "s"}, {"a": 2, "b": float("nan"), "c": "t"}]
    )
    assert p.read_text().splitlines() == ["a,b,c", "1,0.123457,s", "2,nan,t"]


# ------------------------------------------------------------------ temperature scaling


def test_temperature_one_reproduces_raw_dece(setup):
    cfg, coco, gt, ids, w = setup
    res = evaluate.evaluate_condition(
        cfg, coco, gt, detections(True), ids, w, "fog", 5, temperature=1.0
    )
    for scope in ("driving", "vehicle"):
        for t in ("0.05", "0.3"):
            assert res.points[(scope, f"dece_cal_t{t}")] == pytest.approx(
                res.points[(scope, f"dece_t{t}")]
            )
            np.testing.assert_allclose(
                res.samples[(scope, f"dece_cal_t{t}")], res.samples[(scope, f"dece_t{t}")]
            )
    assert ("driving", 0.3, True) in res.reliability and ("driving", 0.3, False) in res.reliability


def test_calibrated_columns_and_paired_gain(setup):
    cfg, coco, gt, ids, w = setup
    # All detections are TPs at 0.9: sharpening (T < 1) moves confidence towards 1 -> D-ECE falls.
    clean = evaluate.evaluate_condition(
        cfg, coco, gt, detections(False), ids, w, "clean", 0, temperature=0.5
    )
    rows = {r["scope"]: r for r in evaluate.summary_rows(cfg, clean, clean)}
    d = rows["driving"]
    assert d["dece_cal_t0.3"] < d["dece_t0.3"]
    assert d["cal_gain_t0.3"] == pytest.approx(d["dece_t0.3"] - d["dece_cal_t0.3"])
    # Every detection is identical here, so the CI collapses onto the point (up to rounding).
    assert d["cal_gain_t0.3_lo"] - 1e-12 <= d["cal_gain_t0.3"] <= d["cal_gain_t0.3_hi"] + 1e-12
    assert "dece_cal_t0.05_lo" in d and "d_dece_cal_t0.3" in d
    rel = evaluate.reliability_rows(clean)
    assert (
        len(rel) == 5 * 2 * 2 * cfg["metrics"]["dece_bins"]
    )  # scopes x thresholds x {raw, cal} x bins
    assert {r["calibrated"] for r in rel} == {False, True}


def test_uncalibrated_rows_have_no_calibration_columns(setup):
    cfg, coco, gt, ids, w = setup
    res = evaluate.evaluate_condition(cfg, coco, gt, detections(False), ids, w, "clean", 0)
    row = evaluate.summary_rows(cfg, res, res)[0]
    assert not any(k.startswith(("dece_cal", "cal_gain")) for k in row)


def test_fit_on_split_uses_configured_scope_and_threshold(setup):
    cfg, coco, gt, ids, _ = setup
    fit = evaluate.fit_on_split(cfg, coco, gt, detections(True))
    # driving scope: 40 persons (TP) + 20 cars (TP) + 20 car FPs, all >= 0.05
    assert fit.n == 80 and fit.n_tp == 60
    cfg["calibration"]["fit_scope"] = "vru"
    assert evaluate.fit_on_split(cfg, coco, gt, detections(True)).n == 40
