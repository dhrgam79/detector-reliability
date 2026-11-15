import numpy as np
import pytest
from scipy.special import expit

from detrel.calibration import (
    apply_temperature,
    binary_nll,
    calibrated_detections,
    fit_temperature,
    logit,
)
from detrel.metrics import MapEvaluator, coco_map, dece
from tests.test_matching import _random_scene

EPS = 1e-12
BOUNDS = (0.05, 20.0)


def synthetic(true_t, n=50_000, seed=0):
    """Scores s = sigmoid(z); labels drawn from the *calibrated* probability sigmoid(z / T)."""
    rng = np.random.RandomState(seed)
    z = rng.normal(0.0, 3.0, n)
    s = expit(z)
    y = rng.uniform(size=n) < expit(z / true_t)
    return s, y


# ------------------------------------------------------------------------- transform


def test_logit_inverse_of_sigmoid_and_clipping():
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(expit(logit(p, EPS)), p)
    assert np.isfinite(logit(np.array([0.0, 1.0]), EPS)).all()


def test_temperature_one_is_identity():
    s = np.random.RandomState(1).uniform(0.05, 1.0, 1000).astype(np.float32)
    np.testing.assert_allclose(apply_temperature(s, 1.0, EPS), s, rtol=1e-6)


def test_temperature_direction():
    s = np.array([0.2, 0.9])
    soft, sharp = apply_temperature(s, 2.0, EPS), apply_temperature(s, 0.5, EPS)
    assert soft[1] < 0.9 and soft[0] > 0.2  # T > 1 pulls towards 0.5
    assert sharp[1] > 0.9 and sharp[0] < 0.2  # T < 1 pushes away
    assert apply_temperature(np.array([0.5]), 3.0, EPS)[0] == pytest.approx(0.5)


def test_ranking_preserved_including_scores_near_one():
    s = np.array([0.999999, 0.9999995, 0.99999994, 1.0, 0.05, 0.5], dtype=np.float32)
    out = apply_temperature(s, 3.0, EPS)
    assert (np.argsort(-s, kind="stable") == np.argsort(-out, kind="stable")).all()
    assert len(np.unique(out)) == len(np.unique(s))  # no new ties


def test_invalid_temperature():
    with pytest.raises(ValueError):
        apply_temperature(np.array([0.5]), 0.0, EPS)


# ------------------------------------------------------------------------------- fit


@pytest.mark.parametrize("true_t", [0.6, 1.0, 1.7, 3.0])
def test_fit_recovers_known_temperature(true_t):
    s, y = synthetic(true_t)
    r = fit_temperature(s, y, BOUNDS, EPS)
    assert r.temperature == pytest.approx(true_t, rel=0.04)
    assert r.nll_after <= r.nll_before + 1e-12
    assert not r.at_bound and r.n == len(s) and r.n_tp == int(y.sum())


def test_fit_improves_dece_on_overconfident_data():
    s, y = synthetic(2.0, seed=3)
    r = fit_temperature(s, y, BOUNDS, EPS)
    before = dece(s, y, 10).value
    after = dece(apply_temperature(s, r.temperature, EPS), y, 10).value
    assert before > 0.05 and after < before / 3


def test_fit_is_global_minimum():
    s, y = synthetic(1.7, n=5000, seed=4)
    r = fit_temperature(s, y, BOUNDS, EPS)
    grid = np.exp(np.linspace(np.log(0.2), np.log(10), 400))
    assert r.nll_after <= min(binary_nll(s, y, t, EPS) for t in grid) + 1e-9


def test_degenerate_labels_hit_bound_and_flag_it():
    s = np.random.RandomState(0).uniform(0.6, 0.99, 200)
    r = fit_temperature(s, np.ones(200, bool), BOUNDS, EPS)
    assert r.at_bound and BOUNDS[0] <= r.temperature <= BOUNDS[1]


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros(0), np.zeros(0, bool), BOUNDS, EPS)


# ------------------------------------------------- mAP unchanged (brief §6.4 requirement)


@pytest.mark.parametrize("temperature", [0.4, 1.8, 7.5])
def test_map_unchanged_by_temperature_scaling(temperature):
    coco, dets = _random_scene(np.random.RandomState(11))
    ids = [img["id"] for img in coco["images"]]
    cats = [1, 3]
    scaled = calibrated_detections(dets, temperature, EPS)

    assert coco_map(coco, scaled, ids, cats) == pytest.approx(
        coco_map(coco, dets, ids, cats), abs=1e-12
    )
    before = MapEvaluator(coco, dets, ids, cats).compute()
    after = MapEvaluator(coco, scaled, ids, cats).compute()
    assert after == pytest.approx(before, abs=1e-12)

    # Same detection set: nothing dropped or re-thresholded.
    assert [len(d.scores) for d in scaled] == [len(d.scores) for d in dets]
    for a, b in zip(scaled, dets, strict=True):
        np.testing.assert_array_equal(a.boxes, b.boxes)
        np.testing.assert_array_equal(a.labels, b.labels)
