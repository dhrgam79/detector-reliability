import copy

import numpy as np
import pytest
import torch
from PIL import Image

from detrel import model
from detrel.config import load_config
from detrel.data import CocoSubset


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


@pytest.fixture
def subset(tmp_path):
    coco = {
        "images": [{"id": i, "file_name": f"{i}.png"} for i in range(5)],
        "annotations": [],
    }
    rng = np.random.RandomState(0)
    for i in range(5):
        Image.fromarray(rng.randint(0, 256, (40, 48, 3)).astype(np.uint8)).save(
            tmp_path / f"{i}.png"
        )
    return CocoSubset(coco, tmp_path, [4, 0, 2, 1, 3])


class FakeDetector:
    """Returns one box per image whose score encodes the image's mean intensity."""

    def __init__(self):
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
                "labels": torch.tensor([3]),
                "scores": im.mean().reshape(1),
            }
            for im in images
        ]


# ------------------------------------------------------------------------------- basics


def test_condition_key():
    assert model.condition_key(None, 0) == "clean"
    assert model.condition_key("fog", 0) == "clean"
    assert model.condition_key("fog", 3) == "fog_3"


def test_resolve_device():
    assert model.resolve_device("cpu").type == "cpu"
    assert model.resolve_device("auto").type in {"cpu", "cuda"}


def test_weights_categories_match_driving_classes(cfg):
    # Label ids output by the detector must be COCO category ids (offline metadata check).
    names = model.category_names(cfg)
    for name, cid in cfg["data"]["driving_classes"].items():
        assert names[cid] == name


def test_build_detector_architecture_offline(cfg):
    det = model.build_detector(cfg, pretrained=False, min_size=64, max_size=64)
    assert not det.training
    assert det.roi_heads.score_thresh == cfg["model"]["score_threshold"]
    assert det.roi_heads.detections_per_img == cfg["model"]["detections_per_image"]
    with torch.inference_mode():
        (out,) = det([torch.rand(3, 50, 60)])
    assert set(out) == {"boxes", "labels", "scores"}
    assert out["boxes"].shape[1] == 4
    assert len(out["scores"]) <= cfg["model"]["detections_per_image"]


# ------------------------------------------------------------------------------ dataset


def test_corrupted_images_clean_matches_source(subset):
    ds = model.CorruptedImages(subset, None, 0, seed=42)
    image_id, t = ds[0]
    assert image_id == 4
    assert t.dtype == torch.float32 and t.shape == (3, 40, 48)
    expected = torch.from_numpy(subset[0].image).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(t, expected)


def test_corrupted_images_deterministic_and_differs_from_clean(subset):
    a = model.CorruptedImages(subset, "gaussian_noise", 3, seed=42)[1][1]
    b = model.CorruptedImages(subset, "gaussian_noise", 3, seed=42)[1][1]
    clean = model.CorruptedImages(subset, None, 0, seed=42)[1][1]
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, clean)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


# ---------------------------------------------------------------------------- inference


@pytest.mark.parametrize(("batch_size", "num_workers"), [(1, 0), (2, 0), (4, 2)])
def test_run_inference_order_and_types(subset, batch_size, num_workers):
    ds = model.CorruptedImages(subset, "fog", 2, seed=42)
    dets = model.run_inference(FakeDetector(), ds, batch_size, num_workers, torch.device("cpu"))
    assert [d.image_id for d in dets] == [4, 0, 2, 1, 3]
    for i, d in enumerate(dets):
        assert d.boxes.dtype == np.float32 and d.boxes.shape == (1, 4)
        assert d.labels.dtype == np.int64 and d.scores.dtype == np.float32
        assert d.scores[0] == pytest.approx(float(ds[i][1].mean()), abs=1e-6)


def test_run_inference_results_independent_of_batching(subset):
    ds = model.CorruptedImages(subset, "snow", 4, seed=1)
    a = model.run_inference(FakeDetector(), ds, 1, 0, torch.device("cpu"))
    b = model.run_inference(FakeDetector(), ds, 3, 2, torch.device("cpu"))
    assert [x.scores.tolist() for x in a] == [x.scores.tolist() for x in b]


def test_run_inference_handles_empty_outputs(subset):
    def empty(images):
        return [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.int64),
                "scores": torch.zeros(0),
            }
            for _ in images
        ]

    dets = model.run_inference(
        empty, model.CorruptedImages(subset, None, 0, 0), 2, 0, torch.device("cpu")
    )
    assert all(d.boxes.shape == (0, 4) and d.scores.shape == (0,) for d in dets)


# -------------------------------------------------------------------------------- cache


def _dets():
    return [
        model.ImageDetections(
            image_id=7,
            boxes=np.array([[0, 0, 5, 5]], np.float32),
            labels=np.array([1], np.int64),
            scores=np.array([0.9], np.float32),
        )
    ]


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "c.pt"
    model.save_detections(path, {"a": 1}, _dets())
    meta, dets = model.load_detections(path)  # weights_only=True inside
    assert meta == {"a": 1}
    np.testing.assert_array_equal(dets[0].boxes, _dets()[0].boxes)
    assert dets[0].image_id == 7 and not list(tmp_path.glob("*.part"))


def test_cached_detections_hit_and_invalidation(tmp_path, cfg):
    path = tmp_path / "evaluation" / "fog_3.pt"
    meta = model.cache_metadata(cfg, "evaluation", [1, 2, 3], "fog", 3)
    calls = []

    def compute():
        calls.append(1)
        return _dets()

    model.cached_detections(path, meta, compute)
    model.cached_detections(path, meta, compute)
    assert len(calls) == 1  # second call served from cache

    cfg["model"]["score_threshold"] = 0.1
    changed = model.cache_metadata(cfg, "evaluation", [1, 2, 3], "fog", 3)
    model.cached_detections(path, changed, compute)
    assert len(calls) == 2  # stale cache recomputed
    assert model.load_detections(path)[0] == changed


def test_cache_metadata_depends_on_image_ids_and_condition(cfg):
    base = model.cache_metadata(cfg, "evaluation", [1, 2, 3], "fog", 3)
    assert base != model.cache_metadata(cfg, "evaluation", [1, 2, 4], "fog", 3)
    assert base != model.cache_metadata(cfg, "evaluation", [1, 2, 3], "fog", 4)
    assert base != model.cache_metadata(cfg, "calibration", [1, 2, 3], "fog", 3)
