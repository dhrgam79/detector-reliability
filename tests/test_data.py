import copy
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from detrel import data
from detrel.config import load_config

CAR, PERSON, DOG = 3, 1, 18  # COCO ids: two driving classes, one non-driving


def make_coco(n_images: int = 40) -> dict:
    """Synthetic COCO dict.

    Image i has: a car if i % 2 == 0, a dog always, a crowd-only person if i % 5 == 0.
    So odd ids divisible by 5 are crowd-only for driving classes and must be ineligible.
    """
    images = [
        {"id": i, "file_name": f"{i:012d}.png", "width": 64, "height": 48} for i in range(n_images)
    ]
    anns, aid = [], 0
    for i in range(n_images):
        objs = [(DOG, 0)]
        if i % 2 == 0:
            objs.append((CAR, 0))
        if i % 5 == 0:
            objs.append((PERSON, 1))
        for cat, crowd in objs:
            anns.append(
                {
                    "id": aid,
                    "image_id": i,
                    "category_id": cat,
                    "bbox": [2, 3, 10, 20],
                    "iscrowd": crowd,
                }
            )
            aid += 1
    # Annotation referring to an image that does not exist must be ignored.
    anns.append(
        {"id": aid, "image_id": 999, "category_id": CAR, "bbox": [0, 0, 1, 1], "iscrowd": 0}
    )
    return {"images": images, "annotations": anns, "categories": []}


@pytest.fixture
def coco() -> dict:
    return make_coco()


@pytest.fixture
def cfg(tmp_path) -> dict:
    c = copy.deepcopy(load_config())
    c["seed"] = 7
    c["data"].update(subset_size=10, calibration_size=4, evaluation_size=6)
    c["data"]["driving_classes"] = {"person": PERSON, "car": CAR}
    c["paths"].update(
        data_dir=str(tmp_path / "coco"),
        split_manifest=str(tmp_path / "split.json"),
    )
    return c


# --------------------------------------------------------------------------- eligibility


def test_eligible_requires_non_crowd_driving_instance(coco):
    ids = data.eligible_image_ids(coco, [PERSON, CAR])
    assert ids == list(range(0, 40, 2))  # odd multiples of 5 have only crowd persons
    assert 999 not in ids


def test_eligible_empty_when_no_class_matches(coco):
    assert data.eligible_image_ids(coco, [90]) == []


def test_class_counts_excludes_crowd(coco):
    counts = data.class_counts(coco, [0, 1, 2, 5], [PERSON, CAR])
    assert counts == {PERSON: 0, CAR: 2}


# ------------------------------------------------------------------------------- split


def test_split_sizes_disjoint_sorted():
    pool = list(range(100, 200))
    s = data.select_and_split(pool, 30, 12, seed=0)
    assert len(s.calibration) == 12 and len(s.evaluation) == 18
    assert not set(s.calibration) & set(s.evaluation)
    assert set(s.calibration) | set(s.evaluation) <= set(pool)
    assert s.calibration == sorted(s.calibration) and s.evaluation == sorted(s.evaluation)


def test_split_deterministic_and_order_independent():
    pool = list(range(100))
    a = data.select_and_split(pool, 20, 10, seed=3)
    b = data.select_and_split(list(reversed(pool)) + pool[:5], 20, 10, seed=3)  # shuffled + dups
    assert a == b


def test_split_changes_with_seed():
    pool = list(range(100))
    assert data.select_and_split(pool, 20, 10, seed=1) != data.select_and_split(
        pool, 20, 10, seed=2
    )


def test_split_pinned_regression():
    # Pinned output: a change here means every published number would change.
    s = data.select_and_split(list(range(50)), 6, 3, seed=42)
    assert s == data.Split(calibration=[13, 30, 39], evaluation=[17, 45, 48])


@pytest.mark.parametrize(("n", "subset", "calib"), [(5, 10, 4), (20, 10, 0), (20, 10, 10)])
def test_split_invalid(n, subset, calib):
    with pytest.raises(ValueError):
        data.select_and_split(list(range(n)), subset, calib, seed=0)


# ---------------------------------------------------------------------------- manifest


def test_make_split_writes_then_verifies_manifest(cfg, coco):
    first = data.make_split(cfg, coco, annotations_sha256="abc")
    stored = json.loads((data.resolve_path(cfg["paths"]["split_manifest"])).read_text())
    assert stored["calibration"] == first.calibration
    assert stored["evaluation"] == first.evaluation
    assert stored["n_eligible"] == 20
    assert data.make_split(cfg, coco, annotations_sha256="abc") == first


def test_make_split_rejects_changed_seed(cfg, coco):
    data.make_split(cfg, coco, annotations_sha256="abc")
    cfg["seed"] = 8
    with pytest.raises(RuntimeError, match="seed"):
        data.make_split(cfg, coco, annotations_sha256="abc")


def test_make_split_rejects_changed_annotations(cfg, coco):
    data.make_split(cfg, coco, annotations_sha256="abc")
    with pytest.raises(RuntimeError, match="annotations_sha256"):
        data.make_split(cfg, coco, annotations_sha256="different")


def test_make_split_rejects_tampered_manifest(cfg, coco):
    data.make_split(cfg, coco, annotations_sha256="abc")
    path = data.resolve_path(cfg["paths"]["split_manifest"])
    m = json.loads(path.read_text())
    m["evaluation"][0] += 1
    path.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="evaluation"):
        data.make_split(cfg, coco, annotations_sha256="abc")


def test_sha256_file(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"abc")
    assert data.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ----------------------------------------------------------------------------- dataset


def test_coco_subset_sample(tmp_path, coco):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.fromarray(np.full((48, 64, 3), 7, np.uint8)).save(img_dir / "000000000000.png")
    Image.fromarray(np.full((48, 64), 9, np.uint8)).save(img_dir / "000000000005.png")  # grayscale

    ds = data.CocoSubset(coco, img_dir, [5, 0])
    assert len(ds) == 2

    s = ds[1]  # image 0: dog, car, crowd person
    assert s.image_id == 0
    assert s.image.shape == (48, 64, 3) and s.image.dtype == np.uint8
    np.testing.assert_array_equal(s.boxes[0], [2, 3, 12, 23])  # xywh -> xyxy
    assert s.boxes.dtype == np.float32 and s.boxes.shape == (3, 4)
    assert s.labels.tolist() == [DOG, CAR, PERSON]
    assert s.iscrowd.tolist() == [False, False, True]

    g = ds[0]  # grayscale file is converted to RGB
    assert g.image.shape == (48, 64, 3) and int(g.image[0, 0, 2]) == 9


def test_coco_subset_image_without_annotations(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "a.png"}], "annotations": []}
    Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(tmp_path / "a.png")
    s = data.CocoSubset(coco, tmp_path, [1])[0]
    assert s.boxes.shape == (0, 4) and s.labels.shape == (0,) and s.iscrowd.shape == (0,)


def test_coco_subset_unknown_id(coco, tmp_path):
    with pytest.raises(KeyError):
        data.CocoSubset(coco, tmp_path, [12345])


# ---------------------------------------------------------------------------- download


def test_ensure_images_downloads_only_missing(cfg, coco, monkeypatch):
    img_dir = data.resolve_path(cfg["paths"]["data_dir"]) / cfg["paths"]["images"]
    img_dir.mkdir(parents=True)
    (img_dir / "000000000002.png").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(data, "_download", lambda url, dest: calls.append((url, dest.name)))
    data.ensure_images(cfg, coco, [0, 2, 4])
    base = cfg["data"]["images_base_url"]
    assert calls == [
        (f"{base}/000000000000.png", "000000000000.png"),
        (f"{base}/000000000004.png", "000000000004.png"),
    ]


def test_ensure_annotations_extracts_from_zip(cfg, tmp_path):
    src_zip = tmp_path / "ann.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.writestr(cfg["data"]["annotations_member"], json.dumps({"images": []}))
        zf.writestr("annotations/instances_train2017.json", "{}")
    cfg["data"]["annotations_url"] = src_zip.as_uri()  # file:// — exercises the real download path

    path = data.ensure_annotations(cfg)
    assert json.loads(path.read_text()) == {"images": []}
    assert not list(path.parent.glob("*.zip")) and not list(path.parent.glob("*.part"))
    assert data.ensure_annotations(cfg) == path  # second call is a no-op
