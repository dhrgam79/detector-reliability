import copy
import json
import math
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from detrel import cli
from detrel.config import load_config
from tests.test_data import make_coco


def test_prepare_data_end_to_end(tmp_path):
    """Full prepare-data flow served from file:// URLs (no network)."""
    coco = make_coco()
    cfg = copy.deepcopy(load_config())
    cfg["data"].update(subset_size=10, calibration_size=4, evaluation_size=6)
    cfg["data"]["driving_classes"] = {"person": 1, "car": 3}

    remote = tmp_path / "remote"
    (remote / "val2017").mkdir(parents=True)
    for img in coco["images"]:
        Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(remote / "val2017" / img["file_name"])
    with zipfile.ZipFile(remote / "ann.zip", "w") as zf:
        zf.writestr(cfg["data"]["annotations_member"], json.dumps(coco))

    cfg["data"]["annotations_url"] = (remote / "ann.zip").as_uri()
    cfg["data"]["images_base_url"] = (remote / "val2017").as_uri()
    cfg["paths"].update(
        data_dir=str(tmp_path / "coco"), split_manifest=str(tmp_path / "split.json")
    )

    _, split = cli.prepare_data(cfg)

    local = tmp_path / "coco" / cfg["paths"]["images"]
    assert sorted(p.name for p in local.iterdir()) == sorted(
        f"{i:012d}.png" for i in split.calibration + split.evaluation
    )
    assert cli.prepare_data(cfg)[1] == split  # rerun verifies the manifest instead of rewriting


def test_baseline_end_to_end_with_fake_detector(tmp_path, monkeypatch):
    """prepare -> clean inference -> cache -> mAP, with a detector that outputs the GT car box."""
    coco = make_coco()
    cfg = copy.deepcopy(load_config())
    cfg["data"].update(subset_size=10, calibration_size=4, evaluation_size=6)
    cfg["data"]["driving_classes"] = {"person": 1, "car": 3}
    cfg["data"]["class_groups"] = {"vru": [1], "vehicle": [3]}
    cfg["model"].update(device="cpu", num_workers=0)
    cfg["paths"].update(
        data_dir=str(tmp_path / "coco"),
        split_manifest=str(tmp_path / "split.json"),
        cache_dir=str(tmp_path / "cache"),
    )
    img_dir = tmp_path / "coco" / cfg["paths"]["images"]
    img_dir.mkdir(parents=True)
    for img in coco["images"]:
        Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(img_dir / img["file_name"])
    ann = tmp_path / "coco" / cfg["paths"]["annotations"]
    ann.parent.mkdir(parents=True, exist_ok=True)
    ann.write_text(json.dumps(coco))

    class GTCar(torch.nn.Module):
        def forward(self, images):
            return [
                {
                    "boxes": torch.tensor([[2.0, 3.0, 12.0, 23.0]]),
                    "labels": torch.tensor([3]),
                    "scores": torch.tensor([0.9]),
                }
                for _ in images
            ]

    built = []
    monkeypatch.setattr(cli.model, "build_detector", lambda c: built.append(1) or GTCar())

    tables = cli.baseline(cfg, None)
    # Every eligible image has exactly one car at that box; persons are crowd-only (ignored).
    assert tables["evaluation"]["driving (8 classes)"]["mAP"] == pytest.approx(1.0)
    assert tables["evaluation"]["group: vehicle"]["mAP50"] == pytest.approx(1.0)
    assert math.isnan(tables["evaluation"]["group: vru"]["mAP"])
    assert len(list((tmp_path / "cache").rglob("clean.pt"))) == 2

    cli.baseline(cfg, None)  # second run: both splits served from cache, model never rebuilt
    assert len(built) == 1


def test_sweep_end_to_end_with_fake_detector(tmp_path, monkeypatch):
    """prepare -> calibration + 3 conditions inference -> metrics -> CSVs + run_info."""
    coco = make_coco()
    cfg = copy.deepcopy(load_config())
    cfg["data"].update(subset_size=10, calibration_size=4, evaluation_size=6)
    cfg["data"]["driving_classes"] = {"person": 1, "car": 3}
    cfg["data"]["class_groups"] = {"vru": [1], "vehicle": [3]}
    cfg["corruptions"].update(names=["gaussian_noise"], severities=[1, 2])
    cfg["metrics"]["bootstrap"]["n_resamples"] = 30
    cfg["model"].update(device="cpu", num_workers=0)
    cfg["paths"].update(
        data_dir=str(tmp_path / "coco"),
        split_manifest=str(tmp_path / "split.json"),
        cache_dir=str(tmp_path / "cache"),
        summary_csv=str(tmp_path / "out" / "summary.csv"),
        reliability_csv=str(tmp_path / "out" / "reliability.csv"),
        run_info=str(tmp_path / "out" / "run_info.json"),
    )
    img_dir = tmp_path / "coco" / cfg["paths"]["images"]
    img_dir.mkdir(parents=True)
    for img in coco["images"]:
        Image.fromarray(np.full((48, 64, 3), 128, np.uint8)).save(img_dir / img["file_name"])
    ann = tmp_path / "coco" / cfg["paths"]["annotations"]
    ann.parent.mkdir(parents=True, exist_ok=True)
    ann.write_text(json.dumps(coco))

    class NoisyCar(torch.nn.Module):
        """Finds the car; confidence falls as the image gets noisier (pixel std grows)."""

        def forward(self, images):
            return [
                {
                    "boxes": torch.tensor([[2.0, 3.0, 12.0, 23.0]]),
                    "labels": torch.tensor([3]),
                    "scores": (0.95 - im.std()).clamp(0.05, 1.0).reshape(1),
                }
                for im in images
            ]

    monkeypatch.setattr(cli.model, "build_detector", lambda c: NoisyCar())
    rows = cli.sweep(cfg, None)

    out = tmp_path / "out"
    lines = (out / "summary.csv").read_text().splitlines()
    assert (
        len(lines) == 1 + 3 * 4
    )  # header + (clean, noise_1, noise_2) x (driving, vru, vehicle, all80)
    assert lines[0].startswith("corruption,severity,scope,n_gt,sufficient,mAP,mAP_lo,mAP_hi,d_mAP")
    assert {(r["corruption"], r["severity"]) for r in rows} == {
        ("clean", 0),
        ("gaussian_noise", 1),
        ("gaussian_noise", 2),
    }
    vehicle = {r["severity"]: r for r in rows if r["scope"] == "vehicle"}
    assert vehicle[0]["mAP50"] == pytest.approx(1.0)
    assert vehicle[0]["d_mAP"] == 0.0
    # All detections are TPs, so D-ECE = 1 - mean confidence: grows with noise.
    assert vehicle[0]["dece_t0.05"] < vehicle[1]["dece_t0.05"] < vehicle[2]["dece_t0.05"]

    info = json.loads((out / "run_info.json").read_text())
    assert info["conditions_total"] == 4 and len(info["conditions_inferred"]) == 4  # calib + 3
    assert (out / "reliability.csv").exists()

    cli.sweep(cfg, None)  # rerun: everything from cache
    assert json.loads((out / "run_info.json").read_text())["conditions_inferred"] == []
