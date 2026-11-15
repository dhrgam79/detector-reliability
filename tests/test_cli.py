import copy
import json
import zipfile

import numpy as np
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

    split = cli.prepare_data(cfg)

    local = tmp_path / "coco" / cfg["paths"]["images"]
    assert sorted(p.name for p in local.iterdir()) == sorted(
        f"{i:012d}.png" for i in split.calibration + split.evaluation
    )
    assert cli.prepare_data(cfg) == split  # rerun verifies the manifest instead of rewriting
