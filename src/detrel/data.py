"""COCO val2017 subset selection, seeded calibration/evaluation split, and image loading.

The subset is drawn only from images containing at least one non-crowd instance of a
driving-relevant class. The resulting image ids are written to a JSON manifest that is
committed to the repo; later runs recompute the split and fail loudly if it differs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from detrel.config import resolve_path

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class Split:
    """Disjoint calibration and evaluation image-id lists (each sorted ascending)."""

    calibration: list[int]
    evaluation: list[int]


@dataclass(frozen=True)
class Sample:
    """One image with its ground truth.

    Attributes:
        image_id: COCO image id.
        image: RGB ``uint8`` array, shape ``(H, W, 3)``.
        boxes: Ground-truth boxes, ``float32`` ``(N, 4)``, ``xyxy`` in pixels.
        labels: COCO category ids, ``int64`` ``(N,)``.
        iscrowd: Crowd flags, ``bool`` ``(N,)``.
    """

    image_id: int
    image: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    iscrowd: np.ndarray


# ------------------------------------------------------------------------ annotations


def load_annotations(path: str | Path) -> dict[str, Any]:
    """Load a COCO instances JSON file."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eligible_image_ids(coco: dict[str, Any], class_ids: Iterable[int]) -> list[int]:
    """Ids of images with at least one non-crowd instance of ``class_ids``.

    Args:
        coco: Parsed COCO instances dictionary.
        class_ids: COCO category ids that make an image eligible.

    Returns:
        Sorted list of image ids.
    """
    wanted = set(class_ids)
    known = {img["id"] for img in coco["images"]}
    ids = {
        a["image_id"]
        for a in coco["annotations"]
        if a["category_id"] in wanted and not a.get("iscrowd", 0) and a["image_id"] in known
    }
    return sorted(ids)


def select_and_split(
    eligible: Sequence[int], subset_size: int, calibration_size: int, seed: int
) -> Split:
    """Draw a seeded subset and split it into calibration and evaluation sets.

    The candidate list is sorted first, so the result does not depend on annotation-file
    order. ``numpy.random.RandomState`` is used because its stream is stable across NumPy
    versions.

    Args:
        eligible: Candidate image ids.
        subset_size: Total images to draw.
        calibration_size: How many of those go to calibration; the rest go to evaluation.
        seed: Global experiment seed.

    Returns:
        The split, each part sorted ascending.

    Raises:
        ValueError: If there are too few candidates or sizes are inconsistent.
    """
    if not 0 < calibration_size < subset_size:
        raise ValueError("need 0 < calibration_size < subset_size")
    pool = np.array(sorted(set(eligible)), dtype=np.int64)
    if len(pool) < subset_size:
        raise ValueError(f"only {len(pool)} eligible images, need {subset_size}")
    chosen = pool[np.random.RandomState(seed).permutation(len(pool))[:subset_size]]
    return Split(
        calibration=sorted(int(i) for i in chosen[:calibration_size]),
        evaluation=sorted(int(i) for i in chosen[calibration_size:]),
    )


def class_counts(
    coco: dict[str, Any], image_ids: Iterable[int], class_ids: Iterable[int]
) -> dict[int, int]:
    """Count non-crowd instances per class over the given images."""
    ids, wanted = set(image_ids), set(class_ids)
    counts = dict.fromkeys(sorted(wanted), 0)
    for a in coco["annotations"]:
        if a["image_id"] in ids and a["category_id"] in wanted and not a.get("iscrowd", 0):
            counts[a["category_id"]] += 1
    return counts


# --------------------------------------------------------------------------- manifest


def build_manifest(
    cfg: dict[str, Any], coco: dict[str, Any], split: Split, annotations_sha256: str
) -> dict[str, Any]:
    """Build the JSON-serialisable record of how the split was produced."""
    data = cfg["data"]
    class_ids = sorted(data["driving_classes"].values())
    return {
        "version": MANIFEST_VERSION,
        "seed": cfg["seed"],
        "subset_size": data["subset_size"],
        "calibration_size": data["calibration_size"],
        "evaluation_size": data["evaluation_size"],
        "driving_class_ids": class_ids,
        "annotations_sha256": annotations_sha256,
        "n_eligible": len(eligible_image_ids(coco, class_ids)),
        "class_counts": {
            "calibration": class_counts(coco, split.calibration, class_ids),
            "evaluation": class_counts(coco, split.evaluation, class_ids),
        },
        "calibration": split.calibration,
        "evaluation": split.evaluation,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write a manifest as indented JSON (image-id lists on one line each)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")


def make_split(cfg: dict[str, Any], coco: dict[str, Any], annotations_sha256: str) -> Split:
    """Compute the split and reconcile it with the committed manifest.

    If no manifest exists, one is written. If it exists, the freshly computed split must match
    it exactly, otherwise the run is aborted: a silent change of the evaluation set would make
    results incomparable across runs.

    Args:
        cfg: Validated config.
        coco: Parsed COCO instances dictionary.
        annotations_sha256: Digest of the annotation file ``coco`` was loaded from.

    Returns:
        The calibration/evaluation split.

    Raises:
        RuntimeError: If the computed split or its inputs disagree with the manifest.
    """
    data = cfg["data"]
    class_ids = data["driving_classes"].values()
    split = select_and_split(
        eligible_image_ids(coco, class_ids),
        data["subset_size"],
        data["calibration_size"],
        cfg["seed"],
    )
    manifest = build_manifest(cfg, coco, split, annotations_sha256)
    path = resolve_path(cfg["paths"]["split_manifest"])

    if not path.exists():
        write_manifest(manifest, path)
        logger.info("Wrote split manifest to %s", path)
    else:
        with path.open(encoding="utf-8") as f:
            stored = json.load(f)
        stored["class_counts"] = {
            k: {int(c): n for c, n in v.items()} for k, v in stored["class_counts"].items()
        }
        diffs = sorted(k for k in manifest if manifest[k] != stored.get(k))
        if diffs:
            raise RuntimeError(
                f"Split differs from committed manifest {path} in fields {diffs}. "
                "Delete the manifest only if the change is intentional."
            )
        logger.info("Split matches committed manifest %s", path)

    for part in ("calibration", "evaluation"):
        logger.info("%s class counts: %s", part, manifest["class_counts"][part])
    return split


# --------------------------------------------------------------------------- download


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically (via a ``.part`` file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def ensure_annotations(cfg: dict[str, Any]) -> Path:
    """Download and extract ``instances_val2017.json`` if it is not already present.

    Returns:
        Path to the annotation file.
    """
    paths, data = cfg["paths"], cfg["data"]
    ann_path = resolve_path(paths["data_dir"]) / paths["annotations"]
    if ann_path.exists():
        return ann_path
    zip_path = ann_path.parent / Path(data["annotations_url"]).name
    if not zip_path.exists():
        logger.info("Downloading %s (~250 MB)", data["annotations_url"])
        _download(data["annotations_url"], zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open(data["annotations_member"]) as src:
        tmp = ann_path.with_suffix(".json.part")
        with tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(ann_path)
    zip_path.unlink()
    logger.info("Extracted %s", ann_path)
    return ann_path


def ensure_images(cfg: dict[str, Any], coco: dict[str, Any], image_ids: Iterable[int]) -> None:
    """Download any of the given images that are missing from the image directory."""
    paths = cfg["paths"]
    image_dir = resolve_path(paths["data_dir"]) / paths["images"]
    names = {img["id"]: img["file_name"] for img in coco["images"]}
    missing = [i for i in image_ids if not (image_dir / names[i]).exists()]
    if missing:
        logger.info("Downloading %d images to %s", len(missing), image_dir)
    base = cfg["data"]["images_base_url"].rstrip("/")
    # Downloads are latency-bound (~1.2 s each sequentially on Colab), so fetch in parallel.
    with ThreadPoolExecutor(max_workers=cfg["data"]["download_workers"]) as pool:
        futures = [
            pool.submit(_download, f"{base}/{names[i]}", image_dir / names[i]) for i in missing
        ]
        for n, fut in enumerate(as_completed(futures), 1):
            fut.result()  # re-raise any download error
            if n % 100 == 0:
                logger.info("  %d/%d", n, len(missing))


# ------------------------------------------------------------------------------ dataset


class CocoSubset:
    """Sequence of :class:`Sample` for a fixed list of COCO image ids.

    Compatible with ``torch.utils.data.DataLoader`` (map-style) without importing torch.
    Ground truth includes all classes and crowd flags; filtering happens at metric time.
    """

    def __init__(self, coco: dict[str, Any], image_dir: str | Path, image_ids: Sequence[int]):
        """Index annotations for ``image_ids``.

        Args:
            coco: Parsed COCO instances dictionary.
            image_dir: Directory containing the image files.
            image_ids: Image ids in the order they should be served.
        """
        self.image_dir = Path(image_dir)
        self.image_ids = list(image_ids)
        wanted = set(self.image_ids)
        self._files = {img["id"]: img["file_name"] for img in coco["images"] if img["id"] in wanted}
        missing = wanted - self._files.keys()
        if missing:
            raise KeyError(f"image ids not in annotations: {sorted(missing)[:5]}")
        self._anns: dict[int, list[dict[str, Any]]] = {i: [] for i in self.image_ids}
        for a in coco["annotations"]:
            if a["image_id"] in wanted:
                self._anns[a["image_id"]].append(a)

    def __len__(self) -> int:
        """Number of images."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Sample:
        """Load image ``idx`` and its ground truth."""
        image_id = self.image_ids[idx]
        with Image.open(self.image_dir / self._files[image_id]) as im:
            image = np.array(im.convert("RGB"), dtype=np.uint8)  # writable copy
        anns = self._anns[image_id]
        xywh = np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4)
        boxes = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
        return Sample(
            image_id=image_id,
            image=image,
            boxes=boxes,
            labels=np.array([a["category_id"] for a in anns], dtype=np.int64),
            iscrowd=np.array([bool(a.get("iscrowd", 0)) for a in anns], dtype=bool),
        )
