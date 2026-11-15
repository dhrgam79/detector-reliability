"""Detector loading, (corrupted) inference over a split, and on-disk detection caching.

Only post-NMS detections are stored: ``box (xyxy)``, ``label`` (COCO category id) and ``score``.
torchvision's Faster R-CNN applies softmax and per-class NMS inside ``roi_heads``, so the
pre-softmax logit of each kept detection is not exposed without patching torchvision internals.
Calibration therefore works on scores (brief §6.4 default path).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from detrel.corruptions import corrupt, derive_seed
from detrel.data import CocoSubset

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
CLEAN = "clean"


@dataclass(frozen=True)
class ImageDetections:
    """Post-NMS detections for one image.

    Attributes:
        image_id: COCO image id.
        boxes: ``float32`` ``(N, 4)`` ``xyxy`` in original-image pixels.
        labels: ``int64`` ``(N,)`` COCO category ids.
        scores: ``float32`` ``(N,)`` in ``[0, 1]``, descending.
    """

    image_id: int
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


# ----------------------------------------------------------------------------- model


def resolve_device(name: str) -> torch.device:
    """Map ``auto | cpu | cuda`` to a torch device."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_detector(
    cfg: dict[str, Any], pretrained: bool = True, **overrides: Any
) -> torch.nn.Module:
    """Build the configured torchvision detector in eval mode.

    Args:
        cfg: Validated config.
        pretrained: Load COCO weights (downloads on first use). ``False`` builds the same
            architecture with random weights, for offline tests.
        **overrides: Extra constructor kwargs (tests use small ``min_size``/``max_size``).

    Returns:
        The detector, in eval mode, on CPU.
    """
    m = cfg["model"]
    weights = torchvision.models.get_model_weights(m["name"])[m["weights"]] if pretrained else None
    model = torchvision.models.get_model(
        m["name"],
        weights=weights,
        weights_backbone=None,
        box_score_thresh=m["score_threshold"],
        box_detections_per_img=m["detections_per_image"],
        **overrides,
    )
    return model.eval()


def category_names(cfg: dict[str, Any]) -> list[str]:
    """Category names indexed by label id, from the configured weights' metadata (offline)."""
    m = cfg["model"]
    return list(torchvision.models.get_model_weights(m["name"])[m["weights"]].meta["categories"])


# --------------------------------------------------------------------------- dataset


def condition_key(corruption: str | None, severity: int) -> str:
    """Name of a test condition, e.g. ``clean`` or ``fog_3``."""
    if severity == 0 or corruption is None:
        return CLEAN
    return f"{corruption}_{severity}"


class CorruptedImages(Dataset):
    """Images of a :class:`CocoSubset`, corrupted on the fly and converted to CHW float tensors.

    Each image is corrupted with a seed derived from (seed, image_id, corruption, severity),
    so results do not depend on worker count, batch size or iteration order.
    """

    def __init__(self, subset: CocoSubset, corruption: str | None, severity: int, seed: int):
        """Wrap ``subset``; ``severity=0`` or ``corruption=None`` serves clean images."""
        self.subset = subset
        self.corruption = corruption
        self.severity = 0 if corruption is None else severity
        self.seed = seed

    def __len__(self) -> int:
        """Number of images."""
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        """Return ``(image_id, image)`` with image ``float32`` ``(3, H, W)`` in ``[0, 1]``."""
        sample = self.subset[idx]
        img = sample.image
        if self.severity > 0:
            assert self.corruption is not None
            s = derive_seed(self.seed, sample.image_id, self.corruption, self.severity)
            img = corrupt(img, self.corruption, self.severity, s)
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        return sample.image_id, tensor


def _collate(batch: list[tuple[int, torch.Tensor]]) -> tuple[list[int], list[torch.Tensor]]:
    ids, images = zip(*batch, strict=True)
    return list(ids), list(images)


# ------------------------------------------------------------------------- inference


def run_inference(
    model: Callable[[list[torch.Tensor]], list[dict[str, torch.Tensor]]],
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool = False,
) -> list[ImageDetections]:
    """Run ``model`` over ``dataset`` and return detections in dataset order.

    Args:
        model: torchvision-style detector: list of CHW tensors -> list of dicts with
            ``boxes``, ``labels``, ``scores``.
        dataset: Yields ``(image_id, image_tensor)``.
        batch_size: Images per forward pass.
        num_workers: DataLoader worker processes (0 = main process).
        device: Device to run on.
        amp: Use fp16 autocast (CUDA only).

    Returns:
        One :class:`ImageDetections` per image.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    use_amp = amp and device.type == "cuda"
    results: list[ImageDetections] = []
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device.type, enabled=use_amp):
        for ids, images in loader:
            outputs = model([im.to(device, non_blocking=True) for im in images])
            for image_id, out in zip(ids, outputs, strict=True):
                results.append(
                    ImageDetections(
                        image_id=int(image_id),
                        boxes=out["boxes"].float().cpu().numpy().reshape(-1, 4),
                        labels=out["labels"].cpu().numpy().astype(np.int64),
                        scores=out["scores"].float().cpu().numpy(),
                    )
                )
    dt = time.perf_counter() - t0
    logger.info("Inference: %d images in %.1fs (%.2f img/s)", len(results), dt, len(results) / dt)
    return results


# ----------------------------------------------------------------------------- cache


def cache_metadata(
    cfg: dict[str, Any], split: str, image_ids: Sequence[int], corruption: str | None, severity: int
) -> dict[str, Any]:
    """Everything that determines a condition's detections; a cache hit requires equality."""
    m = cfg["model"]
    ids_digest = hashlib.sha256(",".join(map(str, image_ids)).encode()).hexdigest()
    return {
        "cache_version": CACHE_VERSION,
        "model": m["name"],
        "weights": m["weights"],
        "score_threshold": m["score_threshold"],
        "detections_per_image": m["detections_per_image"],
        "amp": m["amp"],
        "seed": cfg["seed"],
        "split": split,
        "image_ids_sha256": ids_digest,
        "condition": condition_key(corruption, severity),
    }


def save_detections(path: Path, meta: dict[str, Any], dets: Sequence[ImageDetections]) -> None:
    """Save detections as tensors (loadable with ``weights_only=True``), atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "detections": [
            {
                "image_id": d.image_id,
                "boxes": torch.from_numpy(d.boxes),
                "labels": torch.from_numpy(d.labels),
                "scores": torch.from_numpy(d.scores),
            }
            for d in dets
        ],
    }
    tmp = path.with_suffix(".pt.part")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_detections(path: Path) -> tuple[dict[str, Any], list[ImageDetections]]:
    """Load a cache file written by :func:`save_detections`."""
    payload = torch.load(path, weights_only=True)
    dets = [
        ImageDetections(
            image_id=int(d["image_id"]),
            boxes=d["boxes"].numpy(),
            labels=d["labels"].numpy(),
            scores=d["scores"].numpy(),
        )
        for d in payload["detections"]
    ]
    return payload["meta"], dets


def cached_detections(
    path: Path, meta: dict[str, Any], compute: Callable[[], list[ImageDetections]]
) -> list[ImageDetections]:
    """Return cached detections if ``meta`` matches, else compute, save and return them."""
    if path.exists():
        stored_meta, dets = load_detections(path)
        if stored_meta == meta:
            logger.info("Cache hit: %s", path.name)
            return dets
        changed = sorted(k for k in meta if stored_meta.get(k) != meta[k])
        logger.warning("Stale cache %s (changed: %s); recomputing", path.name, changed)
    dets = compute()
    save_detections(path, meta, dets)
    logger.info("Cached %d images -> %s", len(dets), path)
    return dets
