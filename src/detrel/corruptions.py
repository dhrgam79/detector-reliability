"""Deterministic camera corruptions applied to RGB uint8 images.

The corruption definitions (ImageNet-C / COCO-C, severities 1-5) are vendored in
:mod:`detrel._vendor.imagecorruptions`. This module adds input validation, a clean
``severity=0`` path, and per-call seeding so each (image, corruption, severity) is reproducible
regardless of call order or process.

Randomness uses ``numpy.random.RandomState``: NumPy guarantees its stream is stable across
versions (NEP 19), whereas ``numpy.random.Generator`` streams may change between releases.
"""

from __future__ import annotations

import hashlib

import numpy as np

from detrel._vendor.imagecorruptions import CORRUPTION_FUNCS

CORRUPTION_NAMES: tuple[str, ...] = tuple(CORRUPTION_FUNCS)
MAX_SEVERITY = 5
MIN_SIDE = 32  # smallest image side the corruption kernels support


def derive_seed(base_seed: int, image_id: int, corruption: str, severity: int) -> int:
    """Derive a stable 32-bit seed for one (image, corruption, severity) triple.

    Uses SHA-256 rather than ``hash()``, which is salted per Python process.

    Args:
        base_seed: Global experiment seed from the config.
        image_id: COCO image id.
        corruption: Corruption name.
        severity: Severity level.

    Returns:
        An integer in ``[0, 2**32)`` suitable for ``numpy.random.RandomState``.
    """
    key = f"{base_seed}|{image_id}|{corruption}|{severity}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def corrupt(image: np.ndarray, corruption: str, severity: int, seed: int) -> np.ndarray:
    """Apply one corruption at one severity.

    Args:
        image: RGB image, ``uint8``, shape ``(H, W, 3)`` with ``H, W >= 32``. Not modified.
        corruption: One of :data:`CORRUPTION_NAMES`.
        severity: ``0`` (clean, returns a copy) or ``1``-``5``.
        seed: Seed for this call; use :func:`derive_seed` for experiment runs.

    Returns:
        Corrupted image with the same shape and dtype as ``image``.

    Raises:
        TypeError: If ``image`` is not a ``uint8`` numpy array.
        ValueError: If the shape, corruption name, or severity is invalid.
    """
    _validate(image, corruption, severity)
    if severity == 0:
        return image.copy()

    rng = np.random.RandomState(seed)
    out = CORRUPTION_FUNCS[corruption](image, severity, rng)
    # Truncating cast matches the reference implementation; values are already in [0, 255].
    return np.asarray(out).astype(np.uint8)


def _validate(image: np.ndarray, corruption: str, severity: int) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a numpy array of dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if min(image.shape[:2]) < MIN_SIDE:
        raise ValueError(f"image sides must be >= {MIN_SIDE} px, got {image.shape[:2]}")
    if corruption not in CORRUPTION_FUNCS:
        raise ValueError(f"unknown corruption {corruption!r}; expected one of {CORRUPTION_NAMES}")
    if isinstance(severity, bool) or not isinstance(severity, int | np.integer):
        raise ValueError(f"severity must be an int, got {type(severity).__name__}")
    if not 0 <= severity <= MAX_SEVERITY:
        raise ValueError(f"severity must be in [0, {MAX_SEVERITY}], got {severity}")
