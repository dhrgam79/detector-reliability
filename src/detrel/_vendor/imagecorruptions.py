"""Five image corruptions vendored from ``imagecorruptions`` 1.1.2.

Origin:
    https://github.com/bethgelab/imagecorruptions (Michaelis et al., 2019), itself adapted from
    https://github.com/hendrycks/robustness (Hendrycks & Dietterich, 2019).
    Licensed under the Apache License, Version 2.0; see ``LICENSE-imagecorruptions`` in this
    directory.

Changes from the original (as required by Apache-2.0, section 4b):
    * Kept only ``fog``, ``snow``, ``motion_blur``, ``gaussian_noise``, ``brightness`` and their
      helpers.
    * ``np.float_`` (removed in NumPy 2.0) replaced with ``np.float64``.
    * All randomness drawn from an explicit ``rng`` argument instead of NumPy's global state.
    * ``cv2.cvtColor(RGB2GRAY)`` replaced with the equivalent NumPy weighted sum (drops OpenCV).
    * Grayscale-input branches removed: inputs are always H x W x 3 here.
    * Severity parameter tables and arithmetic are unchanged, so corruption strength matches the
      published ImageNet-C / COCO-C benchmarks.

Every function takes a float or uint8 H x W x 3 array in [0, 255] and returns a float array in
[0, 255]; the caller converts to uint8.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import zoom as scizoom
from skimage.color import hsv2rgb, rgb2hsv

Rng = np.random.RandomState

# ----------------------------------------------------------------------------- helpers


def _plasma_fractal(rng: Rng, mapsize: int = 256, wibbledecay: float = 3) -> np.ndarray:
    """Diamond-square heightmap of side ``mapsize`` (a power of two), scaled to [0, 1]."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array: np.ndarray) -> np.ndarray:
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares() -> None:
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize] = (
            wibbledmean(squareaccum)
        )

    def filldiamonds() -> None:
        size = maparray.shape[0]
        drgrid = maparray[stepsize // 2 : size : stepsize, stepsize // 2 : size : stepsize]
        ulgrid = maparray[0:size:stepsize, 0:size:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:size:stepsize, stepsize // 2 : size : stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2 : size : stepsize, 0:size:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def _clipped_zoom(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2
    return scizoom(
        img[top0 : top0 + ch0, top1 : top1 + ch1], (zoom_factor, zoom_factor, 1), order=1
    )


def _motion_blur_kernel(width: int, sigma: float) -> np.ndarray:
    x = np.arange(width)
    k = np.exp(-(x**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    return k / np.sum(k)


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx < 0:
        shifted = np.roll(image, shift=image.shape[1] + dx, axis=1)
        shifted[:, dx:] = shifted[:, dx - 1 : dx]
    elif dx > 0:
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx : dx + 1]
    else:
        shifted = image

    if dy < 0:
        shifted = np.roll(shifted, shift=image.shape[0] + dy, axis=0)
        shifted[dy:, :] = shifted[dy - 1 : dy, :]
    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy : dy + 1, :]
    return shifted


def _motion_blur(x: np.ndarray, radius: int, sigma: float, angle: float) -> np.ndarray:
    width = radius * 2 + 1
    kernel = _motion_blur_kernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]:
            break  # simulated motion exceeded image borders
        blurred = blurred + kernel[i] * _shift(x, dx, dy)
    return blurred


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _rgb_to_gray(x: np.ndarray) -> np.ndarray:
    # Same coefficients OpenCV uses for COLOR_RGB2GRAY.
    return (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114).astype(x.dtype)


# ------------------------------------------------------------------------- corruptions


def gaussian_noise(x: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Additive Gaussian sensor noise."""
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.asarray(x) / 255.0
    return np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255


def motion_blur(x: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Linear motion blur at a random angle in [-45, 45] degrees."""
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.asarray(x)
    angle = rng.uniform(-45, 45)
    return np.clip(_motion_blur(x, radius=c[0], sigma=c[1], angle=angle), 0, 255)


def fog(x: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Fog: add a plasma-fractal haze and compress contrast."""
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    shape = np.asarray(x).shape
    map_size = _next_power_of_2(int(max(shape)))
    x = np.asarray(x) / 255.0
    max_val = x.max()
    haze = _plasma_fractal(rng, mapsize=map_size, wibbledecay=c[1])[: shape[0], : shape[1]]
    x = x + c[0] * haze[..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def snow(x: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Snow: motion-blurred snowflake layer plus whitening of the scene."""
    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]

    x = np.asarray(x, dtype=np.float32) / 255.0
    snow_layer = rng.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    snow_layer = _clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0
    snow_layer = np.clip(snow_layer.squeeze(), 0, 1)
    snow_layer = _motion_blur(snow_layer, radius=c[4], sigma=c[5], angle=rng.uniform(-135, -45))

    # Round to 8-bit and crop to the image size.
    snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.0
    snow_layer = snow_layer[..., np.newaxis][: x.shape[0], : x.shape[1], :]

    gray = _rgb_to_gray(x).reshape(x.shape[0], x.shape[1], 1)
    x = c[6] * x + (1 - c[6]) * np.maximum(x, gray * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def brightness(x: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Brightness: raise the HSV value channel. Deterministic; ``rng`` is unused."""
    del rng
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    x = rgb2hsv(np.asarray(x) / 255.0)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    return np.clip(hsv2rgb(x), 0, 1) * 255


CORRUPTION_FUNCS = {
    "fog": fog,
    "snow": snow,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "brightness": brightness,
}
