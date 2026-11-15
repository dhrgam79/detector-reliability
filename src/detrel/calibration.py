"""Post-hoc temperature scaling on detection confidences (confidence-logit variant).

Each kept detection's confidence ``p`` is treated as a sigmoid output: ``z = logit(p)`` and the
calibrated confidence is ``p' = sigmoid(z / T)`` with a single scalar ``T > 0`` fitted on the
clean calibration split by minimising binary cross-entropy against TP/FP labels.

This is **not** softmax temperature scaling over class logits: torchvision's Faster R-CNN
does not expose the kept detection's logits (see DECISIONS D28). Because the map
``p -> p'`` is strictly increasing, it preserves the order of scores, so detection-to-GT
matching and mAP are unchanged; only the confidence values move.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit

from detrel.model import ImageDetections

logger = logging.getLogger(__name__)


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    """``log(p / (1 - p))`` in float64, with ``p`` clipped to ``[eps, 1 - eps]``."""
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q) - np.log1p(-q)


def apply_temperature(scores: np.ndarray, temperature: float, eps: float) -> np.ndarray:
    """Calibrated confidences ``sigmoid(logit(p) / T)`` (float64).

    Args:
        scores: Confidences in ``[0, 1]``.
        temperature: ``T > 0``; ``T > 1`` softens (lowers high scores), ``T < 1`` sharpens.
        eps: Logit clipping.

    Returns:
        Calibrated confidences, same shape as ``scores``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return expit(logit(scores, eps) / temperature)


def calibrated_detections(
    detections: Sequence[ImageDetections], temperature: float, eps: float
) -> list[ImageDetections]:
    """Copies of ``detections`` with temperature-scaled scores; boxes and labels unchanged.

    No re-thresholding is applied afterwards (DECISIONS D7): the detection set stays identical.
    """
    return [
        ImageDetections(
            d.image_id, d.boxes, d.labels, apply_temperature(d.scores, temperature, eps)
        )
        for d in detections
    ]


def binary_nll(scores: np.ndarray, tp: np.ndarray, temperature: float, eps: float) -> float:
    """Mean binary cross-entropy of calibrated confidences against TP labels."""
    z = logit(scores, eps) / temperature
    y = np.asarray(tp, dtype=np.float64)
    # log(sigmoid(z)) and log(1 - sigmoid(z)) = log(sigmoid(-z)), numerically stable.
    return float(-(y * log_expit(z) + (1.0 - y) * log_expit(-z)).mean())


@dataclass(frozen=True)
class FitResult:
    """Outcome of a temperature fit.

    Attributes:
        temperature: Fitted ``T``.
        nll_before: Mean BCE at ``T = 1``.
        nll_after: Mean BCE at the fitted ``T``.
        n: Number of detections used.
        n_tp: Number of true positives among them.
        at_bound: True if ``T`` hit a search bound (fit is then unreliable).
    """

    temperature: float
    nll_before: float
    nll_after: float
    n: int
    n_tp: int
    at_bound: bool


def fit_temperature(
    scores: np.ndarray,
    tp: np.ndarray,
    bounds: tuple[float, float],
    eps: float,
) -> FitResult:
    """Fit a single temperature by minimising binary cross-entropy.

    The search runs over ``a = 1 / T``: BCE is convex in ``a`` (a one-parameter logistic
    regression without intercept), so a bounded scalar search finds the global minimum.

    Args:
        scores: ``(D,)`` confidences of the fitting detections.
        tp: ``(D,)`` TP labels (same matching as D-ECE).
        bounds: ``(T_min, T_max)``.
        eps: Logit clipping.

    Returns:
        :class:`FitResult`.

    Raises:
        ValueError: If there are no detections.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(tp, dtype=bool)
    if len(s) == 0:
        raise ValueError("cannot fit a temperature on zero detections")
    t_lo, t_hi = bounds
    z = logit(s, eps)
    yf = y.astype(np.float64)

    def nll_of_a(a: float) -> float:
        return float(-(yf * log_expit(a * z) + (1.0 - yf) * log_expit(-a * z)).mean())

    res = minimize_scalar(
        nll_of_a, bounds=(1.0 / t_hi, 1.0 / t_lo), method="bounded", options={"xatol": 1e-9}
    )
    temperature = float(1.0 / res.x)
    at_bound = bool(
        np.isclose(temperature, t_lo, rtol=1e-4) or np.isclose(temperature, t_hi, rtol=1e-4)
    )
    result = FitResult(
        temperature=temperature,
        nll_before=binary_nll(s, y, 1.0, eps),
        nll_after=float(res.fun),
        n=len(s),
        n_tp=int(y.sum()),
        at_bound=at_bound,
    )
    if at_bound:
        logger.warning(
            "Temperature %.4g hit a search bound %s; fit unreliable", temperature, bounds
        )
    logger.info(
        "Fitted T = %.4f on %d detections (%d TP): NLL %.4f -> %.4f",
        result.temperature,
        result.n,
        result.n_tp,
        result.nll_before,
        result.nll_after,
    )
    return result
