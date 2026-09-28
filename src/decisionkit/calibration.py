"""Per-primitive temperature scaling, fitted on held-out cases only."""
from __future__ import annotations

import numpy as np

from .train import softmax


def fit_temperature_multi(pairs, grid: np.ndarray | None = None) -> tuple[float, float]:
    """One temperature for many decisions, tolerating different candidate counts.

    Pairs are (logits, target) rows; a 2-D array counts as many rows. Candidate
    counts differ inside a primitive on real data (a score question can offer
    four or five levels), so the cross-entropy is summed per decision rather
    than stacked into one array.
    """
    prepared = [(np.asarray(logits, dtype=np.float64), np.asarray(target, dtype=np.float64))
                for logits, target in pairs]
    if not prepared:
        raise ValueError('no decisions to calibrate')
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 40)) if grid is None else np.asarray(grid, float)
    best, best_ce = 1.0, float('inf')
    for temperature in grid:
        total = 0.0
        for logits, target in prepared:
            probabilities = softmax(logits / temperature)
            total += float(-(target * np.log(np.clip(probabilities, 1e-12, 1.0))).sum())
        if total < best_ce:
            best, best_ce = float(temperature), total
    return best, best_ce


def fit_temperature(logits: np.ndarray, target: np.ndarray,
                    grid: np.ndarray | None = None) -> tuple[float, float]:
    """Return (temperature, cross_entropy) for one decision or a stack of them."""
    return fit_temperature_multi([(logits, target)], grid)


def expected_calibration_error(confidences, correct, bins: int = 10) -> float:
    """Equal-width ECE over the max probability of each decision."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.size == 0:
        return 0.0
    total = confidences.size
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (confidences >= low) & (confidences < high if index < bins - 1 else confidences <= high)
        if mask.any():
            error += mask.sum() / total * abs(correct[mask].mean() - confidences[mask].mean())
    return float(error)
