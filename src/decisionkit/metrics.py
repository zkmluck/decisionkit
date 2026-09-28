"""Metrics with explicit denominators. Accuracy is not calibration."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

import numpy as np

from .calibration import expected_calibration_error


def summarize(records: list[dict]) -> dict:
    """records: one dict per decision, with gold, prediction, probabilities and status."""
    if not records:
        raise ValueError('no decisions to summarise')
    total = len(records)
    correct = np.array([row['correct'] for row in records], dtype=np.float64)
    confidences = np.array([row['top_probability'] for row in records], dtype=np.float64)
    brier = sum(sum((row['probabilities'][key] - (key == row['gold'])) ** 2
                    for key in row['probabilities']) for row in records) / total
    nll = -sum(math.log(max(row['probabilities'][row['gold']], 1e-12)) for row in records) / total
    labels = sorted({row['gold'] for row in records} | {row['prediction'] for row in records})
    confusion = defaultdict(lambda: defaultdict(int))
    for row in records:
        confusion[row['gold']][row['prediction']] += 1
    f1 = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in labels if other != label)
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    decided = [row for row in records if row['status'] == 'decided']
    return {'n': total, 'accuracy': float(correct.mean()),
            'macro_f1': float(np.mean(f1)) if f1 else None,
            'brier_sum': float(brier), 'nll': float(nll),
            'ece_10_equal_width': expected_calibration_error(confidences, correct),
            'coverage_decided': len(decided) / total,
            'accuracy_when_decided': (float(np.mean([row['correct'] for row in decided]))
                                      if decided else None),
            'confusion_gold_by_prediction': {gold: dict(preds) for gold, preds in confusion.items()}}


def latency(seconds: list[float]) -> dict:
    if not seconds:
        return {'n': 0}
    ordered = sorted(seconds)
    return {'n': len(ordered), 'mean_ms': statistics.fmean(ordered) * 1000,
            'p50_ms': ordered[len(ordered) // 2] * 1000,
            'p95_ms': ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000,
            'max_ms': ordered[-1] * 1000}
