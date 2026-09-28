import math

import numpy as np
import pytest

from decisionkit.calibration import expected_calibration_error, fit_temperature
from decisionkit.metrics import latency, summarize
from decisionkit.train import softmax


def record(gold, prediction, probabilities, status='decided'):
    return {'gold': gold, 'prediction': prediction, 'probabilities': probabilities,
            'top_probability': max(probabilities.values()), 'status': status,
            'correct': gold == prediction}


def test_accuracy_brier_nll_and_coverage():
    records = [record('yes', 'yes', {'yes': 0.8, 'no': 0.2}),
               record('no', 'yes', {'yes': 0.6, 'no': 0.4}, status='review')]
    summary = summarize(records)
    assert summary['accuracy'] == pytest.approx(0.5)
    assert summary['brier_sum'] == pytest.approx((0.04 + 0.04 + 0.36 + 0.36) / 2)
    assert summary['nll'] == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2)
    assert summary['coverage_decided'] == pytest.approx(0.5)
    assert summary['accuracy_when_decided'] == pytest.approx(1.0)
    assert summary['confusion_gold_by_prediction'] == {'yes': {'yes': 1, 'no': 0},
                                                       'no': {'yes': 1, 'no': 0}}


def test_ece_is_zero_when_the_confidences_match_the_hits():
    assert expected_calibration_error([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.0)
    assert expected_calibration_error([0.2, 0.2, 0.2, 0.2], [0, 0, 1, 1]) == pytest.approx(0.3)
    assert expected_calibration_error([], []) == 0.0


def test_temperature_is_fitted_towards_the_targets():
    logits = np.array([[4.0, -4.0]] * 10)
    target = np.array([[0.6, 0.4]] * 10)
    temperature, cross_entropy = fit_temperature(logits, target)
    probabilities = np.stack([softmax(row) for row in logits])
    cross_entropy_at_one = -float((target * np.log(probabilities)).sum())
    assert temperature > 1.0
    assert cross_entropy <= cross_entropy_at_one + 1e-12


def test_latency_percentiles():
    stats = latency([0.001, 0.002, 0.003, 0.004])
    assert stats['n'] == 4 and stats['p50_ms'] == pytest.approx(3.0)
    assert stats['max_ms'] == pytest.approx(4.0)
    assert latency([]) == {'n': 0}
