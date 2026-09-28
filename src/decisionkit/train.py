"""Fit the candidate scorer with numpy only: soft cross-entropy plus Brier.

The loss is the one a decision head wants: the whole target distribution is
supervised, not just the argmax. Gradients are exact, so no autograd is needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import ContractError, prepare
from .features import DEFAULT_DIM, question_features


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax; a 1-D input is treated as a single row."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def loss_and_gradient(logits: np.ndarray, target: np.ndarray, brier_weight: float):
    """Soft cross-entropy plus brier_weight * sum-of-candidates Brier."""
    probabilities = softmax(logits)
    difference = probabilities - target
    cross_entropy = float(-(target * np.log(np.clip(probabilities, 1e-12, 1.0))).sum())
    brier = float((difference ** 2).sum())
    jacobian = np.diag(probabilities) - np.outer(probabilities, probabilities)
    gradient = difference + brier_weight * (jacobian @ (2.0 * difference))
    return cross_entropy + brier_weight * brier, gradient


def load_rows(path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ContractError(f'{path} contains no rows')
    return rows


def row_examples(row: dict, dim: int):
    """Yield (candidate rows, target distribution, question id) for one dataset row."""
    batch = prepare(row['request'])
    request = batch.requests[0]
    labels = row.get('labels') or {}
    for question in request.questions:
        target_map = labels.get(question.id)
        if not target_map:
            continue
        target = np.array([float(target_map.get(key, 0.0)) for key in question.keys], dtype=np.float64)
        total = target.sum()
        if total <= 0:
            continue
        rows = question_features(request.state, question.text, question.candidates, dim)
        yield rows, target / total, question.id


class LinearScorer:
    """One weight vector shared by every candidate: order cannot change a score."""

    def __init__(self, dim: int = DEFAULT_DIM, weights: np.ndarray | None = None, bias: float = 0.0):
        self.dim = dim
        self.weights = np.zeros(dim, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)

    def logits(self, rows) -> np.ndarray:
        return np.array([float(self.weights[indices] @ values) + self.bias for indices, values in rows])


def train(rows: list[dict], epochs: int = 16, learning_rate: float = 0.02, dim: int = DEFAULT_DIM,
          brier_weight: float = 0.1, l2: float = 1e-5, seed: int = 0, progress=None) -> tuple[LinearScorer, list]:
    scorer = LinearScorer(dim)
    moment1 = np.zeros(dim)
    moment2 = np.zeros(dim)
    scratch = np.zeros(dim)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0
    history = []
    examples = [(row, list(row_examples(row, dim))) for row in rows]
    examples = [item for item in examples if item[1]]
    if not examples:
        raise ContractError('no labelled questions to train on')
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        epoch_loss = 0.0
        updates = 0
        for position in rng.permutation(len(examples)):
            for candidate_rows, target, _ in examples[position][1]:
                logits = scorer.logits(candidate_rows)
                loss, gradient = loss_and_gradient(logits, target, brier_weight)
                touched = []
                decay = l2 / len(candidate_rows)
                for (indices, values), weight in zip(candidate_rows, gradient):
                    np.add.at(scratch, indices, weight * values + decay * scorer.weights[indices])
                    touched.append(indices)
                grad_w = scratch
                step += 1
                moment1 = beta1 * moment1 + (1 - beta1) * grad_w
                moment2 = beta2 * moment2 + (1 - beta2) * (grad_w ** 2)
                corrected1 = moment1 / (1 - beta1 ** step)
                corrected2 = moment2 / (1 - beta2 ** step)
                scorer.weights -= learning_rate * corrected1 / (np.sqrt(corrected2) + epsilon)
                epoch_loss += float(loss)
                updates += 1
                scratch[np.concatenate(touched)] = 0.0
        history.append({'epoch': epoch + 1, 'mean_loss': epoch_loss / max(updates, 1), 'updates': updates})
        if progress:
            progress(history[-1])
    return scorer, history


def save_model(path, scorer: LinearScorer, temperatures: dict | None = None, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=scorer.weights, bias=np.array([scorer.bias]),
             temperatures=json.dumps(temperatures or {}), meta=json.dumps(meta or {}))


def load_model(path) -> tuple[LinearScorer, dict, dict]:
    with np.load(Path(path), allow_pickle=False) as bundle:
        scorer = LinearScorer(len(bundle['weights']), bundle['weights'], float(bundle['bias'][0]))
        return scorer, json.loads(str(bundle['temperatures'])), json.loads(str(bundle['meta']))
