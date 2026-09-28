"""Score candidates, normalise per question, and return typed answers."""
from __future__ import annotations

import time

import numpy as np

from .contract import API_VERSION, prepare
from .features import DEFAULT_DIM, feature_tokens, question_features
from .train import LinearScorer, softmax

STATUS_DECIDED = 'decided'
STATUS_REVIEW = 'review'


class DecisionModel:
    """The whole runtime: no vocabulary head, no decoding, one pass per question."""

    def __init__(self, scorer: LinearScorer | None = None, temperatures: dict | None = None,
                 meta: dict | None = None):
        self.scorer = scorer or LinearScorer(DEFAULT_DIM)
        self.temperatures = dict(temperatures or {})
        self.meta = dict(meta or {})

    def info(self) -> dict:
        return {'api_version': API_VERSION, 'model': self.meta.get('name', 'decisionkit-linear'),
                'dim': self.scorer.dim, 'types': ['boolean', 'choice', 'score'],
                'temperatures': self.temperatures, 'output_token_decoding': False,
                'candidate_order_matters': False,
                'probability_semantics': 'model distribution; not a success rate unless the '
                                         'temperatures were fitted on your own cases'}

    def evaluate(self, payload) -> dict:
        started = time.perf_counter()
        batch = prepare(payload)
        results, questions, paths, tokens = [], 0, 0, 0
        for request in batch.requests:
            answers = []
            for question in request.questions:
                rows = question_features(request.state, question.text, question.candidates, self.scorer.dim)
                logits = self.scorer.logits(rows)
                temperature = float(self.temperatures.get(question.type, 1.0))
                probabilities = softmax(logits / temperature)
                answers.append(self._answer(question, probabilities,
                                            batch.min_probability, batch.min_margin))
                questions += 1
                paths += len(question.candidates)
                tokens += feature_tokens(request.state, question.text, question.candidates[0])
            results.append({'id': request.id, 'answers': answers})
        return {'api_version': API_VERSION, 'model': self.info()['model'], 'results': results,
                'usage': {'questions': questions, 'candidate_paths': paths, 'input_tokens': tokens,
                          'generated_tokens': 0, 'truncated_inputs': 0,
                          'wall_ms': round((time.perf_counter() - started) * 1000, 3)}}

    @staticmethod
    def _answer(question, probabilities: np.ndarray, min_probability: float, min_margin: float) -> dict:
        ranking = np.argsort(-probabilities, kind='stable')
        top = int(ranking[0])
        second = float(probabilities[ranking[1]]) if len(ranking) > 1 else 0.0
        top_probability = float(probabilities[top])
        margin = top_probability - second
        status = STATUS_DECIDED if top_probability >= min_probability and margin >= min_margin else STATUS_REVIEW
        distribution = {key: float(value) for key, value in zip(question.keys, probabilities)}
        answer = {'id': question.id, 'type': question.type, 'distribution': distribution,
                  'top_probability': round(top_probability, 6), 'margin': round(margin, 6),
                  'status': status, 'generated_tokens': 0}
        if question.type == 'boolean':
            answer.update(value=bool(top == 0), probability=round(distribution['true'], 6))
        elif question.type == 'choice':
            answer.update(value=question.keys[top], description=question.candidates[top])
        else:
            levels = np.arange(len(probabilities))
            answer.update(level=top, score=round(float(levels @ probabilities), 6),
                          legend=question.candidates)
        return answer

    def ask(self, state, question: str, kind: str = 'boolean', options=None, levels=None,
            criteria=None, question_id: str = 'q0') -> dict:
        """Convenience wrapper returning the single answer object."""
        body = {'id': question_id, 'type': kind, 'question': question}
        if options is not None:
            body['options'] = options
        if levels is not None:
            body['levels'] = levels
        if criteria is not None:
            body['criteria'] = criteria
        return self.evaluate({'state': state, 'questions': [body]})['results'][0]['answers'][0]


def load(path) -> DecisionModel:
    from .train import load_model

    scorer, temperatures, meta = load_model(path)
    return DecisionModel(scorer, temperatures, meta)
