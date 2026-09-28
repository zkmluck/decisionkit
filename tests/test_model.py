import numpy as np

from decisionkit.model import DecisionModel
from decisionkit.train import LinearScorer


def seeded_model(dim: int = 1 << 12) -> DecisionModel:
    rng = np.random.default_rng(0)
    return DecisionModel(LinearScorer(dim, rng.normal(0, 0.4, dim)))


CHOICE = {'id': 'next', 'type': 'choice', 'question': 'Which action next?',
          'options': {'read': 'read the failing test', 'patch': 'patch the implementation',
                      'ask': 'ask the user to clarify'}}


def test_distributions_are_normalised_and_typed():
    model = seeded_model()
    state = 'the last run reported FAILED with a traceback'
    response = model.evaluate({'state': state, 'questions': [
        {'id': 'spec', 'type': 'boolean', 'question': 'Is the implementation complete?'},
        CHOICE,
        {'id': 'risk', 'type': 'score', 'question': 'How risky?',
         'levels': ['none', 'low', 'medium', 'high']}]})
    answers = response['results'][0]['answers']
    for answer in answers:
        assert abs(sum(answer['distribution'].values()) - 1.0) < 1e-9
        assert answer['generated_tokens'] == 0
        assert answer['status'] in ('decided', 'review')
    assert answers[0]['type'] == 'boolean' and isinstance(answers[0]['value'], bool)
    assert answers[1]['value'] in CHOICE['options']
    assert 0 <= answers[2]['score'] <= 3
    assert response['usage']['generated_tokens'] == 0


def test_candidate_order_cannot_change_the_decision():
    model = seeded_model()
    state = 'one assertion failed and the log points at it'
    forward = model.evaluate({'state': state, 'questions': [CHOICE]})['results'][0]['answers'][0]
    reversed_options = dict(reversed(list(CHOICE['options'].items())))
    backward = model.evaluate({'state': state,
                               'questions': [{**CHOICE, 'options': reversed_options}]})
    backward = backward['results'][0]['answers'][0]
    assert forward['value'] == backward['value']
    assert abs(forward['top_probability'] - backward['top_probability']) < 1e-12
    for key, value in forward['distribution'].items():
        assert abs(value - backward['distribution'][key]) < 1e-12


def test_low_confidence_becomes_review_not_a_guess():
    model = DecisionModel(LinearScorer(1 << 8))  # zero weights: a flat distribution
    answer = model.evaluate({'state': 'anything', 'questions': [CHOICE],
                             'min_probability': 0.6})['results'][0]['answers'][0]
    assert answer['status'] == 'review'
    assert answer['top_probability'] < 0.6


def test_ask_wrapper_returns_one_answer():
    model = seeded_model()
    answer = model.ask('cat README.md', 'How risky is this command?', kind='score',
                       levels=['safe', 'risky'])
    assert answer['id'] == 'q0' and answer['type'] == 'score'
