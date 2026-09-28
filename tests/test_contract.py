import pytest

from decisionkit.contract import ContractError, prepare


def base_question():
    return {'id': 'q1', 'type': 'boolean', 'question': 'Did it pass?'}


def test_boolean_defaults_to_true_false_keys():
    batch = prepare({'state': 'tests passed', 'questions': [base_question()]})
    question = batch.requests[0].questions[0]
    assert question.keys == ['true', 'false']
    assert question.candidates == ['TRUE', 'FALSE']


def test_structured_state_becomes_canonical_json():
    batch = prepare({'state': {'b': 1, 'a': [2]}, 'questions': [base_question()]})
    assert batch.requests[0].state == '{"a":[2],"b":1}'


def test_choice_needs_two_to_255_distinct_candidates():
    with pytest.raises(ContractError, match='2..255'):
        prepare({'state': 'x', 'questions': [{'id': 'q', 'type': 'choice',
                                              'question': 'Pick', 'options': ['only']}]})
    with pytest.raises(ContractError, match='distinct'):
        prepare({'state': 'x', 'questions': [{'id': 'q', 'type': 'choice',
                                              'question': 'Pick', 'options': ['same', 'same']}]})
    batch = prepare({'state': 'x', 'questions': [{'id': 'q', 'type': 'choice',
                                                  'question': 'Pick', 'options': {'a': 'A', 'b': 'B'}}]})
    assert batch.requests[0].questions[0].keys == ['a', 'b']


def test_score_levels_are_ordered_and_bounded():
    with pytest.raises(ContractError, match='score.levels'):
        prepare({'state': 'x', 'questions': [{'id': 'q', 'type': 'score',
                                              'question': 'Risk?', 'levels': ['one']}]})
    batch = prepare({'state': 'x', 'questions': [{'id': 'q', 'type': 'score',
                                                  'question': 'Risk?',
                                                  'levels': ['low', 'high']}]})
    assert batch.requests[0].questions[0].keys == ['0', '1']


def test_state_and_ids_are_required_and_unique():
    with pytest.raises(ContractError, match='state'):
        prepare({'questions': [base_question()]})
    with pytest.raises(ContractError, match='unique'):
        prepare({'state': 'x', 'questions': [base_question(), base_question()]})
    with pytest.raises(ContractError, match='type must be'):
        prepare({'state': 'x', 'questions': [{'id': 'q', 'type': 'nope', 'question': 'x'}]})


def test_batch_wrapper_and_limits():
    batch = prepare({'requests': [{'state': 'a', 'questions': [base_question()]},
                                  {'state': 'b', 'questions': [base_question()]}]})
    assert [request.state for request in batch.requests] == ['a', 'b']
    assert [request.id for request in batch.requests] == ['0', '1']
    with pytest.raises(ContractError, match='1..32'):
        prepare({'requests': []})
    with pytest.raises(ContractError, match='candidate paths'):
        prepare({'state': 'x', 'questions': [
            {'id': f'q{index}', 'type': 'choice', 'question': 'Pick',
             'options': {str(option): f'option {option}' for option in range(255)}}
            for index in range(5)]})


def test_review_thresholds_must_be_probabilities():
    with pytest.raises(ContractError, match='min_probability'):
        prepare({'state': 'x', 'questions': [base_question()], 'min_probability': 2})
    batch = prepare({'state': 'x', 'questions': [base_question()], 'min_probability': 0.75})
    assert batch.min_probability == 0.75
