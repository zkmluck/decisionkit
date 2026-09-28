"""The typed-decisions converter, checked offline on hand-built rows."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'import_typed_decisions.py'
SPEC = importlib.util.spec_from_file_location('import_typed_decisions', MODULE_PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def dataset_row(questions, gold):
    return {'id': 'case-1', 'workflow': 'agent_trace_observability', 'split': 'test',
            'state': json.dumps({'trace_summary': {'tool_errors': 3, 'steps': 11}}),
            'questions': json.dumps(questions), 'gold': json.dumps(gold)}


def test_noul_without_criteria_keeps_the_contract_defaults():
    row = dataset_row({'needs_review': {'type': 'noul', 'instructions': 'Needs review?'}},
                      {'needs_review': {'probabilities': {'false': 0.6, 'true': 0.4}}})
    converted, skipped = TOOL.convert_row(row)
    question = converted['request']['questions'][0]
    assert question['type'] == 'boolean' and 'criteria' not in question
    assert converted['labels'] == {'needs_review': {'false': 0.6, 'true': 0.4}}
    assert skipped == {}


def test_choice_and_score_map_to_options_and_levels():
    row = dataset_row(
        {'action': {'type': 'choice', 'instructions': 'What next?',
                    'criteria': {'stop': 'Halt now.', 'go': 'Continue.'}},
         'risk': {'type': 'score', 'instructions': 'How risky?',
                  'criteria': ['Benign.', 'Low.', 'High.']}},
        {'action': {'probabilities': {'stop': 0.2, 'go': 0.8}},
         'risk': {'probabilities': {'0': 0.1, '1': 0.7, '2': 0.2}}})
    converted, skipped = TOOL.convert_row(row)
    questions = {question['id']: question for question in converted['request']['questions']}
    assert questions['action']['options'] == {'stop': 'Halt now.', 'go': 'Continue.'}
    assert questions['risk']['levels'] == ['Benign.', 'Low.', 'High.']
    assert skipped == {}


def test_unmappable_questions_are_skipped_with_a_reason():
    row = dataset_row(
        {'action': {'type': 'choice', 'instructions': 'What next?'},
         'risk': {'type': 'score', 'instructions': 'How risky?', 'criteria': ['only one']},
         'future': {'type': 'ranking', 'instructions': 'Order these.'}},
        {'action': {'probabilities': {'a': 1.0}}})
    converted, skipped = TOOL.convert_row(row)
    assert converted is None
    assert skipped['choice_without_options'] == 1
    assert skipped['score_without_levels'] == 1
    assert skipped['unknown_type:ranking'] == 1
    assert skipped['empty_row'] == 1


def test_label_keys_must_match_the_prepared_candidates():
    row = dataset_row(
        {'action': {'type': 'choice', 'instructions': 'What next?',
                    'criteria': {'stop': 'Halt now.', 'go': 'Continue.'}}},
        {'action': {'probabilities': {'stop': 0.5, 'continue': 0.5}}})
    converted, skipped = TOOL.convert_row(row)
    assert converted is None
    assert skipped['label_mismatch'] == 1


def test_converted_rows_satisfy_the_contract():
    from decisionkit.contract import prepare

    row = dataset_row(
        {'outcome': {'type': 'choice', 'instructions': 'How did it go?',
                     'criteria': {'success': 'Done correctly.', 'failure': 'Did not finish.'}}},
        {'outcome': {'probabilities': {'success': 0.6, 'failure': 0.4}}})
    converted, _ = TOOL.convert_row(row)
    batch = prepare(converted['request'])
    assert batch.requests[0].state.startswith('{"trace_summary"')
    assert batch.requests[0].questions[0].keys == ['success', 'failure']
