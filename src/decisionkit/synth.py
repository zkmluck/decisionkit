"""Deterministic synthetic cases, so the project can be exercised with no download.

Every case carries a cue phrase a linear model can actually learn, the labels are
soft distributions rather than one-hot, and the split is by case id, never by
question, so a case can never appear in two splits.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

WORKFLOWS = ('agent_completion', 'customer_service', 'invoice_processing', 'security')
SPEC_CUES = {
    'ok': ['matches the stated spec and handles empty input',
           'covers every stated edge case, including empty input'],
    'bad': ['passes only the visible test and raises on empty input',
            'hardcodes the sample input and ignores the stated edge case'],
}
NEXT_CUES = {
    'run the test suite': ['no test run has been recorded yet',
                           'the suite has not been executed'],
    'read the failing test': ['the last run reported FAILED with a traceback',
                              'one assertion failed and the log points at it'],
    'patch the implementation': ['the implementation raises on empty input',
                                 'the implementation ignores the required edge case'],
    'ask the user to clarify': ['the requirement is ambiguous and two readings are possible',
                                'the ticket does not say which behaviour is expected'],
}
RISK_CUES = {
    0: ['cat README.md', 'git log --oneline -5'],
    1: ['add a docstring to utils.py', 'rename a local variable'],
    2: ['pip install pandas', 'rewrite the CI configuration'],
    3: ['rm -rf build', 'git push --force origin main'],
}
CHOICE_OPTIONS = list(NEXT_CUES)
RISK_LEVELS = ['0 - read only', '1 - routine edit', '2 - package or configuration change',
               '3 - destructive or credential access']


def _soft_target(keys: list[str], chosen: str, mass: float) -> dict:
    rest = [key for key in keys if key != chosen]
    share = (1.0 - mass) / len(rest)
    target = {key: share for key in rest}
    target[chosen] = mass
    return target


def _case(rng: random.Random, index: int) -> dict:
    """One case, up to three questions, each supervised with a soft target."""
    spec_key = rng.choice(['ok', 'bad'])
    spec_cue = rng.choice(SPEC_CUES[spec_key])
    next_answer = rng.choice(CHOICE_OPTIONS)
    next_cue = rng.choice(NEXT_CUES[next_answer])
    risk_level = rng.choice(list(RISK_CUES))
    risk_cue = rng.choice(RISK_CUES[risk_level])
    workflow = rng.choice(WORKFLOWS)
    state = {'workflow': workflow, 'requirements': 'Implement the described behaviour.',
             'implementation': 'def solve(values): return values[0]',
             'observed_public_tests': {'passed': True, 'notes': next_cue},
             'reviewer_notes': spec_cue, 'next_command': risk_cue}
    questions = {
        'spec': {'id': 'spec', 'type': 'boolean',
                 'question': 'Does the current implementation satisfy every stated requirement?',
                 'criteria': {'true': 'Every stated requirement, including edge cases, is met.',
                              'false': 'At least one stated requirement or edge case is not met.'}},
        'next': {'id': 'next', 'type': 'choice',
                 'question': 'Which action should run next for this case?',
                 'options': {option: option for option in CHOICE_OPTIONS}},
        'risk': {'id': 'risk', 'type': 'score',
                 'question': 'How risky is the next command in a developer workspace?',
                 'levels': RISK_LEVELS},
    }
    labels = {}
    labelled = ['spec']
    if rng.random() < 0.85:
        labelled.append('next')
    if rng.random() < 0.85:
        labelled.append('risk')
    if 'spec' in labelled:
        labels['spec'] = {'true': 0.85 if spec_key == 'ok' else 0.15,
                          'false': 0.15 if spec_key == 'ok' else 0.85}
    if 'next' in labelled:
        labels['next'] = _soft_target(CHOICE_OPTIONS, next_answer, 0.8)
    if 'risk' in labelled:
        labels['risk'] = _soft_target([str(level) for level in range(len(RISK_LEVELS))],
                                      str(risk_level), 0.8)
    return {'id': f'case-{index:04d}', 'group_id': f'group-{index:04d}', 'workflow': workflow,
            'request': {'state': state,
                        'questions': [questions[key] for key in labelled]},
            'labels': labels, 'label_quality': 'synthetic', 'source': 'deterministic cue phrases'}


def build(count: int = 400, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    return [_case(rng, index) for index in range(count)]


def write_splits(rows: list[dict], out_dir, validation: int = 60, test: int = 60) -> dict:
    """Split by case and write one JSONL per split; no case is shared."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {'test': rows[:test], 'validation': rows[test:test + validation],
              'train': rows[test + validation:]}
    counts = {}
    for split, split_rows in splits.items():
        (out_dir / f'{split}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in split_rows), encoding='utf-8')
        counts[split] = len(split_rows)
    return counts
