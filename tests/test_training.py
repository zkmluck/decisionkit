import pytest

from decisionkit.model import DecisionModel
from decisionkit.synth import build, write_splits
from decisionkit.train import load_rows, train


@pytest.fixture(scope='module')
def splits(tmp_path_factory):
    directory = tmp_path_factory.mktemp('synthetic')
    write_splits(build(240, seed=11), directory, validation=40, test=40)
    return {name: load_rows(directory / f'{name}.jsonl')
            for name in ('train', 'validation', 'test')}


def test_splits_never_share_a_case(splits):
    ids = {name: {row['id'] for row in rows} for name, rows in splits.items()}
    assert not ids['train'] & ids['validation']
    assert not ids['train'] & ids['test']
    assert not ids['validation'] & ids['test']


def test_training_reduces_the_loss_and_learns_the_cues(splits):
    scorer, history = train(splits['train'], epochs=16, dim=1 << 15, seed=0)
    assert history[-1]['mean_loss'] < history[0]['mean_loss']
    model = DecisionModel(scorer)
    correct = total = 0
    for row in splits['test']:
        response = model.evaluate(row['request'])
        for answer in response['results'][0]['answers']:
            target = row['labels'][answer['id']]
            gold = max(target, key=target.get)
            prediction = ('true' if answer['value'] else 'false') if answer['type'] == 'boolean' \
                else answer['value'] if answer['type'] == 'choice' else str(answer['level'])
            correct += gold == prediction
            total += 1
    assert total > 0
    assert correct / total >= 0.95


def test_unseen_cue_text_still_decides_correctly(splits):
    scorer, _ = train(splits['train'], epochs=16, dim=1 << 15, seed=0)
    model = DecisionModel(scorer)
    answer = model.ask('the implementation raises on empty input',
                       'Which action should run next for this case?', kind='choice',
                       options={'run the test suite': 'run the test suite',
                                'read the failing test': 'read the failing test',
                                'patch the implementation': 'patch the implementation',
                                'ask the user to clarify': 'ask the user to clarify'})
    assert answer['value'] == 'patch the implementation'
