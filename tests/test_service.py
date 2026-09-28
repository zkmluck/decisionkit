import json
import threading
import urllib.error
import urllib.request

import pytest

from decisionkit.client import DecisionClient
from decisionkit.model import DecisionModel
from decisionkit.service import server_for
from decisionkit.train import LinearScorer


@pytest.fixture()
def client():
    from decisionkit.synth import build
    from decisionkit.train import train

    scorer, _ = train(build(60, seed=3), epochs=4, dim=1 << 12, seed=0)
    server = server_for(DecisionModel(scorer), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield DecisionClient(f'http://127.0.0.1:{server.server_address[1]}')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_info_and_roundtrip(client):
    info = client.info()
    assert info['status'] == 'ready'
    assert info['output_token_decoding'] is False
    answer = client.ask('the last run reported FAILED with a traceback',
                        'Which action should run next for this case?', type='choice',
                        options={'read the failing test': 'read the failing test',
                                 'run the test suite': 'run the test suite'})
    assert answer['type'] == 'choice'
    assert abs(sum(answer['distribution'].values()) - 1) < 1e-9


def test_bad_payloads_are_rejected_not_guessed(client):
    with pytest.raises(urllib.error.HTTPError) as caught:
        client.evaluate('state', [{'id': 'q', 'type': 'choice', 'question': 'pick'}])
    assert caught.value.code == 400
    body = json.loads(caught.value.read().decode('utf-8'))
    assert 'choice.options' in body['error']


def test_unknown_paths_are_404(client):
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(client.endpoint + '/api/nope', timeout=5)
    assert caught.value.code == 404
