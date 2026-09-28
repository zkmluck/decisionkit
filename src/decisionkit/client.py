"""Standard-library client for the loopback service."""
from __future__ import annotations

import json
import urllib.request


class DecisionClient:
    def __init__(self, endpoint: str = 'http://127.0.0.1:8765', timeout: float = 30.0):
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(self.endpoint + path,
                                         data=json.dumps(payload).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.endpoint + path, timeout=self.timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    def info(self) -> dict:
        return self._get('/api/info')

    def evaluate(self, state, questions, **options) -> dict:
        return self._post('/api/evaluate', {'state': state, 'questions': questions, **options})

    def ask(self, state, question: str, **options) -> dict:
        """Return the single answer object instead of the envelope."""
        body = {'id': 'q0', 'type': options.pop('type', 'boolean'), 'question': question, **options}
        return self.evaluate(state, [body])['results'][0]['answers'][0]
