"""Loopback HTTP service. Standard library only; binds 127.0.0.1 by default."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .contract import ContractError

MAX_BYTES = 1_000_000


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'decisionkit'

        def _send(self, status: int, value: dict) -> None:
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ('/health', '/api/info'):
                self._send(200, {'status': 'ready', **model.info()})
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self):  # noqa: N802
            if self.path != '/api/evaluate':
                self._send(404, {'error': 'not found'})
                return
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BYTES:
                self._send(413, {'error': f'request must be 1..{MAX_BYTES} bytes'})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode('utf-8'),
                                     parse_constant=lambda value: (_ for _ in ()).throw(
                                         ValueError(f'nonfinite JSON: {value}')))
                self._send(200, model.evaluate(payload))
            except ContractError as exc:
                self._send(400, {'error': str(exc)})
            except (ValueError, TypeError, KeyError) as exc:
                self._send(400, {'error': str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                print(f'inference error: {type(exc).__name__}: {exc}', flush=True)
                self._send(500, {'error': 'decision failed; consult the service log'})

        def log_message(self, *args):
            return

    return Handler


def server_for(model, host: str = '127.0.0.1', port: int = 0) -> ThreadingHTTPServer:
    """A startable server; the caller decides when to serve_forever/shutdown."""
    return ThreadingHTTPServer((host, port), make_handler(model))


def serve(model, host: str = '127.0.0.1', port: int = 8765) -> None:
    server = server_for(model, host, port)
    print(json.dumps({'event': 'ready', 'host': host, 'port': port, **model.info()}, indent=2), flush=True)
    server.serve_forever()
