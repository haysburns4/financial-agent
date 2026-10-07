"""A stand-in for mlx_lm.server: answers /v1/models at once, and completions
only after a short "load", like the real server's background model load."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, MODEL, LOAD_SECONDS = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
STARTED = time.monotonic()


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(max(0.0, LOAD_SECONDS - (time.monotonic() - STARTED)))
        self._json(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "h"}}]})

    def log_message(self, *args: object) -> None:
        print("request", flush=True)


print("loading", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
