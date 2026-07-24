"""Mock upstream servers that exercise tools over actual local HTTP.

These helpers implement the upstream wire protocols on localhost. Tools use
their normal HTTP clients, while responses remain deterministic and require
no external credentials.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

JsonDict = dict[str, Any]


def _read_json_body(handler: BaseHTTPRequestHandler) -> JsonDict:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length).decode("utf-8") if length else ""
    return json.loads(raw) if raw else {}


@contextmanager
def run_mock_web_search_server(
    handle_request: Callable[[JsonDict], list[JsonDict]],
) -> Iterator[str]:
    """Serve Volcengine-style SSE search responses.

    ``handle_request`` receives the decoded POST body and returns a list of
    JSON-serializable result objects, each streamed back as one
    ``data: <json>`` SSE event, terminated by ``data: [DONE]``.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = _read_json_body(self)
            results = handle_request(body)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for result in results:
                self.wfile.write(f"data: {json.dumps(result, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, format_: str, *args: Any) -> None:  # noqa: N802
            pass

    with _serve(Handler, path="/search") as api_url:
        yield api_url


@contextmanager
def run_mock_json_server(
    handle_request: Callable[[JsonDict], JsonDict],
    *,
    path: str = "/",
) -> Iterator[str]:
    """Serve plain ``POST json -> POST json`` responses.

    This is the shared wire protocol behind SandboxFusion (``code_interpreter``)
    and the NMR HTTP backends (``nmr_generate`` / ``nmr_repair`` / ``nmr_rerank``).
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = _read_json_body(self)
            payload = handle_request(body)
            response_body = json.dumps(payload, ensure_ascii=False).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format_: str, *args: Any) -> None:  # noqa: N802
            pass

    with _serve(Handler, path=path) as api_url:
        yield api_url


@contextmanager
def run_mock_get_json_server(
    handle_request: Callable[[str, dict[str, list[str]]], JsonDict],
    *,
    status_code: Callable[[str, dict[str, list[str]]], int] | None = None,
) -> Iterator[str]:
    """Serve plain ``GET ?query -> JSON`` responses, routed by request path.

    This is the shared wire protocol behind the literature-search tools
    (``semantic_scholar_search``, ``crossref_search``, ``wikipedia_search``),
    which issue plain HTTPS GET requests with query-string parameters.
    ``handle_request`` receives the request path (without query string) and
    the parsed query parameters, and returns the JSON body to send back.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            payload = handle_request(parsed.path, query)
            code = status_code(parsed.path, query) if status_code else 200
            response_body = json.dumps(payload, ensure_ascii=False).encode()

            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format_: str, *args: Any) -> None:  # noqa: N802
            pass

    with _serve(Handler, path="") as base_url:
        yield base_url


def run_mock_sandbox_server(
    handle_request: Callable[[JsonDict], JsonDict],
) -> Iterator[str]:
    """Serve SandboxFusion-style ``{"status": ..., "run_result": {...}}`` responses."""
    return run_mock_json_server(handle_request, path="/run_code")


@contextmanager
def _serve(handler_cls: type[BaseHTTPRequestHandler], *, path: str) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}{path}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
