"""Launch and manage a local vLLM OpenAI-compatible server."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _poll_ready(url: str, timeout: float = 600.0, interval: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3.0)  # noqa: S310
            return True
        except Exception:
            time.sleep(interval)
    return False


@contextlib.contextmanager
def vllm_server(
    model_path: str,
    *,
    tensor_parallel_size: int = 1,
    port: int | None = None,
    extra_args: list[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Start a vLLM OpenAI-compatible server and yield ``(base_url, model_name)``.

    The server process is terminated when the context exits, even on error.

    Args:
        model_path: HuggingFace model directory or model id served by vLLM.
        tensor_parallel_size: Number of GPUs to shard the model across.
        port: TCP port; a random free port is chosen when ``None``.
        extra_args: Extra flags forwarded verbatim to ``vllm.entrypoints.openai.api_server``.

    Yields:
        ``(base_url, model_name)`` — *base_url* is ``http://127.0.0.1:<port>/v1`` and
        *model_name* equals *model_path* (matches ``--served-model-name``).
    """
    port = port or _free_port()
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--served-model-name",
        model_path,
        *(extra_args or []),
    ]
    print(
        f"starting local vLLM server  "
        f"model={model_path}  port={port}  tp={tensor_parallel_size}"
    )
    proc = subprocess.Popen(cmd)  # stdout/stderr inherited so server logs are visible
    base_url = f"http://127.0.0.1:{port}/v1"
    health_url = f"http://127.0.0.1:{port}/health"
    try:
        print("waiting for vLLM server to be ready (up to 600 s)…", flush=True)
        if not _poll_ready(health_url, timeout=600.0):
            raise RuntimeError(
                f"vLLM server on port {port} did not become ready within 600 s"
            )
        print(f"vLLM server ready → {base_url}")
        yield base_url, model_path
    finally:
        print("shutting down vLLM server…")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("vLLM server stopped.")


__all__ = ["vllm_server"]
