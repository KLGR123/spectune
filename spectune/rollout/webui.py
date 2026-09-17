"""Interactive live-agent web demo.

``python -m spectune.rollout.webui`` shares its CLI flags and LLM-backend
dispatch with ``python -m spectune.rollout`` / ``python -m spectune.rollout.eval``
(see :mod:`spectune.rollout.cli`), but instead of scoring a batch of samples it
starts a small Flask server: type a query in the browser and watch the trained
agent think, call tools, and answer in real time, one turn/tool-call/result at
a time (see :mod:`spectune.rollout.interactive`).

    python -m spectune.rollout.webui --backend local --model /path/to/merged/huggingface
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request

from spectune.llm import LlmConfig, create_llm_client, vllm_server
from spectune.rollout.cli import add_llm_args, build_rollout_config
from spectune.rollout.interactive import InteractiveAgent

JsonDict = dict[str, Any]

app = Flask(__name__)

_agent: InteractiveAgent | None = None
_agent_lock = threading.Lock()
_conversations: dict[str, list[JsonDict]] = {}
_conversations_lock = threading.Lock()
_model_label = ""


def _build_agent(args: argparse.Namespace) -> tuple[InteractiveAgent, Callable[[], None]]:
    """Instantiate an ``InteractiveAgent`` per ``--backend``.

    Mirrors ``spectune.rollout.cli.run_with_backend``'s backend-selection and
    error-message logic, but returns the built agent (and a shutdown callback)
    instead of immediately running and tearing a batch down -- the web server
    needs the backend (e.g. a local vLLM server) to stay alive for as long as
    the process serves requests.
    """
    config = build_rollout_config(args)

    if args.backend == "local":
        model_path = args.model
        if not model_path:
            print("error: --backend local requires --model <path>", file=sys.stderr)
            sys.exit(1)
        vllm_extra: list[str] = []
        if args.max_model_len is not None:
            vllm_extra += ["--max-model-len", str(args.max_model_len)]
        if args.gpu_memory_utilization is not None:
            vllm_extra += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        cm = vllm_server(
            model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            port=args.vllm_port,
            extra_args=vllm_extra or None,
        )
        base_url, model_name = cm.__enter__()
        agent = InteractiveAgent(config, LlmConfig(base_url=base_url, model=model_name))
        return agent, lambda: cm.__exit__(None, None, None)

    if args.backend == "litellm":
        overrides: dict[str, object] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "max_concurrency": config.max_concurrency,
        }
        if args.model:
            overrides["model"] = args.model
        llm = create_llm_client("litellm", **overrides)
        if not llm.available:
            print(
                "error: litellm endpoint not configured — set LITELLM_API_KEY, "
                "LITELLM_API_BASE, and LITELLM_MODEL (see secrets.env.example)",
                file=sys.stderr,
            )
            sys.exit(1)
        agent = InteractiveAgent(config, llm=llm)
        return agent, lambda: None

    llm_config = LlmConfig()
    if args.model:
        llm_config = dataclasses.replace(llm_config, model=args.model)
    if not llm_config.available:
        print(
            "error: LLM endpoint not configured — set SPECTUNE_LLM_BASE_URL and "
            "SPECTUNE_LLM_MODEL, or use --backend local with --model <path> "
            "(see secrets.env.example)",
            file=sys.stderr,
        )
        sys.exit(1)
    agent = InteractiveAgent(config, llm_config)
    return agent, lambda: None


def _drain(agen: Any) -> Iterator[JsonDict]:
    """Synchronously exhaust an async generator, one event per iteration.

    Flask's dev server is sync; this drives ``InteractiveAgent.stream_query``
    (async, so tool calls can run concurrently) on a private event loop scoped
    to the request without needing an async web framework.
    """
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.close()


@app.route("/")
def index() -> str:
    return render_template_string(HTML, model=_model_label, tools=", ".join(_agent.tool_names if _agent else []))


@app.route("/api/chat", methods=["POST"])
def api_chat() -> Response:
    body = request.get_json(force=True, silent=True) or {}
    text = str(body.get("text", "")).strip()
    conversation_id = str(body.get("conversation_id") or "") or uuid.uuid4().hex
    if not text:
        return jsonify({"error": "empty query"}), 400

    def generate() -> Iterator[str]:
        # One agent (and its tool manager) is shared by every request; a
        # single lock keeps concurrent browser tabs from racing the same
        # LlmClient/ToolManager state instead of queueing them.
        with _agent_lock:
            with _conversations_lock:
                messages = _conversations.get(conversation_id)
            try:
                for event in _drain(_agent.stream_query(text, messages=messages)):
                    if event["type"] == "done":
                        with _conversations_lock:
                            _conversations[conversation_id] = event["messages"]
                    yield json.dumps({**event, "conversation_id": conversation_id}, ensure_ascii=False) + "\n"
            except Exception as exc:  # surface backend errors to the browser, not a bare 500 mid-stream
                yield (
                    json.dumps(
                        {"type": "error", "content": str(exc), "conversation_id": conversation_id},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/api/reset", methods=["POST"])
def api_reset() -> Response:
    body = request.get_json(force=True, silent=True) or {}
    conversation_id = str(body.get("conversation_id") or "")
    with _conversations_lock:
        _conversations.pop(conversation_id, None)
    return jsonify({"ok": True})


HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>Spectune Interactive Agent</title>
<style>
  :root {
    --bg: #f6f8fa; --panel: #fff; --border: #d0d7de;
    --blue: #0969da; --green: #1a7f37; --orange: #953800;
    --purple: #8250df; --gray: #57606a; --red: #cf222e;
    --code-bg: #f6f8fa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: #24292f; font-size: 14px; }
  header { background: #24292f; color: #fff; padding: 12px 24px;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; font-weight: 600; }
  header .sub { font-size: 12px; color: #aaa; }
  .layout { display: flex; flex-direction: column; height: calc(100vh - 44px); }

  #transcript { flex: 1; overflow-y: auto; padding: 16px; }
  #status { padding: 6px 16px; font-size: 12px; color: var(--gray); min-height: 18px; }
  #composer { border-top: 1px solid var(--border); background: var(--panel);
              padding: 10px 16px; display: flex; gap: 10px; align-items: flex-end; }
  #composer textarea { flex: 1; min-height: 64px; max-height: 240px; resize: vertical;
                        font-family: inherit; font-size: 13px; border: 1px solid var(--border);
                        border-radius: 8px; padding: 8px 10px; }
  #composer button { font-size: 13px; border: 1px solid var(--border); border-radius: 6px;
                      padding: 8px 14px; background: #fff; cursor: pointer; height: 38px; }
  #composer button.primary { background: var(--blue); color: #fff; border-color: var(--blue); }
  #composer button:disabled { opacity: .5; cursor: not-allowed; }

  .turn { max-width: 900px; margin: 0 auto 14px; }
  .turn-label { font-size: 11px; font-weight: 700; letter-spacing: .06em;
                text-transform: uppercase; padding: 3px 8px; border-radius: 4px 4px 0 0;
                display: inline-block; }
  .turn-user .turn-label { background: #f0fff4; color: var(--green); }
  .turn-assistant .turn-label { background: #fff8f0; color: var(--orange); }

  .seg { margin-bottom: 6px; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); }
  .seg-label { padding: 3px 10px; font-size: 11px; font-weight: 700;
               letter-spacing: .06em; text-transform: uppercase; }
  .seg-text   .seg-label { background: #f6f8fa; color: var(--gray); }
  .seg-tool   .seg-label { background: #fff0f0; color: var(--red); }
  .seg-result .seg-label { background: #f0f6ff; color: var(--blue); }
  .seg-think  .seg-label { background: #f5f0ff; color: var(--purple); }
  .seg-think  .seg-body  { background: #faf7ff; }
  .seg-user   .seg-label { background: #f0fff4; color: var(--green); }
  .seg-user   .seg-body  { background: #f6fffb; font-family: inherit; font-size: 13px; }
  .seg-error  .seg-label { background: #ffe0e0; color: var(--red); }
  .seg-body { padding: 8px 10px; white-space: pre-wrap;
              font-family: "SFMono-Regular", Consolas, monospace;
              font-size: 12px; line-height: 1.55; background: var(--code-bg); overflow-x: auto; }
  .seg-text .seg-body { font-family: inherit; font-size: 13px; background: #fff; }

  .final-answer { max-width: 900px; margin: 0 auto 14px; border: 1px solid var(--green);
                  border-radius: 8px; background: #f0fff4; padding: 10px 14px; }
  .final-answer .title { font-size: 11px; font-weight: 700; letter-spacing: .06em;
                          text-transform: uppercase; color: var(--green); margin-bottom: 6px; }
  .final-answer ol { padding-left: 20px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>Spectune Interactive Agent</h1>
  <span class="sub">model: {{ model }}</span>
  <span class="sub">tools: {{ tools }}</span>
  <button style="margin-left:auto" onclick="newConversation()">New conversation</button>
</header>
<div class="layout">
  <div id="transcript"></div>
  <div id="status"></div>
  <div id="composer">
    <textarea id="input" placeholder="输入 NMR 谱图数据 / 问题…" onkeydown="onKeyDown(event)"></textarea>
    <button class="primary" id="send-btn" onclick="send()">Send</button>
  </div>
</div>

<script>
let conversationId = localStorage.getItem('spectune_conversation_id') || '';
let busy = false;

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function newConversation() {
  conversationId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  localStorage.setItem('spectune_conversation_id', conversationId);
  document.getElementById('transcript').innerHTML = '';
  document.getElementById('status').textContent = 'New conversation started.';
}
if (!conversationId) newConversation();

function onKeyDown(e) {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); send(); }
}

function segHtml(type, content, name) {
  const map = {
    think:         ['seg-think',  'Think'],
    tool_call:     ['seg-tool',   'Tool Call'],
    tool_response: ['seg-result', name ? `Tool Response · ${esc(name)}` : 'Tool Response'],
    text:          ['seg-text',   'Text'],
    user:          ['seg-user',   'User'],
    error:         ['seg-error',  'Error'],
  };
  const [cls, label] = map[type] || ['seg-text', type];
  const body = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
  return `<div class="seg ${cls}"><div class="seg-label">${label}</div><div class="seg-body">${esc(body)}</div></div>`;
}

function appendAssistantTurn() {
  const transcript = document.getElementById('transcript');
  const div = document.createElement('div');
  div.className = 'turn turn-assistant';
  div.innerHTML = '<div class="turn-label">Assistant</div>';
  transcript.appendChild(div);
  return div;
}

function appendSegment(turnEl, type, content, name) {
  turnEl.insertAdjacentHTML('beforeend', segHtml(type, content, name));
  turnEl.scrollIntoView({block: 'end', behavior: 'smooth'});
}

function appendFinalAnswer(smiles) {
  if (!smiles || !smiles.length) return;
  const transcript = document.getElementById('transcript');
  const items = smiles.map(s => `<li>${esc(s)}</li>`).join('');
  transcript.insertAdjacentHTML('beforeend',
    `<div class="final-answer"><div class="title">Final Candidates</div><ol>${items}</ol></div>`);
}

async function send() {
  if (busy) return;
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text) return;

  const transcript = document.getElementById('transcript');
  transcript.insertAdjacentHTML('beforeend',
    `<div class="turn turn-user"><div class="turn-label">User</div>${segHtml('user', text)}</div>`);
  input.value = '';

  busy = true;
  document.getElementById('send-btn').disabled = true;
  document.getElementById('status').textContent = 'Agent is thinking…';
  const assistantTurn = appendAssistantTurn();

  try {
    const res = await fetch('api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, conversation_id: conversationId}),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      let idx;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        conversationId = event.conversation_id || conversationId;
        if (event.type === 'user') continue;  // already rendered locally
        if (event.type === 'done') {
          appendFinalAnswer(event.smiles);
          document.getElementById('status').textContent = 'Done.';
          continue;
        }
        if (event.type === 'tool_call') document.getElementById('status').textContent = 'Calling tool…';
        else if (event.type === 'error') document.getElementById('status').textContent = 'Error.';
        else document.getElementById('status').textContent = 'Agent is thinking…';
        appendSegment(assistantTurn, event.type, event.content, event.name);
      }
    }
  } catch (e) {
    appendSegment(assistantTurn, 'error', String(e));
    document.getElementById('status').textContent = 'Request failed.';
  } finally {
    busy = false;
    document.getElementById('send-btn').disabled = false;
  }
}
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_llm_args(parser)
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the web server on (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=7865, help="Port to serve the web UI on (default: 7865).")
    return parser


def main(argv: list[str] | None = None) -> None:
    global _agent, _model_label
    parser = build_parser()
    args = parser.parse_args(argv)

    agent, shutdown = _build_agent(args)
    _agent = agent
    _model_label = f"{args.backend}:{args.model or 'default'}"

    print(f"Interactive agent ready — model={_model_label}  tools={','.join(agent.tool_names)}")
    print(f"Open http://localhost:{args.port}")
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
