import argparse
import json
import re
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = ROOT / "outputs" / "datasets" / "verl" / "sft.parquet"

app = Flask(__name__)
PARQUET_FILE: Path = DEFAULT_PARQUET
_df_cache: dict[str, list[dict]] = {}


def parse_output_segments(raw: str) -> list[dict]:
    """Split assistant text into typed segments."""
    segments: list[dict] = []

    cleaned = re.sub(
        r"\buser\s*\n\s*(<tool_response>.*?</tool_response>)\s*\nassistant",
        r"\1",
        raw,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"\buser\s*\n(.*?)\n\s*assistant\b",
        lambda m: f"<user_turn>{m.group(1)}</user_turn>" if m.group(1).strip() else "",
        cleaned,
        flags=re.DOTALL,
    )

    TOKEN = re.compile(
        r"(<tool_call>.*?</tool_call>)"
        r"|(<tool_response>.*?</tool_response>)"
        r"|(<think>.*?</think>)"
        r"|(<user_turn>.*?</user_turn>)",
        re.DOTALL,
    )

    last = 0
    for m in TOKEN.finditer(cleaned):
        text = cleaned[last : m.start()].strip()
        if text:
            segments.append({"type": "text", "content": text})

        full = m.group(0)
        if full.startswith("<tool_call>"):
            inner = re.sub(r"^<tool_call>\s*|\s*</tool_call>$", "", full, flags=re.DOTALL).strip()
            try:
                pretty = json.dumps(json.loads(inner), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pretty = inner
            segments.append({"type": "tool_call", "content": pretty})
        elif full.startswith("<tool_response>"):
            inner = re.sub(r"^<tool_response>\s*|\s*</tool_response>$", "", full, flags=re.DOTALL).strip()
            try:
                pretty = json.dumps(json.loads(inner), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pretty = inner
            segments.append({"type": "tool_response", "content": pretty})
        elif full.startswith("<think>"):
            inner = re.sub(r"^<think>\s*|\s*</think>$", "", full, flags=re.DOTALL).strip()
            segments.append({"type": "think", "content": inner})
        else:
            inner = re.sub(r"^<user_turn>\s*|\s*</user_turn>$", "", full, flags=re.DOTALL).strip()
            segments.append({"type": "user_turn", "content": inner})

        last = m.end()

    tail = cleaned[last:].strip()
    if tail:
        segments.append({"type": "text", "content": tail})

    return segments


def load_df(source: str) -> list[dict]:
    if source in _df_cache:
        return _df_cache[source]

    path = PARQUET_FILE
    df = pd.read_parquet(path)
    records = []
    for idx, row in df.iterrows():
        messages = row["messages"]
        turns = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                turns.append({"role": "system", "segments": [{"type": "text", "content": content}]})
            elif role == "user":
                stripped = content.strip()
                if stripped.startswith("<tool_response>"):
                    inner = re.sub(
                        r"^<tool_response>\s*|\s*</tool_response>$", "", stripped, flags=re.DOTALL
                    ).strip()
                    try:
                        pretty = json.dumps(json.loads(inner), ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        pretty = inner
                    turns.append({"role": "tool_response", "segments": [{"type": "tool_response", "content": pretty}]})
                else:
                    turns.append({"role": "user", "segments": [{"type": "text", "content": content}]})
            else:  # assistant
                turns.append({"role": "assistant", "segments": parse_output_segments(content)})

        reward_details = row.get("reward_details", {}) or {}
        if not isinstance(reward_details, dict):
            try:
                reward_details = dict(reward_details)
            except Exception:
                reward_details = {}

        records.append(
            {
                "index": int(idx),
                "sample_id": str(row.get("sample_id", "")),
                "gt_smiles": str(row.get("gt_smiles", "")),
                "score": float(row.get("reward_score", 0.0)),
                "n_rounds": int(row.get("n_rounds", 0)),
                "gt_rank": reward_details.get("gt_rank"),
                "turns": turns,
            }
        )

    _df_cache[source] = records
    return records


HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>SFT Teacher Rollouts Viewer</title>
<style>
  :root {
    --bg: #f6f8fa; --panel: #fff; --border: #d0d7de;
    --blue: #0969da; --green: #1a7f37; --orange: #953800;
    --purple: #8250df; --gray: #57606a; --teal: #0e7490;
    --code-bg: #f6f8fa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: #24292f; font-size: 14px; }
  header { background: #24292f; color: #fff; padding: 12px 24px;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; font-weight: 600; }
  header .sub { font-size: 12px; color: #aaa; }

  .layout { display: flex; height: calc(100vh - 44px); overflow: hidden; }

  #sidebar { width: 260px; min-width: 180px; border-right: 1px solid var(--border);
             overflow-y: auto; background: var(--panel); flex-shrink: 0; }
  .sidebar-header { padding: 8px 12px; font-size: 11px; font-weight: 600; color: var(--gray);
                    text-transform: uppercase; letter-spacing: .05em;
                    border-bottom: 1px solid var(--border); }
  .exp-item { padding: 8px 14px; cursor: pointer; font-size: 13px; color: #24292f;
              border-bottom: 1px solid var(--border); word-break: break-all; line-height: 1.4; }
  .exp-item:hover { background: #f0f6ff; }
  .exp-item.active { background: #dbeafe; font-weight: 600; color: var(--blue); }

  #main { flex: 1; overflow-y: auto; display: flex; flex-direction: column; }

  #controls { padding: 10px 16px; border-bottom: 1px solid var(--border);
              background: var(--panel); display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  #controls label { font-size: 13px; color: var(--gray); }
  #controls select, #controls input { font-size: 13px; border: 1px solid var(--border);
                                      border-radius: 6px; padding: 4px 8px; background: #fff; }
  #traj-counter { font-size: 13px; color: var(--gray); margin-left: auto; }
  #dice-btn { font-size: 13px; padding: 5px 14px; border-radius: 6px; border: 1px solid var(--border);
              background: #0969da; color: #fff; cursor: pointer; font-weight: 600; }
  #dice-btn:hover { background: #0550ae; }
  #dice-btn:disabled { background: #8c959f; cursor: not-allowed; }
  #goto-input { font-size: 13px; border: 1px solid var(--border); border-radius: 6px;
                padding: 4px 8px; background: #fff; width: 80px; }
  #goto-btn { font-size: 13px; padding: 5px 12px; border-radius: 6px; border: 1px solid var(--border);
              background: #6e7781; color: #fff; cursor: pointer; font-weight: 600; }
  #goto-btn:hover { background: #57606a; }
  #viewer { padding: 16px; flex: 1; }

  .traj-card { background: var(--panel); border: 1px solid var(--border);
               border-radius: 8px; margin-bottom: 24px; overflow: hidden; }
  .traj-header { padding: 10px 14px; background: #f0f6ff;
                 border-bottom: 1px solid var(--border);
                 display: flex; gap: 12px; align-items: center; font-size: 13px; flex-wrap: wrap; }
  .traj-header strong { font-size: 14px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 12px; font-weight: 600; }
  .badge-score     { background: #dafbe1; color: var(--green); }
  .badge-score.low { background: #fff0b3; color: #633c01; }
  .badge-score.bad { background: #ffe0e0; color: #cf222e; }
  .badge-round     { background: #eaeef2; color: var(--gray); }
  .badge-rank      { background: #e8f5ff; color: var(--blue); }
  .meta-row { padding: 8px 14px; border-bottom: 1px solid var(--border);
              font-size: 12px; color: var(--gray); word-break: break-all; }
  .meta-row .gts { color: #24292f; font-weight: 500; font-family: monospace; }

  .section { border-top: 1px solid var(--border); }
  .section-title { padding: 7px 14px; font-size: 12px; font-weight: 600;
                   text-transform: uppercase; letter-spacing: .05em; cursor: pointer;
                   display: flex; align-items: center; gap: 6px; user-select: none; }
  .section-title::before { content: "▾"; transition: transform .15s; }
  .section-title.collapsed::before { transform: rotate(-90deg); }
  .section-content { padding: 12px 14px; font-size: 13px; }
  .section-content.hidden { display: none; }

  .turn { margin-bottom: 10px; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); }
  .turn-label { padding: 4px 12px; font-size: 11px; font-weight: 700;
                letter-spacing: .06em; text-transform: uppercase; }
  .turn-system    .turn-label { background: #f0f0f0; color: var(--gray); }
  .turn-user      .turn-label { background: #f0fff4; color: var(--green); }
  .turn-assistant .turn-label { background: #fff8f0; color: var(--orange); }
  .turn-tool_response .turn-label { background: #f0f6ff; color: var(--blue); }

  .seg { margin: 4px 8px 4px 8px; border-radius: 4px; overflow: hidden; border: 1px solid var(--border); }
  .seg-label { padding: 2px 8px; font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
  .seg-text   .seg-label { background: #f6f8fa; color: var(--gray); }
  .seg-tool   .seg-label { background: #fff0f0; color: #cf222e; }
  .seg-result .seg-label { background: #f0f6ff; color: var(--blue); }
  .seg-think  .seg-label { background: #f5f0ff; color: var(--purple); }
  .seg-think  .seg-body  { background: #faf7ff; }
  .seg-user   .seg-label { background: #f0fff4; color: var(--green); }
  .seg-body { padding: 8px 10px; white-space: pre-wrap; font-family: "SFMono-Regular", Consolas, monospace;
              font-size: 12px; line-height: 1.55; background: var(--code-bg); overflow-x: auto; }
  .seg-text .seg-body, .seg-user .seg-body { font-family: inherit; font-size: 13px; background: #fff; }
  .plain-body { padding: 8px 10px; white-space: pre-wrap; font-family: inherit;
                font-size: 13px; line-height: 1.6; }
</style>
</head>
<body>
<header>
  <h1>SFT Teacher Rollouts Viewer</h1>
  <span class="sub">{{ parquet_file }}</span>
</header>
<div class="layout">
  <div id="sidebar">
    <div class="sidebar-header">Datasets</div>
    <div id="src-list"><div style="padding:16px;color:var(--gray);font-size:13px">Loading…</div></div>
  </div>
  <div id="main">
    <div id="controls">
      <label>Filter score ≥</label>
      <input type="number" id="min-score" value="-99" step="0.05" style="width:70px"/>
      <button id="dice-btn" onclick="rollOne()">🎲 Roll</button>
      <input type="number" id="goto-input" placeholder="# index" min="1"
             onkeydown="if(event.key==='Enter') goToIndex()"/>
      <button id="goto-btn" onclick="goToIndex()">Go</button>
      <span id="traj-counter"></span>
    </div>
    <div id="viewer"><p style="color:var(--gray);padding:24px">← Select a dataset from the sidebar</p></div>
  </div>
</div>

<script>
let currentSource = '';
let totalCount = 0;

fetch('api/sources').then(r => r.json()).then(sources => {
  const list = document.getElementById('src-list');
  if (!sources.length) {
    list.innerHTML = '<div style="padding:16px;color:var(--gray);font-size:13px">No parquet files found</div>';
    return;
  }
  list.innerHTML = sources.map(s =>
    `<div class="exp-item" onclick="loadSource('${escAttr(s)}', this)">${escHtml(s)}</div>`
  ).join('');
});

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function escAttr(s) {
  return String(s).replace(/'/g, "\\'");
}

function loadSource(source, el) {
  document.querySelectorAll('.exp-item').forEach(e => e.classList.remove('active'));
  el.classList.add('active');
  currentSource = source;
  totalCount = 0;
  document.getElementById('traj-counter').textContent = '';
  document.getElementById('viewer').innerHTML = '<p style="color:var(--gray);padding:24px">Loading…</p>';
  fetch(`api/count?source=${encodeURIComponent(source)}`)
    .then(r => r.json())
    .then(d => { totalCount = d.count; rollOne(); });
}

function rollOne() {
  if (!currentSource) return;
  const minScore = parseFloat(document.getElementById('min-score').value) || -99;
  const btn = document.getElementById('dice-btn');
  btn.disabled = true;
  btn.textContent = '…';
  fetch(`api/random?source=${encodeURIComponent(currentSource)}&min_score=${minScore}`)
    .then(r => r.json())
    .then(t => {
      btn.disabled = false;
      btn.textContent = '🎲 Roll';
      if (!t) {
        document.getElementById('viewer').innerHTML = '<p style="color:var(--gray);padding:24px">No trajectories match.</p>';
        document.getElementById('traj-counter').textContent = '';
        return;
      }
      document.getElementById('traj-counter').textContent = `#${t.index + 1} / ${totalCount} total`;
      document.getElementById('viewer').innerHTML = renderTraj(t);
    });
}

function goToIndex() {
  if (!currentSource) return;
  const val = parseInt(document.getElementById('goto-input').value);
  if (isNaN(val) || val < 1) return;
  fetch(`api/get?source=${encodeURIComponent(currentSource)}&idx=${val}`)
    .then(r => r.json())
    .then(t => {
      if (!t) {
        document.getElementById('viewer').innerHTML = `<p style="color:var(--gray);padding:24px">Index #${val} not found.</p>`;
        document.getElementById('traj-counter').textContent = '';
        return;
      }
      document.getElementById('traj-counter').textContent = `#${t.index + 1} / ${totalCount} total`;
      document.getElementById('viewer').innerHTML = renderTraj(t);
    });
}

function scoreBadgeClass(s) {
  if (s >= 0.5) return 'badge-score';
  if (s >= 0)   return 'badge-score low';
  return 'badge-score bad';
}

function renderSegs(segs) {
  return segs.map(seg => {
    if (seg.type === 'text') {
      return `<div class="seg seg-text"><div class="seg-label">Text</div><div class="seg-body">${escHtml(seg.content)}</div></div>`;
    } else if (seg.type === 'think') {
      return `<div class="seg seg-think"><div class="seg-label">Think</div><div class="seg-body">${escHtml(seg.content)}</div></div>`;
    } else if (seg.type === 'tool_call') {
      return `<div class="seg seg-tool"><div class="seg-label">Tool Call</div><div class="seg-body">${escHtml(seg.content)}</div></div>`;
    } else if (seg.type === 'tool_response') {
      return `<div class="seg seg-result"><div class="seg-label">Tool Response</div><div class="seg-body">${escHtml(seg.content)}</div></div>`;
    } else if (seg.type === 'user_turn') {
      return `<div class="seg seg-user"><div class="seg-label">User</div><div class="seg-body">${escHtml(seg.content)}</div></div>`;
    }
    return '';
  }).join('');
}

function renderTurn(turn, turnIdx) {
  const roleLabel = {system: 'System', user: 'User', assistant: 'Assistant', tool_response: 'Tool Response'}[turn.role] || turn.role;
  const roleClass = `turn-${turn.role}`;
  const collapsed  = turn.role === 'system' ? ' collapsed' : '';
  const hiddenCls  = turn.role === 'system' ? ' hidden' : '';
  const body = turn.role === 'assistant'
    ? renderSegs(turn.segments)
    : `<div class="plain-body">${escHtml(turn.segments[0]?.content ?? '')}</div>`;
  return `<div class="turn ${roleClass}">
    <div class="turn-label">${escHtml(roleLabel)}</div>
    <div class="section">
      <div class="section-title${collapsed}" onclick="toggleSection(this)">Turn ${turnIdx + 1}</div>
      <div class="section-content${hiddenCls}">${body}</div>
    </div>
  </div>`;
}

function renderTraj(t) {
  const rankBadge = t.gt_rank != null
    ? `<span class="badge badge-rank">rank #${t.gt_rank}</span>`
    : `<span class="badge badge-rank" style="background:#fff0f0;color:#cf222e">not found</span>`;
  const turns = t.turns.map((turn, i) => renderTurn(turn, i)).join('');
  return `<div class="traj-card">
    <div class="traj-header">
      <strong>#${t.index + 1}</strong>
      <span class="badge ${scoreBadgeClass(t.score)}">score ${t.score.toFixed(3)}</span>
      ${rankBadge}
      <span class="badge badge-round">rounds ${t.n_rounds}</span>
      <span style="font-size:12px;color:var(--gray);margin-left:4px">${escHtml(t.sample_id)}</span>
    </div>
    <div class="meta-row">GT: <span class="gts">${escHtml(t.gt_smiles)}</span></div>
    <div class="section">
      <div class="section-title" onclick="toggleSection(this)">Conversation (${t.turns.length} turns)</div>
      <div class="section-content">${turns}</div>
    </div>
  </div>`;
}

function toggleSection(titleEl) {
  titleEl.classList.toggle('collapsed');
  titleEl.nextElementSibling.classList.toggle('hidden');
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML, parquet_file=str(PARQUET_FILE))


@app.route("/api/sources")
def api_sources():
    if not PARQUET_FILE.exists():
        return jsonify([])
    return jsonify([PARQUET_FILE.stem])


@app.route("/api/count")
def api_count():
    source = request.args.get("source", "")
    if not source:
        return jsonify({"count": 0})
    records = load_df(source)
    return jsonify({"count": len(records)})


@app.route("/api/random")
def api_random():
    import random

    source = request.args.get("source", "")
    if not source:
        return jsonify(None)
    min_score = float(request.args.get("min_score", -99))
    records = load_df(source)
    filtered = [r for r in records if r["score"] >= min_score]
    if not filtered:
        return jsonify(None)
    return jsonify(random.choice(filtered))


@app.route("/api/get")
def api_get():
    source = request.args.get("source", "")
    if not source:
        return jsonify(None)
    idx = int(request.args.get("idx", 0))
    records = load_df(source)
    match = next((r for r in records if r["index"] + 1 == idx), None)
    return jsonify(match)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    args = parser.parse_args()

    PARQUET_FILE = Path(args.parquet).resolve()
    print(f"Serving rollout data from: {PARQUET_FILE}")
    print(f"Open http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
