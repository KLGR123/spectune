import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJ_DIR = ROOT / "outputs" / "trajectories" / "eval"

app = Flask(__name__)
TRAJ_DIR: Path = DEFAULT_TRAJ_DIR


def parse_output_segments(raw: str) -> list[dict]:
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


def parse_messages(messages: list[dict]) -> tuple[str, str, list[dict]]:
    """Return (system_prompt, user_input, turns).

    turns is a list of dicts: {"role": "user"|"assistant", "segments": [...]}
    The first user message is returned separately as user_input.
    """
    system_prompt = ""
    user_input = ""
    turns: list[dict] = []
    first_user_seen = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        if role == "system":
            system_prompt = content
        elif role == "user":
            if not first_user_seen:
                first_user_seen = True
                user_input = content
            else:
                # Subsequent user messages are tool responses or continuations
                if "<tool_response>" in content:
                    segs = parse_output_segments(content)
                else:
                    segs = [{"type": "user_turn", "content": content}]
                turns.append({"role": "user", "segments": segs})
        elif role == "assistant":
            segs = parse_output_segments(content)
            turns.append({"role": "assistant", "segments": segs})

    return system_prompt, user_input, turns


HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>Eval Rollouts Viewer</title>
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
  .layout { display: flex; height: calc(100vh - 44px); overflow: hidden; }

  /* sidebar */
  #sidebar { width: 280px; min-width: 200px; border-right: 1px solid var(--border);
             overflow-y: auto; background: var(--panel); padding: 8px 0; flex-shrink: 0; }
  .file-link { display: block; padding: 6px 14px; cursor: pointer;
               color: var(--blue); font-size: 13px; border-bottom: 1px solid #f0f0f0;
               word-break: break-all; line-height: 1.4; }
  .file-link:hover { background: #f0f6ff; }
  .file-link.active { background: #dbeafe; font-weight: 600; }
  .file-meta { font-size: 11px; color: var(--gray); margin-top: 2px; word-break: break-all; }

  /* main */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #controls { padding: 10px 16px; border-bottom: 1px solid var(--border);
              background: var(--panel); display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  #controls label { font-size: 13px; color: var(--gray); }
  #controls select, #controls input { font-size: 13px; border: 1px solid var(--border);
                                      border-radius: 6px; padding: 4px 8px; background: #fff; }
  #controls button { font-size: 13px; border: 1px solid var(--border); border-radius: 6px;
                     padding: 4px 10px; background: #fff; cursor: pointer; }
  #controls button:disabled { color: #8c959f; cursor: not-allowed; }
  #traj-counter { font-size: 13px; color: var(--gray); margin-left: auto; }
  #viewer { flex: 1; overflow-y: auto; padding: 16px; }

  /* trajectory card */
  .traj-card { background: var(--panel); border: 1px solid var(--border);
               border-radius: 8px; margin-bottom: 24px; overflow: hidden; }
  .traj-header { padding: 10px 14px; background: #f0f6ff;
                 border-bottom: 1px solid var(--border);
                 display: flex; gap: 16px; align-items: center; font-size: 13px; flex-wrap: wrap; }
  .traj-header strong { font-size: 14px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 12px; font-weight: 600; }
  .badge-score     { background: #dafbe1; color: var(--green); }
  .badge-score.low { background: #fff0b3; color: #633c01; }
  .badge-score.bad { background: #ffe0e0; color: var(--red); }
  .badge-rounds    { background: #eaeef2; color: var(--gray); }
  .badge-hit       { background: #dafbe1; color: var(--green); }
  .badge-miss      { background: #ffe0e0; color: var(--red); }
  .meta-row { padding: 8px 14px; border-bottom: 1px solid var(--border);
              font-size: 12px; color: var(--gray); display: flex; flex-wrap: wrap; gap: 12px; }
  .meta-row .val { color: #24292f; font-weight: 500; font-family: "SFMono-Regular", Consolas, monospace; }

  /* reward details */
  .reward-section .section-title { background: #f5f0ff; color: var(--purple); }
  .reward-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; padding: 4px 0; }
  .reward-item { font-size: 12px; }
  .reward-item .rk { color: var(--gray); }
  .reward-item .rv { font-weight: 600; font-family: "SFMono-Regular", Consolas, monospace; }
  .candidates-list { margin-top: 8px; }
  .candidate { display: flex; align-items: center; gap: 8px; padding: 3px 0;
               font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; }
  .candidate .rank-num { color: var(--gray); min-width: 20px; }
  .candidate .smiles   { flex: 1; }
  .candidate.gt-match  { color: var(--green); font-weight: 600; }

  /* sections */
  .section { border-top: 1px solid var(--border); }
  .section-title { padding: 7px 14px; font-size: 12px; font-weight: 600;
                   text-transform: uppercase; letter-spacing: .05em; cursor: pointer;
                   display: flex; align-items: center; gap: 6px; user-select: none; }
  .section-title::before { content: "▾"; transition: transform .15s; }
  .section-title.collapsed::before { transform: rotate(-90deg); }
  .section-content { padding: 12px 14px; font-size: 13px; }
  .section-content.hidden { display: none; }

  .system-section .section-title { background: #f0f0f0; color: var(--gray); }
  .input-section  .section-title { background: #f0fff4; color: var(--green); }
  .output-section .section-title { background: #fff8f0; color: var(--orange); }
  .input-text { white-space: pre-wrap; line-height: 1.6; }

  /* conversation turns */
  .turn { margin-bottom: 10px; }
  .turn-label { font-size: 11px; font-weight: 700; letter-spacing: .06em;
                text-transform: uppercase; padding: 3px 8px; border-radius: 4px 4px 0 0;
                display: inline-block; }
  .turn-assistant .turn-label { background: #fff8f0; color: var(--orange); }
  .turn-user      .turn-label { background: #f0fff4; color: var(--green); }

  /* segments */
  .seg { margin-bottom: 6px; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); }
  .seg-label { padding: 3px 10px; font-size: 11px; font-weight: 700;
               letter-spacing: .06em; text-transform: uppercase; }
  .seg-text   .seg-label { background: #f6f8fa; color: var(--gray); }
  .seg-tool   .seg-label { background: #fff0f0; color: var(--red); }
  .seg-result .seg-label { background: #f0f6ff; color: var(--blue); }
  .seg-think  .seg-label { background: #f5f0ff; color: var(--purple); }
  .seg-think  .seg-body  { background: #faf7ff; }
  .seg-user   .seg-label { background: #f0fff4; color: var(--green); }
  .seg-user   .seg-body  { background: #f6fffb; font-family: inherit; }
  .seg-body { padding: 8px 10px; white-space: pre-wrap;
              font-family: "SFMono-Regular", Consolas, monospace;
              font-size: 12px; line-height: 1.55; background: var(--code-bg); overflow-x: auto; }
  .seg-text .seg-body { font-family: inherit; font-size: 13px; background: #fff; }
</style>
</head>
<body>
<header>
  <h1>Eval Rollouts Viewer</h1>
  <span style="font-size:12px;color:#aaa;">{{ traj_dir }}</span>
</header>
<div class="layout">
  <div id="sidebar">
    {% for f in files %}
    <a class="file-link" onclick="loadFile('{{ f.name }}', this)">
      {{ f.name }}
      <div class="file-meta">{{ f.count }} samples</div>
    </a>
    {% endfor %}
  </div>
  <div id="main">
    <div id="controls">
      <label>Filter score ≥</label>
      <input type="number" id="min-score" value="-99" step="0.05" style="width:70px"
             onchange="applyFilters()"/>
      <label>Sort by</label>
      <select id="sort-by" onchange="applyFilters()">
        <option value="index">Index</option>
        <option value="score_desc">Score ↓</option>
        <option value="score_asc">Score ↑</option>
      </select>
      <button id="prev-page" onclick="changePage(-1)">← Prev</button>
      <label>Page
        <input type="number" id="page-number" min="1" value="1" style="width:58px"
               onkeydown="if(event.key==='Enter') goToPage()"/>
      </label>
      <span id="page-total">/ 1</span>
      <button id="next-page" onclick="changePage(1)">Next →</button>
      <span id="traj-counter"></span>
    </div>
    <div id="viewer"><p style="color:var(--gray);padding:24px">← Select a file from the sidebar</p></div>
  </div>
</div>

<script>
let currentFile = '';
let currentPage = 1;
let totalPages = 1;
let requestSerial = 0;

function loadFile(name, el) {
  document.querySelectorAll('.file-link').forEach(a => a.classList.remove('active'));
  el.classList.add('active');
  currentFile = name;
  loadPage(1);
}

function applyFilters() { if (currentFile) loadPage(1); }
function changePage(d)   { loadPage(currentPage + d); }
function goToPage()      { loadPage(parseInt(document.getElementById('page-number').value, 10)); }

async function loadPage(page) {
  if (!currentFile || !Number.isFinite(page)) return;
  page = Math.max(1, Math.min(page, totalPages));
  const minScore = parseFloat(document.getElementById('min-score').value) || -99;
  const sortBy   = document.getElementById('sort-by').value;
  const serial = ++requestSerial;
  const viewer = document.getElementById('viewer');
  viewer.innerHTML = '<p style="color:var(--gray);padding:24px">Loading…</p>';

  const params = new URLSearchParams({
    file: currentFile, page: String(page), page_size: '10',
    min_score: String(minScore), sort: sortBy,
  });

  try {
    const res = await fetch('api/file?' + params);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (serial !== requestSerial) return;

    currentPage = data.page;
    totalPages  = data.pages;
    document.getElementById('page-number').value = currentPage;
    document.getElementById('page-number').max   = totalPages;
    document.getElementById('page-total').textContent = `/ ${totalPages}`;
    document.getElementById('prev-page').disabled = currentPage <= 1;
    document.getElementById('next-page').disabled = currentPage >= totalPages;

    const first = data.filtered_total ? (currentPage - 1) * data.page_size + 1 : 0;
    const last  = first ? first + data.items.length - 1 : 0;
    document.getElementById('traj-counter').textContent =
      `Showing ${first}–${last} of ${data.filtered_total} (${data.total} total)`;
    viewer.innerHTML = data.items.length
      ? data.items.map(renderTraj).join('')
      : '<p style="color:var(--gray);padding:24px">No trajectories match.</p>';
  } catch (e) {
    if (serial !== requestSerial) return;
    viewer.innerHTML = `<p style="color:var(--red);padding:24px">Load failed: ${esc(e.message)}</p>`;
  }
}

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function scoreBadgeClass(s) {
  if (s >= 0.5) return 'badge-score';
  if (s >= 0)   return 'badge-score low';
  return 'badge-score bad';
}

function renderSeg(seg) {
  if (seg.type === 'text')          return `<div class="seg seg-text"><div class="seg-label">Text</div><div class="seg-body">${esc(seg.content)}</div></div>`;
  if (seg.type === 'think')         return `<div class="seg seg-think"><div class="seg-label">Think</div><div class="seg-body">${esc(seg.content)}</div></div>`;
  if (seg.type === 'tool_call')     return `<div class="seg seg-tool"><div class="seg-label">Tool Call</div><div class="seg-body">${esc(seg.content)}</div></div>`;
  if (seg.type === 'tool_response') return `<div class="seg seg-result"><div class="seg-label">Tool Response</div><div class="seg-body">${esc(seg.content)}</div></div>`;
  return `<div class="seg seg-user"><div class="seg-label">User</div><div class="seg-body">${esc(seg.content)}</div></div>`;
}

function renderTurns(turns) {
  return turns.map(turn => {
    const cls   = turn.role === 'assistant' ? 'turn-assistant' : 'turn-user';
    const label = turn.role === 'assistant' ? 'Assistant' : 'User';
    return `<div class="turn ${cls}"><div class="turn-label">${label}</div>${turn.segments.map(renderSeg).join('')}</div>`;
  }).join('');
}

function renderReward(t) {
  const d = t.reward_details || {};
  const gtRank = d.gt_rank != null ? d.gt_rank : '?';
  const hitBadge = (d.gt_rank === 0)
    ? `<span class="badge badge-hit">Top-1 Hit</span>`
    : `<span class="badge badge-miss">Rank ${d.gt_rank != null ? d.gt_rank + 1 : '?'}</span>`;

  const components = d.weighted_components || {};
  const compHtml = Object.entries(components).map(([k, v]) =>
    `<div class="reward-item"><span class="rk">${esc(k)}</span>: <span class="rv">${typeof v === 'number' ? v.toFixed(3) : esc(String(v))}</span></div>`
  ).join('');

  const candidates = (d.answer_candidates || []).map((s, i) => {
    const isGt = s === t.gt_smiles;
    return `<div class="candidate${isGt ? ' gt-match' : ''}"><span class="rank-num">${i+1}.</span><span class="smiles">${esc(s)}</span>${isGt ? ' ✓' : ''}</div>`;
  }).join('');

  return `
  <div class="section reward-section">
    <div class="section-title collapsed" onclick="toggleSection(this)">Reward Details</div>
    <div class="section-content hidden">
      <div style="margin-bottom:8px">${hitBadge}</div>
      <div class="reward-grid">${compHtml}</div>
      ${candidates ? `<div class="candidates-list"><div style="font-size:12px;color:var(--gray);margin:8px 0 4px">Candidates (GT: <span style="font-family:monospace">${esc(t.gt_smiles)}</span>):</div>${candidates}</div>` : ''}
    </div>
  </div>`;
}

function renderTraj(t) {
  const systemHtml = t.system_prompt ? `
    <div class="section system-section">
      <div class="section-title collapsed" onclick="toggleSection(this)">System</div>
      <div class="section-content hidden"><div class="input-text">${esc(t.system_prompt)}</div></div>
    </div>` : '';

  return `
  <div class="traj-card">
    <div class="traj-header">
      <strong>#${t.index + 1}</strong>
      <span class="badge ${scoreBadgeClass(t.score)}">score ${t.score.toFixed(3)}</span>
      <span class="badge badge-rounds">${t.n_rounds} round${t.n_rounds !== 1 ? 's' : ''}</span>
      <span style="font-size:12px;color:var(--gray)">id: ${esc(t.sample_id)}</span>
    </div>
    <div class="meta-row">
      <span>GT: <span class="val">${esc(t.gt_smiles)}</span></span>
      <span>tool_calls: <span class="val">${t.tool_call_count ?? '?'}</span></span>
    </div>
    ${systemHtml}
    <div class="section input-section">
      <div class="section-title" onclick="toggleSection(this)">Input</div>
      <div class="section-content"><div class="input-text">${esc(t.user_input)}</div></div>
    </div>
    <div class="section output-section">
      <div class="section-title" onclick="toggleSection(this)">Conversation</div>
      <div class="section-content">${renderTurns(t.turns)}</div>
    </div>
    ${renderReward(t)}
  </div>`;
}

function toggleSection(el) {
  el.classList.toggle('collapsed');
  el.nextElementSibling.classList.toggle('hidden');
}
</script>
</body>
</html>
"""


def _count_file(path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


@app.route("/")
def index():
    files = []
    for path in sorted(TRAJ_DIR.glob("*.jsonl")):
        files.append({"name": path.name, "count": _count_file(path)})
    return render_template_string(HTML, files=files, traj_dir=str(TRAJ_DIR))


@app.route("/api/file")
def api_file():
    name = request.args.get("file", "")
    if not name or "/" in name or "\\" in name:
        abort(400)
    target = (TRAJ_DIR / name).resolve()
    if not target.is_relative_to(TRAJ_DIR.resolve()) or not target.exists():
        abort(404)

    stat = target.stat()
    results = list(_load_file(str(target), stat.st_mtime_ns, stat.st_size))

    min_score = request.args.get("min_score", default=-99.0, type=float)
    sort_by = request.args.get("sort", "index")
    page = max(request.args.get("page", default=1, type=int) or 1, 1)
    page_size = min(max(request.args.get("page_size", default=10, type=int) or 10, 1), 50)

    filtered = [item for item in results if item["score"] >= min_score]
    if sort_by == "score_desc":
        filtered.sort(key=lambda item: item["score"], reverse=True)
    elif sort_by == "score_asc":
        filtered.sort(key=lambda item: item["score"])

    filtered_total = len(filtered)
    pages = max((filtered_total + page_size - 1) // page_size, 1)
    page = min(page, pages)
    start = (page - 1) * page_size
    return jsonify(
        {
            "items": filtered[start : start + page_size],
            "total": len(results),
            "filtered_total": filtered_total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }
    )


@lru_cache(maxsize=8)
def _load_file(path: str, _mtime_ns: int, _size: int) -> tuple[dict, ...]:
    results = []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = d.get("messages", [])
            system_prompt, user_input, turns = parse_messages(messages)
            details = d.get("reward_details", {}) or {}

            results.append(
                {
                    "index": idx,
                    "score": float(d.get("reward_score", 0)),
                    "sample_id": d.get("sample_id", ""),
                    "gt_smiles": d.get("gt_smiles", ""),
                    "n_rounds": d.get("n_rounds", 0),
                    "system_prompt": system_prompt,
                    "user_input": user_input,
                    "turns": turns,
                    "tool_call_count": details.get("tool_call_count"),
                    "reward_details": details,
                }
            )
    return tuple(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--traj-dir", default=str(DEFAULT_TRAJ_DIR))
    args = parser.parse_args()

    TRAJ_DIR = Path(args.traj_dir).resolve()
    print(f"Serving eval trajectories from: {TRAJ_DIR}")
    print(f"Open http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
