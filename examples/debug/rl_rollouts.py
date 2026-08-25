import argparse
import json
import re
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJ_DIR = ROOT / "outputs" / "trajectories" / "rl"

app = Flask(__name__)
TRAJ_DIR: Path = DEFAULT_TRAJ_DIR


def _extract_role_content(raw: str, role: str) -> str:
    """Return content following the first standalone ``role`` marker."""
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == role:
            parts = []
            for j in range(i + 1, len(lines)):
                if lines[j].strip() in ("assistant", "system", "user"):
                    break
                parts.append(lines[j])
            return "\n".join(parts).strip()
    return ""


def extract_input_sections(raw: str) -> tuple[str, str]:
    """Return the system prefix and first user turn from a serialized prompt."""
    system_prompt = _extract_role_content(raw, "system")
    user_input = _extract_role_content(raw, "user")
    if not system_prompt and not user_input:
        user_input = raw.strip()
    return system_prompt, user_input


def extract_user_input(raw: str) -> str:
    """Return the first user turn, preserving the historical helper API."""
    return extract_input_sections(raw)[1]


def parse_output_segments(raw: str) -> list[dict]:
    """
    Split model output into alternating segments:
      - {"type": "text",          "content": "..."}
      - {"type": "think",         "content": "..."}  (<think> block, tags stripped)
      - {"type": "tool_call",     "content": "..."}
      - {"type": "tool_response", "content": "..."}
      - {"type": "user_turn",     "content": "..."}  (mid-conversation user message)
    """
    segments: list[dict] = []

    # Strip the "user\n…\nassistant" role wrappers around tool_response
    cleaned = re.sub(
        r"\buser\s*\n\s*(<tool_response>.*?</tool_response>)\s*\nassistant",
        r"\1",
        raw,
        flags=re.DOTALL,
    )

    # Convert remaining mid-conversation user turns into a custom tag, skip empty ones
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


HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>RL Rollouts Viewer</title>
<style>
  :root {
    --bg: #f6f8fa; --panel: #fff; --border: #d0d7de;
    --blue: #0969da; --green: #1a7f37; --orange: #953800;
    --purple: #8250df; --gray: #57606a;
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
  #sidebar { width: 260px; min-width: 200px; border-right: 1px solid var(--border);
             overflow-y: auto; background: var(--panel); padding: 8px 0; flex-shrink: 0; }
  .group-toggle { padding: 7px 12px; font-size: 12px; font-weight: 600;
                  color: #24292f; cursor: pointer; display: flex; align-items: center;
                  gap: 6px; user-select: none; border-bottom: 1px solid var(--border); }
  .group-toggle:hover { background: #f0f6ff; }
  .group-toggle .arrow { font-size: 10px; color: var(--gray); display: inline-block;
                         transition: transform .15s; }
  .group-toggle.open .arrow { transform: rotate(90deg); }
  .group-children { display: none; background: #fafbfc; }
  .group-children.open { display: block; }
  .file-link { display: block; padding: 5px 12px 5px 26px; cursor: pointer;
               color: var(--blue); font-size: 13px; }
  .file-link:hover { background: #f0f6ff; }
  .file-link.active { background: #dbeafe; font-weight: 600; }

  /* main */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #controls { padding: 10px 16px; border-bottom: 1px solid var(--border);
              background: var(--panel); display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  #controls label { font-size: 13px; color: var(--gray); }
  #controls select, #controls input { font-size: 13px; border: 1px solid var(--border);
                                      border-radius: 6px; padding: 4px 8px; background: #fff; }
  #traj-counter { font-size: 13px; color: var(--gray); margin-left: auto; }
  #viewer { flex: 1; overflow-y: auto; padding: 16px; }

  /* trajectory card */
  .traj-card { background: var(--panel); border: 1px solid var(--border);
               border-radius: 8px; margin-bottom: 24px; overflow: hidden; }
  .traj-header { padding: 10px 14px; background: #f0f6ff;
                 border-bottom: 1px solid var(--border);
                 display: flex; gap: 16px; align-items: center; font-size: 13px; }
  .traj-header strong { font-size: 14px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 12px; font-weight: 600; }
  .badge-score { background: #dafbe1; color: var(--green); }
  .badge-score.low { background: #fff0b3; color: #633c01; }
  .badge-score.bad { background: #ffe0e0; color: #cf222e; }
  .badge-step  { background: #eaeef2; color: var(--gray); }
  .meta-row { padding: 8px 14px; border-bottom: 1px solid var(--border);
              font-size: 12px; color: var(--gray); }
  .meta-row span { margin-right: 16px; }
  .meta-row .gts { color: #24292f; font-weight: 500; }

  /* sections */
  .section { border-top: 1px solid var(--border); }
  .section-title { padding: 7px 14px; font-size: 12px; font-weight: 600;
                   text-transform: uppercase; letter-spacing: .05em; cursor: pointer;
                   display: flex; align-items: center; gap: 6px; user-select: none; }
  .section-title::before { content: "▾"; transition: transform .15s; }
  .section-title.collapsed::before { transform: rotate(-90deg); }
  .section-content { padding: 12px 14px; font-size: 13px; }
  .section-content.hidden { display: none; }

  /* input */
  .system-section .section-title { background: #f0f0f0; color: var(--gray); }
  .input-section .section-title { background: #f0fff4; color: var(--green); }
  .input-text { white-space: pre-wrap; line-height: 1.6; }

  /* output segments */
  .output-section .section-title { background: #fff8f0; color: var(--orange); }
  .seg { margin-bottom: 10px; border-radius: 6px; overflow: hidden;
         border: 1px solid var(--border); }
  .seg-label { padding: 3px 10px; font-size: 11px; font-weight: 700;
               letter-spacing: .06em; text-transform: uppercase; }
  .seg-text   .seg-label { background: #f6f8fa; color: var(--gray); }
  .seg-tool   .seg-label { background: #fff0f0; color: #cf222e; }
  .seg-result .seg-label { background: #f0f6ff; color: var(--blue); }
  .seg-think  .seg-label { background: #f5f0ff; color: var(--purple); }
  .seg-think  .seg-body  { background: #faf7ff; }
  .seg-user   .seg-label { background: #f0fff4; color: var(--green); }
  .seg-user   .seg-body  { background: #f6fffb; font-family: inherit; font-size: 13px; }
  .seg-body { padding: 8px 10px; white-space: pre-wrap; font-family: "SFMono-Regular", Consolas, monospace;
              font-size: 12px; line-height: 1.55; background: var(--code-bg); overflow-x: auto; }
  .seg-text .seg-body { font-family: inherit; font-size: 13px; background: #fff; }
</style>
</head>
<body>
<header>
  <h1>RL Rollouts Viewer</h1>
  <span style="font-size:12px;color:#aaa;">{{ traj_dir }}</span>
</header>
<div class="layout">
  <div id="sidebar">
    {% for group, files in tree.items() %}
    <div>
      <div class="group-toggle" onclick="toggleGroup(this)">
        <span class="arrow">▶</span>{{ group }}
      </div>
      <div class="group-children">
        {% for f in files %}
        <a class="file-link" onclick="loadFile('{{ f.rel }}', this)">{{ f.name }}</a>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </div>
  <div id="main">
    <div id="controls">
      <label>Filter score ≥</label>
      <input type="number" id="min-score" value="-99" step="0.05" style="width:70px"
             oninput="renderVisible()"/>
      <label>Sort by</label>
      <select id="sort-by" onchange="renderVisible()">
        <option value="index">Index</option>
        <option value="score_desc">Score ↓</option>
        <option value="score_asc">Score ↑</option>
      </select>
      <span id="traj-counter"></span>
    </div>
    <div id="viewer"><p style="color:var(--gray);padding:24px">← Select a file from the sidebar</p></div>
  </div>
</div>

<script>
let allTrajs = [];

function loadFile(rel, el) {
  document.querySelectorAll('.file-link').forEach(a => a.classList.remove('active'));
  el.classList.add('active');
  fetch('/api/file?rel=' + encodeURIComponent(rel))
    .then(r => r.json())
    .then(data => { allTrajs = data; renderVisible(); });
}

function renderVisible() {
  const minScore = parseFloat(document.getElementById('min-score').value) || -99;
  const sortBy   = document.getElementById('sort-by').value;

  let trajs = allTrajs.filter(t => t.score >= minScore);

  if (sortBy === 'score_desc') trajs.sort((a, b) => b.score - a.score);
  else if (sortBy === 'score_asc') trajs.sort((a, b) => a.score - b.score);

  document.getElementById('traj-counter').textContent =
    `Showing ${trajs.length} / ${allTrajs.length}`;

  document.getElementById('viewer').innerHTML = trajs.map(renderTraj).join('');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function scoreBadgeClass(s) {
  if (s >= 0.5) return 'badge-score';
  if (s >= 0)   return 'badge-score low';
  return 'badge-score bad';
}

function renderTraj(t) {
  const systemHtml = t.system_prompt ? `
    <div class="section system-section">
      <div class="section-title collapsed" onclick="toggleSection(this)">System</div>
      <div class="section-content hidden">
        <div class="input-text">${escHtml(t.system_prompt)}</div>
      </div>
    </div>` : '';
  const segsHtml = t.segments.map(seg => {
    if (seg.type === 'text') {
      return `<div class="seg seg-text">
        <div class="seg-label">Text</div>
        <div class="seg-body">${escHtml(seg.content)}</div>
      </div>`;
    } else if (seg.type === 'think') {
      return `<div class="seg seg-think">
        <div class="seg-label">Think</div>
        <div class="seg-body">${escHtml(seg.content)}</div>
      </div>`;
    } else if (seg.type === 'user_turn') {
      return `<div class="seg seg-user">
        <div class="seg-label">User</div>
        <div class="seg-body">${escHtml(seg.content)}</div>
      </div>`;
    } else if (seg.type === 'tool_call') {
      return `<div class="seg seg-tool">
        <div class="seg-label">Tool Call</div>
        <div class="seg-body">${escHtml(seg.content)}</div>
      </div>`;
    } else {
      return `<div class="seg seg-result">
        <div class="seg-label">Tool Response</div>
        <div class="seg-body">${escHtml(seg.content)}</div>
      </div>`;
    }
  }).join('');

  return `
  <div class="traj-card">
    <div class="traj-header">
      <strong>#${t.index + 1}</strong>
      <span class="badge ${scoreBadgeClass(t.score)}">score ${t.score.toFixed(3)}</span>
      <span class="badge badge-step">step ${t.step}</span>
    </div>
    <div class="meta-row">
      <span>GT: <span class="gts">${escHtml(t.gts)}</span></span>
      <span>tool_calls: ${t.tool_call_count ?? '?'}</span>
      <span>validity: ${t.smiles_validity ?? '?'}</span>
    </div>
    ${systemHtml}
    <div class="section input-section">
      <div class="section-title" onclick="toggleSection(this)">Input</div>
      <div class="section-content">
        <div class="input-text">${escHtml(t.user_input)}</div>
      </div>
    </div>
    <div class="section output-section">
      <div class="section-title" onclick="toggleSection(this)">Output</div>
      <div class="section-content">${segsHtml}</div>
    </div>
  </div>`;
}

function toggleGroup(el) {
  el.classList.toggle('open');
  el.nextElementSibling.classList.toggle('open');
}

function toggleSection(titleEl) {
  titleEl.classList.toggle('collapsed');
  titleEl.nextElementSibling.classList.toggle('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
  const first = document.querySelector('.group-toggle');
  if (first) toggleGroup(first);
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    tree: dict[str, list[dict]] = {}
    for path in sorted(TRAJ_DIR.rglob("*.jsonl")):
        group = path.parent.name
        rel = str(path.relative_to(TRAJ_DIR))
        tree.setdefault(group, []).append({"rel": rel, "name": path.name})
    return render_template_string(HTML, tree=tree, traj_dir=str(TRAJ_DIR))


@app.route("/api/file")
def api_file():
    rel = request.args.get("rel", "")
    # security: prevent path traversal
    target = (TRAJ_DIR / rel).resolve()
    if not str(target).startswith(str(TRAJ_DIR.resolve())):
        abort(400)
    if not target.exists():
        abort(404)

    results = []
    with open(target, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            system_prompt, user_input = extract_input_sections(d.get("input", ""))
            segments = parse_output_segments(d.get("output", ""))

            results.append(
                {
                    "index": idx,
                    "score": float(d.get("score", 0)),
                    "step": d.get("step"),
                    "gts": d.get("gts", ""),
                    "tool_call_count": d.get("tool_call_count"),
                    "smiles_validity": d.get("smiles_validity"),
                    "system_prompt": system_prompt,
                    "user_input": user_input,
                    "segments": segments,
                }
            )
    return jsonify(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--traj-dir", default=str(DEFAULT_TRAJ_DIR))
    args = parser.parse_args()

    TRAJ_DIR = Path(args.traj_dir).resolve()
    print(f"Serving trajectories from: {TRAJ_DIR}")
    print(f"Open http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
