"""SFT dataset statistics viewer: score / length distributions and tool call sequence tree."""

import argparse
import json
import re
from pathlib import Path

from flask import Flask, jsonify, render_template_string

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = ROOT / "outputs" / "datasets" / "verl" / "train_sft.parquet"

app = Flask(__name__)
PARQUET_PATH: Path = DEFAULT_PARQUET
_cache: dict | None = None


def _extract_tool_sequence(messages: list) -> list[str]:
    seq = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        if role != "assistant":
            continue
        for call_str in re.findall(r"<tool_call>(.*?)</tool_call>", str(content), re.DOTALL):
            try:
                name = json.loads(call_str.strip()).get("name", "unknown")
            except (json.JSONDecodeError, AttributeError):
                m = re.search(r'"name"\s*:\s*"([^"]+)"', call_str)
                name = m.group(1) if m else "unknown"
            seq.append(name)
    return seq


def _build_trie(sequences: list[list[str]]) -> dict:
    root: dict = {"name": "start", "count": len(sequences), "end_count": 0, "children": {}}
    for seq in sequences:
        node = root
        for tool in seq:
            if tool not in node["children"]:
                node["children"][tool] = {"name": tool, "count": 0, "end_count": 0, "children": {}}
            node["children"][tool]["count"] += 1
            node = node["children"][tool]
        node["end_count"] += 1
    return root


def _trie_to_list(node: dict) -> dict:
    children = sorted(node["children"].values(), key=lambda n: -n["count"])
    return {
        "name": node["name"],
        "count": node["count"],
        "end_count": node["end_count"],
        "children": [_trie_to_list(c) for c in children],
    }


def _make_score_hist(scores: list[float], n_bins: int = 20) -> dict:
    if not scores:
        return {"labels": [], "counts": []}
    step = 1.0 / n_bins
    counts = [0] * n_bins
    labels = [f"{i * step:.2f}" for i in range(n_bins)]
    for s in scores:
        idx = min(max(int(s * n_bins), 0), n_bins - 1)
        counts[idx] += 1
    return {"labels": labels, "counts": counts}


def _make_len_hist(lengths: list[int]) -> dict:
    if not lengths:
        return {"labels": [], "counts": []}
    lo, hi = min(lengths), max(lengths)
    labels = [str(v) for v in range(lo, hi + 1)]
    counts = [0] * (hi - lo + 1)
    for v in lengths:
        counts[v - lo] += 1
    return {"labels": labels, "counts": counts}


def load_stats() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    import pandas as pd

    df = pd.read_parquet(PARQUET_PATH)
    scores: list[float] = []
    lengths: list[int] = []
    tool_counts: list[int] = []
    seqs: list[list[str]] = []

    for _, row in df.iterrows():
        scores.append(float(row.get("reward_score", 0.0)))
        messages = row.get("messages")
        if messages is None:
            messages = []
        else:
            messages = list(messages)
        lengths.append(len(messages))
        seq = _extract_tool_sequence(messages)
        tool_counts.append(len(seq))
        seqs.append(seq)

    n = len(df)
    _cache = {
        "total": n,
        "avg_score": sum(scores) / n if n else 0.0,
        "avg_messages": sum(lengths) / n if n else 0.0,
        "avg_tool_calls": sum(tool_counts) / n if n else 0.0,
        "score_hist": _make_score_hist(scores),
        "length_hist": _make_len_hist(lengths),
        "tool_tree": _trie_to_list(_build_trie(seqs)),
    }
    return _cache


HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>SFT Dataset Statistics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f6f8fa; --panel: #fff; --border: #d0d7de;
    --blue: #0969da; --green: #1a7f37; --gray: #57606a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: #24292f; font-size: 14px; }

  header { background: #24292f; color: #fff; padding: 12px 24px;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; font-weight: 600; }
  header .sub { font-size: 12px; color: #aaa; }

  .stat-bar { background: var(--panel); border-bottom: 1px solid var(--border);
              padding: 12px 24px; display: flex; gap: 40px; flex-wrap: wrap; }
  .stat-item { display: flex; flex-direction: column; gap: 2px; }
  .stat-label { font-size: 11px; color: var(--gray); text-transform: uppercase; letter-spacing: .05em; }
  .stat-value { font-size: 20px; font-weight: 700; color: #24292f; }

  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
  .chart-card { background: var(--panel); border: 1px solid var(--border);
                border-radius: 8px; padding: 16px 16px 12px; }
  .chart-card h3 { font-size: 13px; font-weight: 600; margin-bottom: 12px; color: #24292f; }

  .tree-card { background: var(--panel); border: 1px solid var(--border);
               border-radius: 8px; margin: 0 16px 16px; padding: 16px; }
  .tree-card h3 { font-size: 13px; font-weight: 600; margin-bottom: 14px; color: #24292f; }

  #tree-root { font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; line-height: 1; }
  .tree-row { display: flex; align-items: center; padding: 2px 0; }
  .tree-toggle { cursor: pointer; font-size: 10px; color: var(--gray); user-select: none;
                 width: 14px; text-align: center; flex-shrink: 0; }
  .tree-toggle:hover { color: var(--blue); }
  .tree-dot { font-size: 7px; color: #aaa; width: 14px; text-align: center; flex-shrink: 0; }
  .tree-name { color: #24292f; }
  .tree-name.root { font-weight: 700; color: var(--blue); }
  .tree-name.end-label { color: var(--gray); font-style: italic; }
  .tree-cnt { margin-left: 6px; font-size: 11px; padding: 1px 6px; border-radius: 10px;
              background: #eaeef2; color: var(--gray); font-weight: 600; flex-shrink: 0; }
  .tree-cnt.end-cnt { background: #dafbe1; color: var(--green); }

  #loading { padding: 48px 24px; color: var(--gray); font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>SFT Dataset Statistics</h1>
  <span class="sub">{{ parquet_path }}</span>
</header>

<div id="loading">Loading…</div>
<div id="content" style="display:none">
  <div class="stat-bar" id="stat-bar"></div>
  <div class="charts">
    <div class="chart-card">
      <h3>Trajectory Length (messages)</h3>
      <canvas id="c-len"></canvas>
    </div>
    <div class="chart-card">
      <h3>Score Distribution</h3>
      <canvas id="c-score"></canvas>
    </div>
  </div>
  <div class="tree-card">
    <h3>Tool Call Sequence Tree</h3>
    <div id="tree-root"></div>
  </div>
</div>

<script>
let _uid = 0;
function uid() { return 'tn' + (++_uid); }

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function mkChart(id, labels, counts, color) {
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: color + 'b0',
        borderColor: color,
        borderWidth: 1,
        borderRadius: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxTicksLimit: 14, font: { size: 11 } } },
        y: { beginAtZero: true, ticks: { font: { size: 11 } } },
      },
    },
  });
}

function renderTree(node, depth) {
  const hasChildren = node.children && node.children.length > 0;
  const childId = uid();
  const indentPx = depth * 18;

  let html = '<div class="tree-row">';
  html += `<span style="display:inline-block;width:${indentPx}px;flex-shrink:0"></span>`;
  if (hasChildren) {
    html += `<span class="tree-toggle" onclick="toggleNode(this,'${childId}')">▾</span>`;
  } else {
    html += '<span class="tree-dot">●</span>';
  }
  const nameClass = depth === 0 ? 'tree-name root' : 'tree-name';
  html += `<span class="${nameClass}">${escHtml(node.name)}</span>`;
  html += `<span class="tree-cnt">${node.count}</span>`;
  html += '</div>';

  if (hasChildren) {
    html += `<div id="${childId}">`;
    if (node.end_count > 0) {
      html += '<div class="tree-row">';
      html += `<span style="display:inline-block;width:${indentPx + 18}px;flex-shrink:0"></span>`;
      html += '<span class="tree-dot" style="color:var(--green)">●</span>';
      html += '<span class="tree-name end-label">[end]</span>';
      html += `<span class="tree-cnt end-cnt">${node.end_count}</span>`;
      html += '</div>';
    }
    for (const child of node.children) {
      html += renderTree(child, depth + 1);
    }
    html += '</div>';
  }
  return html;
}

function toggleNode(el, id) {
  const div = document.getElementById(id);
  const hidden = div.style.display === 'none';
  div.style.display = hidden ? '' : 'none';
  el.textContent = hidden ? '▾' : '▸';
}

fetch('/api/stats').then(r => r.json()).then(d => {
  if (d.error) {
    document.getElementById('loading').textContent = 'Error: ' + d.error;
    return;
  }
  document.getElementById('loading').style.display = 'none';
  document.getElementById('content').style.display = 'block';

  document.getElementById('stat-bar').innerHTML = [
    ['Records',        d.total],
    ['Avg Score',      d.avg_score.toFixed(4)],
    ['Avg Messages',   d.avg_messages.toFixed(1)],
    ['Avg Tool Calls', d.avg_tool_calls.toFixed(1)],
  ].map(([lbl, val]) =>
    `<div class="stat-item">
      <span class="stat-label">${lbl}</span>
      <span class="stat-value">${val}</span>
    </div>`
  ).join('');

  mkChart('c-len',   d.length_hist.labels, d.length_hist.counts, '#2563eb');
  mkChart('c-score', d.score_hist.labels,  d.score_hist.counts,  '#16a34a');

  document.getElementById('tree-root').innerHTML = renderTree(d.tool_tree, 0);
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML, parquet_path=str(PARQUET_PATH))


@app.route("/api/stats")
def api_stats():
    try:
        return jsonify(load_stats())
    except FileNotFoundError:
        return jsonify({"error": f"file not found: {PARQUET_PATH}"}), 404


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python examples/debug/sft_stats.py")
    parser.add_argument("--port",    type=int, default=7863)
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET),
                        help="Path to train_sft.parquet (default: outputs/datasets/verl/train_sft.parquet)")
    args = parser.parse_args()

    PARQUET_PATH = Path(args.parquet).resolve()
    print(f"Serving SFT stats from: {PARQUET_PATH}")
    print(f"Open http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
