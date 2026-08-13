import argparse
import json
import re
from pathlib import Path

from flask import Flask, jsonify, render_template_string

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJ_DIR = ROOT / "outputs" / "trajectories"

app = Flask(__name__)
TRAJ_DIR: Path = DEFAULT_TRAJ_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_segments(raw: str) -> list[tuple[str, str]]:
    """Return [(tool_name, response_content), ...] for each tool call in output."""
    cleaned = re.sub(
        r"\buser\s*\n\s*(<tool_response>.*?</tool_response>)\s*\nassistant",
        r"\1",
        raw,
        flags=re.DOTALL,
    )
    calls = re.findall(r"<tool_call>(.*?)</tool_call>", cleaned, re.DOTALL)
    responses = re.findall(r"<tool_response>(.*?)</tool_response>", cleaned, re.DOTALL)
    result = []
    for i, call_str in enumerate(calls):
        try:
            name = json.loads(call_str.strip()).get("name", "unknown")
        except (json.JSONDecodeError, AttributeError):
            m = re.search(r'"name"\s*:\s*"([^"]+)"', call_str)
            name = m.group(1) if m else "unknown"
        response = responses[i].strip() if i < len(responses) else ""
        result.append((name, response))
    return result


def _is_failure(response: str) -> bool:
    """Heuristic: does the tool response indicate a failure?"""
    try:
        data = json.loads(response)
        if isinstance(data, dict) and ("error" in data or data.get("success") is False):
            return True
    except json.JSONDecodeError:
        pass
    lower = response.lower()
    return any(kw in lower for kw in ["error:", "exception:", "traceback", "failed:"])


def _nmr_rank(response: str, gts: str) -> int | None:
    """Return the rank of gts in nmr_generate candidates (data.candidates[].rank/smiles)."""
    if not gts:
        return None
    try:
        data = json.loads(response)
        candidates = data.get("data", {}).get("candidates", [])
        for cand in candidates:
            if cand.get("smiles") == gts or cand.get("canonical_smiles") == gts:
                return int(cand.get("rank", candidates.index(cand) + 1))
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>Tool Statistics – Spectune</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root { --bg:#f6f8fa; --panel:#fff; --border:#d0d7de; --gray:#57606a; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:#24292f; font-size:14px; }
  header { background:#24292f; color:#fff; padding:12px 24px;
           display:flex; align-items:center; gap:16px; }
  header h1 { font-size:16px; font-weight:600; }
  .controls { padding:10px 24px; background:var(--panel);
              border-bottom:1px solid var(--border);
              display:flex; align-items:center; gap:12px; }
  .controls label { font-size:13px; color:var(--gray);
                    display:flex; align-items:center; gap:8px; }
  .controls input[type=range] { width:160px; cursor:pointer; }
  .charts { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:16px; }
  .chart-card { background:var(--panel); border:1px solid var(--border);
                border-radius:8px; padding:16px 16px 12px; }
  .chart-card h3 { font-size:13px; font-weight:600; color:#24292f; margin-bottom:12px; }
  .span2 { grid-column: span 2; }
</style>
</head>
<body>
<header>
  <h1>Tool Statistics</h1>
  <span style="font-size:12px;color:#aaa;">{{ traj_dir }}</span>
</header>
<div class="controls">
  <label>Smoothing
    <input type="range" id="smooth" min="1" max="15" value="1"
           oninput="document.getElementById('sv').textContent=this.value; redraw()"/>
    <span id="sv">1</span>
  </label>
</div>
<div class="charts">
  <div class="chart-card"><h3>Tool Call Frequency</h3><canvas id="c1"></canvas></div>
  <div class="chart-card"><h3>Tool Reward Sum (per file)</h3><canvas id="c2"></canvas></div>
  <div class="chart-card"><h3>Tool Failure Rate</h3><canvas id="c3"></canvas></div>
  <div class="chart-card"><h3>nmr_generate Answer Rank (positive-score only)</h3><canvas id="c4"></canvas></div>
  <div class="chart-card"><h3>nmr_generate Tool Accuracy (all trajectories)</h3><canvas id="c5"></canvas></div>
  <div class="chart-card"><h3>Tool Calls per Turn (first query vs. follow-up)</h3><canvas id="c6"></canvas></div>
</div>
<script>
const PALETTE = [
  '#2563eb','#16a34a','#dc2626','#d97706','#7c3aed',
  '#0891b2','#db2777','#65a30d','#9333ea','#ea580c'
];

function smooth(arr, w) {
  if (w <= 1) return arr.slice();
  return arr.map((_, i) => {
    const slice = arr.slice(Math.max(0, i - w + 1), i + 1).filter(v => v != null);
    return slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : null;
  });
}

function mkDatasets(obj, labels, w) {
  return Object.entries(obj).map(([name, vals], i) => ({
    label: name,
    data: smooth(vals, w).map((v, j) => ({ x: labels[j], y: v })),
    borderColor: PALETTE[i % PALETTE.length],
    backgroundColor: PALETTE[i % PALETTE.length] + '22',
    borderWidth: 2, pointRadius: 2, tension: 0.3, spanGaps: true,
  }));
}

function mkChart(id, datasets, labels, yLabel) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { maxTicksLimit: 12, font: { size: 11 } } },
        y: {
          title: { display: !!yLabel, text: yLabel, font: { size: 11 } },
          ticks: { font: { size: 11 } }
        }
      }
    }
  });
}

let charts = [];
let rawData = null;

fetch('/api/stats').then(r => r.json()).then(data => { rawData = data; redraw(); });

function redraw() {
  if (!rawData) return;
  const w = parseInt(document.getElementById('smooth').value);
  const { labels, tool_freq, tool_reward, tool_fail, nmr_ranks, nmr_acc, turn_tool_counts } = rawData;
  charts.forEach(c => c.destroy());
  charts = [
    mkChart('c1', mkDatasets(tool_freq,   labels, w), labels, 'calls'),
    mkChart('c2', mkDatasets(tool_reward, labels, w), labels, 'reward sum'),
    mkChart('c3', mkDatasets(tool_fail,   labels, w), labels, 'failure rate'),
    mkChart('c4', [{
      label: 'avg rank',
      data: smooth(nmr_ranks, w).map((v, i) => ({ x: labels[i], y: v })),
      borderColor: PALETTE[0], backgroundColor: PALETTE[0] + '22',
      borderWidth: 2, pointRadius: 2, tension: 0.3, spanGaps: true,
    }], labels, 'rank'),
    mkChart('c5', mkDatasets(nmr_acc, labels, w), labels, 'accuracy'),
    mkChart('c6', mkDatasets(turn_tool_counts, labels, w), labels, 'avg tool calls'),
  ];
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML, traj_dir=str(TRAJ_DIR))


@app.route("/api/stats")
def api_stats():
    def sort_key(p: Path):
        return int(p.stem) if p.stem.isdigit() else p.stem

    files = sorted(TRAJ_DIR.rglob("*.jsonl"), key=sort_key)
    if not files:
        return jsonify({"labels": [], "tool_freq": {}, "tool_reward": {}, "tool_fail": {}, "nmr_ranks": []})

    labels: list[str] = []
    per_file: list[dict] = []

    for path in files:
        labels.append(path.stem)
        fc: dict[str, int] = {}
        fr: dict[str, float] = {}
        ff: dict[str, list[int]] = {}  # tool -> [fails, total]
        nr: list[int] = []
        r1: list[int] = []   # rank_1 acc (0/1) for every nmr_generate call
        ra: list[int] = []   # rank_all acc (0/1) for every nmr_generate call
        tc1: list[int] = []  # tool call count before first follow-up user turn
        tc2: list[int] = []  # tool call count after first follow-up user turn

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = float(d.get("score", 0))
                gts = d.get("gts", "")
                output = d.get("output", "")

                for tool_name, response in _parse_tool_segments(output):
                    fc[tool_name] = fc.get(tool_name, 0) + 1
                    fr[tool_name] = fr.get(tool_name, 0.0) + score
                    counts = ff.setdefault(tool_name, [0, 0])
                    counts[1] += 1
                    if _is_failure(response):
                        counts[0] += 1
                    if tool_name == "nmr_generate" and score > 0.5:
                        rank = _nmr_rank(response, gts)
                        if rank is not None:
                            nr.append(rank)
                    if tool_name == "nmr_generate" and gts:
                        try:
                            cands = json.loads(response).get("data", {}).get("candidates", [])
                            hit_all = int(any(
                                c.get("smiles") == gts or c.get("canonical_smiles") == gts
                                for c in cands
                            ))
                            hit_1 = int(bool(cands) and (
                                cands[0].get("smiles") == gts or cands[0].get("canonical_smiles") == gts
                            ))
                            ra.append(hit_all)
                            r1.append(hit_1)
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            pass

                # Per-turn tool call counts: strip tool_response user wrappers,
                # then split at the first remaining user turn (follow-up query).
                cleaned = re.sub(
                    r"\buser\s*\n\s*(<tool_response>.*?</tool_response>)\s*\nassistant",
                    r"\1", output, flags=re.DOTALL,
                )
                followup = re.search(r"\buser\s*\n", cleaned)
                if followup:
                    before_text = cleaned[:followup.start()]
                    after_text  = cleaned[followup.start():]
                    tc1.append(len(re.findall(r"<tool_call>", before_text)))
                    tc2.append(len(re.findall(r"<tool_call>", after_text)))
                else:
                    tc1.append(len(re.findall(r"<tool_call>", cleaned)))

        per_file.append({"fc": fc, "fr": fr, "ff": ff, "nr": nr, "r1": r1, "ra": ra, "tc1": tc1, "tc2": tc2})

    all_tools = sorted({t for d in per_file for t in d["fc"]})
    tool_freq   = {t: [d["fc"].get(t, 0)   for d in per_file] for t in all_tools}
    tool_reward = {t: [d["fr"].get(t, 0.0) for d in per_file] for t in all_tools}
    tool_fail   = {
        t: [
            d["ff"][t][0] / d["ff"][t][1] if t in d["ff"] and d["ff"][t][1] > 0 else None
            for d in per_file
        ]
        for t in all_tools
    }
    nmr_ranks = [
        sum(d["nr"]) / len(d["nr"]) if d["nr"] else None
        for d in per_file
    ]
    nmr_acc = {
        "rank_1":   [sum(d["r1"]) / len(d["r1"]) if d["r1"] else None for d in per_file],
        "rank_all": [sum(d["ra"]) / len(d["ra"]) if d["ra"] else None for d in per_file],
    }
    turn_tool_counts = {
        "first query":  [sum(d["tc1"]) / len(d["tc1"]) if d["tc1"] else None for d in per_file],
        "follow-up":    [sum(d["tc2"]) / len(d["tc2"]) if d["tc2"] else None for d in per_file],
    }

    return jsonify({
        "labels":           labels,
        "tool_freq":        tool_freq,
        "tool_reward":      tool_reward,
        "tool_fail":        tool_fail,
        "nmr_ranks":        nmr_ranks,
        "nmr_acc":          nmr_acc,
        "turn_tool_counts": turn_tool_counts,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--traj-dir", default=str(DEFAULT_TRAJ_DIR))
    args = parser.parse_args()

    TRAJ_DIR = Path(args.traj_dir).resolve()
    print(f"Serving tool statistics from: {TRAJ_DIR}")
    print(f"Open http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
