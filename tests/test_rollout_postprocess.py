import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from spectune.rollout.postprocess import build_parser, postprocess


def _write_dump(path: Path, *rows: dict) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _record(sample_id: str, reward_score: float) -> dict:
    return {
        "sample_id": sample_id,
        "gt_smiles": "CCO",
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        "reward_score": reward_score,
        "reward_details": {"ok": True},
        "n_rounds": 1,
    }


def test_postprocess_merges_multiple_input_paths(tmp_path: Path):
    first = _write_dump(tmp_path / "a.jsonl", _record("a", 0.8), _record("b", 0.0))
    second = _write_dump(tmp_path / "b.jsonl", _record("c", 0.9))
    output = tmp_path / "out.parquet"

    n = postprocess([first, second], output, min_reward=0.1, show_progress=False)

    assert n == 2
    written = pd.read_parquet(output)
    assert set(written["sample_id"]) == {"a", "c"}


def test_postprocess_accepts_a_single_path(tmp_path: Path):
    only = _write_dump(tmp_path / "only.jsonl", _record("a", 0.5))
    output = tmp_path / "out.parquet"

    n = postprocess(only, output, min_reward=0.1, show_progress=False)

    assert n == 1


def test_postprocess_rejects_empty_input_list(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one input path"):
        postprocess([], tmp_path / "out.parquet")


def test_cli_input_flag_accepts_multiple_paths():
    args = build_parser().parse_args(["--input", "a.jsonl", "b.jsonl", "--output", "out.parquet"])
    assert args.input == ["a.jsonl", "b.jsonl"]


def test_cli_input_flag_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--output", "out.parquet"])
