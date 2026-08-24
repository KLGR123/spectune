import argparse
import json
from pathlib import Path

import pytest

from spectune.rollout.cli import build_rollout_config
from spectune.rollout.io import checkpoint_path, load_checkpoint, load_jsonl, load_samples
from spectune.rollout.rollout import RolloutRecord


def test_load_jsonl_infers_data_type_from_augmentation(tmp_path: Path):
    path = tmp_path / "nmrexp.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "NMRexp:bulk:raw:1:aug",
                "gt_smiles": "CCO",
                "turns": [{"role": "user", "content": "q"}],
                "augmentation": {"information_type": "reaction"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_jsonl(path)

    assert samples[0]["data_type"] == "reaction"


def test_load_jsonl_infers_data_type_from_sample_id_prefix(tmp_path: Path):
    path = tmp_path / "others.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "formula:train:1",
                "gt_smiles": "CCO",
                "turns": [{"role": "user", "content": "q"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_jsonl(path)

    assert samples[0]["data_type"] == "formula"


def test_load_samples_dispatches_on_suffix_and_merges(tmp_path: Path):
    jsonl_path = tmp_path / "a.jsonl"
    jsonl_path.write_text(
        json.dumps({"sample_id": "a:1", "gt_smiles": "CCO", "turns": [{"role": "user", "content": "q"}]}) + "\n",
        encoding="utf-8",
    )
    other_jsonl_path = tmp_path / "b.jsonl"
    other_jsonl_path.write_text(
        json.dumps({"sample_id": "b:1", "gt_smiles": "CCN", "turns": [{"role": "user", "content": "q2"}]}) + "\n",
        encoding="utf-8",
    )

    samples = load_samples([jsonl_path, other_jsonl_path])

    assert {s["sample_id"] for s in samples} == {"a:1", "b:1"}


def test_load_samples_rejects_unsupported_suffix(tmp_path: Path):
    bad_path = tmp_path / "data.csv"
    bad_path.write_text("not jsonl", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported input format"):
        load_samples([bad_path])


def test_checkpoint_path_derives_sibling_file():
    path = checkpoint_path("outputs/evals/test.jsonl")
    assert path == Path("outputs/evals/test.checkpoint.jsonl")


def test_load_checkpoint_round_trips_rollout_records(tmp_path: Path):
    ckpt = tmp_path / "out.checkpoint.jsonl"
    record = RolloutRecord(
        sample_id="s1",
        gt_smiles="CCO",
        messages=[{"role": "user", "content": "hi"}],
        reward_score=0.9,
        reward_details={"gt_rank": 1},
        n_rounds=1,
    )
    ckpt.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")

    records, ids = load_checkpoint(ckpt)

    assert ids == {"s1"}
    assert records[0].gt_smiles == "CCO"


def test_load_checkpoint_skips_malformed_lines(tmp_path: Path):
    ckpt = tmp_path / "out.checkpoint.jsonl"
    ckpt.write_text("not json\n{\"sample_id\": \"only_partial\"}\n", encoding="utf-8")

    records, ids = load_checkpoint(ckpt)

    assert records == []
    assert ids == set()


def test_build_rollout_config_reads_shared_cli_args():
    args = argparse.Namespace(
        rounds=3,
        temperature=0.5,
        top_p=0.9,
        max_tokens=2048,
        nmr_gen_topk=15,
        tools=["nmr_generate", "web_search"],
        max_assistant_turns=8,
        max_concurrency=4,
        no_progress=True,
        skills=None,
    )

    config = build_rollout_config(args)

    assert config.max_rounds == 3
    assert config.temperature == 0.5
    assert config.nmr_gen_topk == 15
    assert config.tool_names == ("nmr_generate", "web_search")
    assert config.max_assistant_turns == 8
    assert config.show_progress is False
