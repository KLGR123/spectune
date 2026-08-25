"""Shared sample loading and checkpointing for offline rollout consumers.

Both ``python -m spectune.rollout`` (SFT rollout / rejection sampling) and
``python -m spectune.rollout.eval`` (hit@k evaluation) need to read the same
two input shapes -- Spectune JSONL (from ``spectune.augmentor`` or the
"others" pool) and verl-compiled Parquet (from ``spectune.artifacts``) -- and
resume from the same ``<output>.checkpoint.jsonl`` convention. Centralizing
that here keeps both entry points thin and in sync.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spectune.jsonl import read_jsonl

JsonDict = dict[str, Any]


def to_python(value: Any) -> Any:
    """Convert numpy/pandas containers from a Parquet row into plain Python values."""
    if isinstance(value, Mapping):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [to_python(item) for item in value]
    if hasattr(value, "tolist"):
        return to_python(value.tolist())
    return value


def load_jsonl(path: str | Path) -> list[JsonDict]:
    """Load a Spectune augmentor/"others" JSONL file as rollout sample records.

    Each record gets a ``data_type`` field (one of ``INFORMATION_TYPES``),
    resolved the same way ``spectune.artifacts.compile`` does, so JSONL and
    Parquet inputs report identical per-type breakdowns.
    """
    from spectune.artifacts.compile import infer_data_type

    records = read_jsonl(path)
    for value in records:
        value.setdefault(
            "data_type",
            infer_data_type(str(value.get("sample_id", "")), value.get("augmentation")),
        )
    return records


def load_parquet(path: str | Path) -> list[JsonDict]:
    """Load a verl-compiled Parquet dataset as rollout sample records.

    Reads ``extra_info.turns`` / ``extra_info.data_type`` -- the
    consumer-agnostic fields written by ``spectune.artifacts.compile`` --
    rather than verl's ``prompt``/``raw_prompt`` columns, whose split
    (``raw_prompt`` intentionally excludes follow-up turns, injected instead
    via ``ScriptedFollowupInteraction`` at verl rollout time) is specific to
    the online agent loop.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("reading Parquet requires pandas/pyarrow; install with `pip install spectune[data]`") from exc

    frame = pd.read_parquet(Path(path).expanduser())
    records: list[JsonDict] = []
    for index, raw_row in frame.iterrows():
        row = {str(key): to_python(value) for key, value in raw_row.to_dict().items()}
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), Mapping) else {}
        turns = extra_info.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{path}: row {index} has no extra_info.turns; recompile with spectune.artifacts")

        reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), Mapping) else {}
        records.append(
            {
                "sample_id": str(extra_info.get("sample_id") or f"{Path(path).stem}:{index}"),
                "gt_smiles": reward_model.get("ground_truth", ""),
                "turns": [dict(t) for t in turns],
                "data_type": extra_info.get("data_type") or "none",
            }
        )
    return records


def load_samples(paths: str | Path | Sequence[str | Path], *, shuffle: bool = True) -> list[JsonDict]:
    """Load one or more JSONL/Parquet files, dispatching on file suffix.

    With more than one path, the combined samples are shuffled together so
    records from different files are interleaved.
    """
    path_list = [paths] if isinstance(paths, str | Path) else list(paths)
    if not path_list:
        raise ValueError("at least one input path is required")

    samples: list[JsonDict] = []
    for path in path_list:
        suffix = Path(path).suffix.lower()
        if suffix == ".jsonl":
            loaded = load_jsonl(path)
        elif suffix == ".parquet":
            loaded = load_parquet(path)
        else:
            raise ValueError(f"unsupported input format for {path}; expected .jsonl or .parquet")
        print(f"  {len(loaded)} samples loaded from {path}")
        samples.extend(loaded)
    if shuffle and len(path_list) > 1:
        random.shuffle(samples)
    return samples


def checkpoint_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.parent / (p.stem + ".checkpoint.jsonl")


def load_checkpoint(ckpt_path: Path) -> tuple[list, set[str]]:
    """Replay a ``<output>.checkpoint.jsonl`` written by a prior, interrupted run."""
    from spectune.rollout.rollout import RolloutRecord

    if not ckpt_path.exists():
        return [], set()
    records: list[RolloutRecord] = []
    ids: set[str] = set()
    with ckpt_path.open(encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                d = json.loads(text)
                record = RolloutRecord(
                    sample_id=d["sample_id"],
                    gt_smiles=d["gt_smiles"],
                    messages=d["messages"],
                    reward_score=d["reward_score"],
                    reward_details=d["reward_details"],
                    n_rounds=d["n_rounds"],
                    reasoning_content=d.get("reasoning_content"),
                )
                records.append(record)
                ids.add(record.sample_id)
            except (json.JSONDecodeError, KeyError):
                pass
    return records, ids


__all__ = [
    "checkpoint_path",
    "load_checkpoint",
    "load_jsonl",
    "load_parquet",
    "load_samples",
    "to_python",
]
