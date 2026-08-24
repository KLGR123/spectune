"""Post-process rollout JSONL dumps into a verl-ready SFT Parquet file.

Reads one or more rollout ``.jsonl`` files, keeps records where
``reward_score > --min-reward``, optionally drops conversations that exceed a
token limit, shuffles the survivors, and writes a single ``train_sft.parquet``
suitable for verl SFT training.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from spectune.jsonl import read_jsonl

DEFAULT_SFT_OUTPUT_DIR = "outputs/datasets/verl"
DEFAULT_SFT_OUTPUT_FILE = "train_sft.parquet"
DEFAULT_MIN_REWARD = 0.1
DEFAULT_MAX_LENGTH = 32768


def _drop_overlong_records(
    records: list[dict],
    *,
    tokenizer_name_or_path: str | Path,
    max_length: int,
    show_progress: bool,
) -> list[dict]:
    """Keep records whose chat-templated ``messages`` fit ``max_length`` tokens."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "--drop-overlong requires transformers; install it with `pip install transformers`"
        ) from exc

    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name_or_path))
    kept: list[dict] = []
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"record {index} has no valid messages list")
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized: dict[str, Any] = tokenizer(text)
        if len(tokenized["input_ids"]) <= max_length:
            kept.append(record)

    if show_progress:
        print(f"  length   {len(kept):>6}/{len(records)} records kept (chat tokens <= {max_length})")
    return kept


def postprocess(
    input_path: str | Path | Iterable[str | Path],
    output_path: str | Path,
    *,
    min_reward: float = DEFAULT_MIN_REWARD,
    seed: int = 42,
    show_progress: bool = True,
    drop_overlong: bool = False,
    tokenizer_name_or_path: str | Path | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> int:
    """Filter, shuffle and write rollout JSONL records to a Parquet file.

    ``input_path`` accepts a single path or an iterable of paths (e.g. every
    ``*.jsonl`` dump under ``outputs/datasets/sft``, listed explicitly).

    Returns the number of rows written.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "postprocess requires pandas/pyarrow; install with `pip install spectune[data]`"
        ) from exc

    jsonl_paths = [input_path] if isinstance(input_path, str | Path) else list(input_path)
    if not jsonl_paths:
        raise ValueError("at least one input path is required")

    all_records: list[dict] = []
    total_loaded = 0
    for path in jsonl_paths:
        loaded = read_jsonl(path)
        kept = [r for r in loaded if r.get("reward_score", 0.0) > min_reward]
        if show_progress:
            print(f"  loaded   {len(loaded):>6}  kept {len(kept):>6}  ({path})")
        all_records.extend(kept)
        total_loaded += len(loaded)

    filtered = all_records
    if show_progress:
        print(f"  total    {len(filtered):>6}/{total_loaded} records kept (reward_score > {min_reward})")

    if drop_overlong:
        if tokenizer_name_or_path is None:
            raise ValueError("tokenizer_name_or_path is required when drop_overlong=True")
        filtered = _drop_overlong_records(
            filtered,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_length=max_length,
            show_progress=show_progress,
        )

    rng = random.Random(seed)
    rng.shuffle(filtered)

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filtered).to_parquet(output_path, index=False)
    return len(filtered)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectune.rollout.postprocess",
        description=(
            "Filter and merge rollout JSONL dumps into a verl-ready SFT Parquet file. "
            "Reads the given *.jsonl files, keeps records where "
            "reward_score > --min-reward, shuffles the survivors, and writes "
            "a single train_sft.parquet."
        ),
    )
    p.add_argument(
        "--input",
        required=True,
        nargs="+",
        metavar="PATH",
        help=(
            "One or more rollout JSONL dump paths, e.g. outputs/datasets/sft/*.jsonl "
            "(shell-expanded). All matching records are merged before filtering."
        ),
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=str(Path(DEFAULT_SFT_OUTPUT_DIR) / DEFAULT_SFT_OUTPUT_FILE),
        help=(
            f"Destination Parquet file "
            f"(default: {DEFAULT_SFT_OUTPUT_DIR}/{DEFAULT_SFT_OUTPUT_FILE})."
        ),
    )
    p.add_argument(
        "--min-reward",
        type=float,
        default=DEFAULT_MIN_REWARD,
        metavar="F",
        help=f"Minimum reward_score to keep a record (default: {DEFAULT_MIN_REWARD}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )
    p.add_argument(
        "--drop-overlong",
        action="store_true",
        help="Drop records whose chat-templated messages exceed --max-length tokens.",
    )
    p.add_argument(
        "--tokenizer",
        metavar="MODEL_OR_PATH",
        help="Tokenizer name or local path; required with --drop-overlong.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        metavar="N",
        help=(
            "Maximum chat-template token length when dropping overlong records "
            f"(default: {DEFAULT_MAX_LENGTH})."
        ),
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-file progress output.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.drop_overlong and not args.tokenizer:
        parser.error("--tokenizer is required with --drop-overlong")
    print(f"loading  {', '.join(args.input)}")
    n = postprocess(
        args.input,
        args.output,
        min_reward=args.min_reward,
        seed=args.seed,
        show_progress=not args.no_progress,
        drop_overlong=args.drop_overlong,
        tokenizer_name_or_path=args.tokenizer,
        max_length=args.max_length,
    )
    print(f"wrote    {n} rows → {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["postprocess", "build_parser", "main"]
