"""Entry point for ``python -m spectune.rollout``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cli import add_llm_args, build_rollout_config, run_with_backend
from .config import DEFAULT_OUTPUT_DIR
from .io import checkpoint_path, load_checkpoint, load_samples
from .rollout import RolloutRecord, write_jsonl, write_parquet
from .sampling.rejection import GtRejectionSampler


def _default_output(input_path: str, fmt: str) -> str:
    stem = Path(input_path).stem  # e.g. nmrexp_sft_2000
    name = stem.replace("_sft_", "_sft_rollout_")
    return str(Path(DEFAULT_OUTPUT_DIR) / f"{name}.{fmt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectune.rollout",
        description=(
            "Offline parallel LLM rollout with tool execution and rejection sampling.\n"
            "Reads JSONL produced by `python -m spectune.augmentor` or a verl "
            "dataset Parquet, runs a hermes tool-agent loop per "
            "sample (tool calls are executed for real and results fed back), "
            "and keeps only trajectories where the model correctly identifies "
            "the ground-truth SMILES (GT rejection sampling).  "
            "Output is a JSONL / Parquet file with a `messages` column "
            "suitable for verl SFT training."
        ),
    )

    p.add_argument(
        "--input",
        required=True,
        nargs="+",
        metavar="PATH",
        help=(
            "Input JSONL or verl Parquet file(s). "
            "When multiple files are given, their samples are combined and shuffled together."
        ),
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(f"Destination file (JSONL or Parquet).  Defaults to {DEFAULT_OUTPUT_DIR}/<stem_rollout>.<fmt>."),
    )
    p.add_argument(
        "--format",
        choices=["jsonl", "parquet"],
        default="jsonl",
        dest="fmt",
        help="Output format (default: jsonl).",
    )
    add_llm_args(p)
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help=(
            "Disable incremental checkpoint saves.  "
            "By default each accepted record is flushed to <output>.checkpoint.jsonl "
            "as it completes; on restart the checkpoint is replayed and already-done "
            "samples are skipped."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = args.output or _default_output(args.input[0], args.fmt)
    config = build_rollout_config(args)

    print(f"loading  {', '.join(args.input)}")
    samples = load_samples(args.input)
    print(f"  {len(samples)} samples loaded total")
    total_samples = len(samples)

    use_checkpoint = not args.no_checkpoint
    ckpt_path = checkpoint_path(output_path)
    prior_records: list[RolloutRecord] = []

    if use_checkpoint:
        prior_records, completed_ids = load_checkpoint(ckpt_path)
        if prior_records:
            print(f"checkpoint  {len(prior_records)} already-done records found in {ckpt_path}")
        samples = [s for s in samples if str(s.get("sample_id", "")) not in completed_ids]
        if len(prior_records):
            print(f"  {len(samples)} remaining samples to process")

    sampler = GtRejectionSampler() if args.rounds is not None else None
    records = run_with_backend(
        args,
        config,
        samples,
        sampler=sampler,
        checkpoint_path=ckpt_path if use_checkpoint else None,
    )
    records = prior_records + records

    accepted = len(records)
    accepted_pct = 100 * accepted / total_samples if total_samples else 0.0
    print(f"accepted {accepted}/{total_samples} samples ({accepted_pct:.1f}%)")

    if args.fmt == "parquet":
        write_parquet(records, output_path)
    else:
        write_jsonl(records, output_path)

    print(f"wrote    {output_path}")

    if use_checkpoint and ckpt_path.exists():
        ckpt_path.unlink()
        print(f"removed  {ckpt_path}")


if __name__ == "__main__":
    main()
