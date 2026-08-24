"""Entry point for ``python -m spectune.rollout.eval``.

Evaluates a model against a compiled verl Parquet (or Spectune JSONL) test
set and reports hit@k metrics with a breakdown by ``data_type`` (the
canonical scenario label written by ``spectune.artifacts.compile`` /
``spectune.rollout.io.load_jsonl``).

This is a thin wrapper around the same rollout machinery as
``python -m spectune.rollout``: it shares CLI flags and backend dispatch via
:mod:`spectune.rollout.cli`, and sample/checkpoint loading via
:mod:`spectune.rollout.io`. The only eval-specific step is grouping the
accepted records by ``data_type`` before printing hit@k.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .cli import add_llm_args, build_rollout_config, run_with_backend
from .io import checkpoint_path, load_checkpoint, load_samples
from .rollout import RolloutRecord, compute_hit_at_k_metrics, write_jsonl, write_parquet

JsonDict = dict


def _default_output(input_path: str, fmt: str) -> str:
    stem = Path(input_path).stem  # e.g. test
    return str(Path("outputs") / "evals" / f"{stem}.{fmt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectune.rollout.eval",
        description=(
            "Evaluate a model on a compiled verl Parquet / Spectune JSONL test set.\n"
            "Runs the same offline tool-agent rollout as `python -m spectune.rollout` "
            "and reports hit@k (rank of the ground-truth SMILES among returned "
            "candidates), overall and broken down by data_type."
        ),
    )
    p.add_argument(
        "--input",
        required=True,
        nargs="+",
        metavar="PATH",
        help=(
            "Input verl Parquet or Spectune JSONL file(s). "
            "When multiple files are given, their samples are combined and shuffled together."
        ),
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Destination file (JSONL or Parquet).  Defaults to outputs/evals/<stem>.<fmt>.",
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
            "By default each completed record is flushed to <output>.checkpoint.jsonl "
            "as it completes; on restart the checkpoint is replayed and already-done "
            "samples are skipped."
        ),
    )
    p.add_argument(
        "--no-hit-at-k",
        action="store_true",
        help="Skip printing hit@k metrics (only write the output file).",
    )
    return p


def _print_metrics(metrics: JsonDict, label: str) -> None:
    n = int(metrics["num_samples"])
    nc = int(metrics["num_completed"])
    print(f"\n{label}  (samples={n}, completed={nc})")
    for k in range(1, int(metrics["max_k"]) + 1):
        print(f"  hit@{k:<3} {metrics[f'hit@{k}']:.4f}")
    print(f"  hit@all {metrics['hit@all']:.4f}")


def main(argv: list[str] | None = None) -> None:
    from spectune.augmentor.config import INFORMATION_TYPES as CANONICAL_TYPES

    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = args.output or _default_output(args.input[0], args.fmt)
    config = build_rollout_config(args)

    print(f"loading  {', '.join(args.input)}")
    samples = load_samples(args.input)
    print(f"  {len(samples)} samples loaded total")
    total_samples = len(samples)

    type_map = {str(s.get("sample_id", "")): s.get("data_type", "none") for s in samples}
    total_by_type: dict[str, int] = defaultdict(int)
    for data_type in type_map.values():
        total_by_type[data_type] += 1

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

    records = run_with_backend(
        args,
        config,
        samples,
        checkpoint_path=ckpt_path if use_checkpoint else None,
    )
    records = prior_records + records

    accepted = len(records)
    accepted_pct = 100 * accepted / total_samples if total_samples else 0.0
    print(f"\naccepted {accepted}/{total_samples} samples ({accepted_pct:.1f}%)")

    if args.fmt == "parquet":
        write_parquet(records, output_path)
    else:
        write_jsonl(records, output_path)
    print(f"wrote    {output_path}")

    if use_checkpoint and ckpt_path.exists():
        ckpt_path.unlink()
        print(f"removed  {ckpt_path}")

    if args.no_hit_at_k:
        return

    overall = compute_hit_at_k_metrics(records, total_samples=total_samples)
    _print_metrics(overall, "overall")

    by_type: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        data_type = type_map.get(record.sample_id, "none")
        by_type[data_type].append(record)

    for data_type in CANONICAL_TYPES:
        n = total_by_type.get(data_type, 0)
        if n == 0:
            continue
        metrics = compute_hit_at_k_metrics(by_type.get(data_type, []), total_samples=n)
        _print_metrics(metrics, data_type)


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]
