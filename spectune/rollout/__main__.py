"""Entry point for ``python -m spectune.rollout``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from spectune.llm import LlmConfig
from spectune.tools.config import NMR_GENERATE_MAX_TOPK

from .config import DEFAULT_MAX_ASSISTANT_TURNS, DEFAULT_OUTPUT_DIR, RolloutConfig
from .rollout import Rollout, write_jsonl, write_parquet
from .sampling.rejection import GtRejectionSampler


def _load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).expanduser().open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            records.append(value)
    return records


def _default_output(input_path: str, fmt: str) -> str:
    stem = Path(input_path).stem  # e.g. nmrexp_sft_2000
    name = stem.replace("_sft_", "_sft_rollout_")
    return str(Path(DEFAULT_OUTPUT_DIR) / f"{name}.{fmt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectune.rollout",
        description=(
            "Offline parallel LLM rollout with tool execution and rejection sampling.\n"
            "Reads a nmrexp_sft_*.jsonl dataset produced by "
            "`python -m spectune.augmentor`, runs a hermes tool-agent loop per "
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
        metavar="PATH",
        help="Input JSONL file (nmrexp_sft_*.jsonl from python -m spectune.augmentor).",
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
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Max rejection-sampling rounds per sample.  "
            "When omitted, each sample is called once and accepted unconditionally."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        metavar="T",
        help="Sampling temperature (default: 0.8).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        metavar="P",
        help="Top-p nucleus sampling (default: 0.95).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        metavar="N",
        help="Max tokens per LLM response (default: 4096).",
    )
    p.add_argument(
        "--nmr-gen-topk",
        type=int,
        default=10,
        metavar="K",
        help=(f"Default topk for nmr_generate candidate generation (default: 10, max: {NMR_GENERATE_MAX_TOPK})."),
    )
    p.add_argument(
        "--tools",
        nargs="+",
        metavar="TOOL",
        default=None,
        help="Tool names to expose to the model (default: DEFAULT_RL_TOOL_NAMES).",
    )
    p.add_argument(
        "--max-assistant-turns",
        type=int,
        default=DEFAULT_MAX_ASSISTANT_TURNS,
        metavar="N",
        help=f"Max LLM generations per agent loop (default: {DEFAULT_MAX_ASSISTANT_TURNS}).",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        metavar="N",
        help="Max concurrent LLM requests (default: 8).",
    )
    p.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help=("LLM model name to use (e.g. GPT-5.4).  Defaults to SPECTUNE_LLM_MODEL from the environment."),
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress progress bar.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = args.output or _default_output(args.input, args.fmt)

    config = RolloutConfig(
        max_rounds=args.rounds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        nmr_gen_topk=args.nmr_gen_topk,
        tool_names=tuple(args.tools) if args.tools else (),
        max_assistant_turns=args.max_assistant_turns,
        max_concurrency=args.max_concurrency,
        show_progress=not args.no_progress,
    )
    llm_config = LlmConfig()
    if args.model:
        import dataclasses

        llm_config = dataclasses.replace(llm_config, model=args.model)
    if not llm_config.available:
        print(
            "error: LLM endpoint not configured — set SPECTUNE_LLM_BASE_URL and "
            "SPECTUNE_LLM_MODEL (see secrets.env.example)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"loading  {args.input}")
    samples = _load_jsonl(args.input)
    print(f"  {len(samples)} samples loaded")

    rollout = Rollout(config, llm_config)
    sampler = GtRejectionSampler() if args.rounds is not None else None

    records = asyncio.run(rollout.run(samples, sampler))

    accepted = len(records)
    total = len(samples)
    print(f"accepted {accepted}/{total} samples ({100 * accepted / total:.1f}%)")

    if args.fmt == "parquet":
        write_parquet(records, output_path)
    else:
        write_jsonl(records, output_path)

    print(f"wrote    {output_path}")


if __name__ == "__main__":
    main()
