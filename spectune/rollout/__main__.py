"""Entry point for ``python -m spectune.rollout``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path

from spectune.llm import LlmConfig
from spectune.tools.config import NMR_GENERATE_MAX_TOPK

from .config import DEFAULT_MAX_ASSISTANT_TURNS, DEFAULT_OUTPUT_DIR, RolloutConfig
from .rollout import Rollout, RolloutRecord, write_jsonl, write_parquet
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


def _checkpoint_path(output_path: str) -> Path:
    p = Path(output_path)
    return p.parent / (p.stem + ".checkpoint.jsonl")


def _load_checkpoint(ckpt_path: Path) -> tuple[list[RolloutRecord], set[str]]:
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
                r = RolloutRecord(
                    sample_id=d["sample_id"],
                    gt_smiles=d["gt_smiles"],
                    messages=d["messages"],
                    reward_score=d["reward_score"],
                    reward_details=d["reward_details"],
                    n_rounds=d["n_rounds"],
                )
                records.append(r)
                ids.add(r.sample_id)
            except (json.JSONDecodeError, KeyError):
                pass
    return records, ids


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
        help="LLM model name to use (e.g. GPT-5.4).  Defaults to SPECTUNE_LLM_MODEL from the environment.",
    )

    # local vLLM options
    
    local_group = p.add_argument_group(
        "local vLLM",
        "Launch a local vLLM server instead of using a hosted API endpoint.\n"
        "Set SPECTUNE_LOCAL_MODEL_PATH in secrets.env to make these the defaults.",
    )
    local_group.add_argument(
        "--local-model",
        metavar="PATH",
        default=os.getenv("SPECTUNE_LOCAL_MODEL_PATH", ""),
        help=(
            "Path to a local HuggingFace model directory (e.g. /fs_mol/liujiarun/models/qwen3-32b).  "
            "When set, spectune starts a vLLM server automatically on a free port and uses it for "
            "rollout; SPECTUNE_LLM_BASE_URL / SPECTUNE_LLM_MODEL are ignored.  "
            "Defaults to SPECTUNE_LOCAL_MODEL_PATH env var."
        ),
    )
    local_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.getenv("SPECTUNE_VLLM_TENSOR_PARALLEL_SIZE", "1")),
        metavar="N",
        help=(
            "Number of GPUs for vLLM tensor parallelism (default: 1).  "
            "Defaults to SPECTUNE_VLLM_TENSOR_PARALLEL_SIZE env var."
        ),
    )
    local_group.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Fixed port for the local vLLM server (default: random free port).",
    )
    local_group.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum sequence length passed to vLLM as --max-model-len.  "
            "Controls KV-cache allocation; defaults to the model's config value.  "
            "Set this to match --max-tokens to avoid OOM during CUDA graph capture."
        ),
    )
    local_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Fraction of GPU memory vLLM may use (0, 1] (default: vLLM default 0.90).  "
            "Lower this (e.g. 0.85) to leave headroom for CUDA graph capture."
        ),
    )

    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress progress bar.",
    )
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

    print(f"loading  {args.input}")
    samples = _load_jsonl(args.input)
    print(f"  {len(samples)} samples loaded")

    use_checkpoint = not args.no_checkpoint
    ckpt_path = _checkpoint_path(output_path)
    prior_records: list[RolloutRecord] = []

    if use_checkpoint:
        prior_records, completed_ids = _load_checkpoint(ckpt_path)
        if prior_records:
            print(f"checkpoint  {len(prior_records)} already-done records found in {ckpt_path}")
        samples = [s for s in samples if str(s.get("sample_id", "")) not in completed_ids]
        if len(prior_records):
            print(f"  {len(samples)} remaining samples to process")

    sampler = GtRejectionSampler() if args.rounds is not None else None

    if args.local_model and not args.model:
        from ._vllm import vllm_server

        vllm_extra: list[str] = []
        if args.max_model_len is not None:
            vllm_extra += ["--max-model-len", str(args.max_model_len)]
        if args.gpu_memory_utilization is not None:
            vllm_extra += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]

        with vllm_server(
            args.local_model,
            tensor_parallel_size=args.tensor_parallel_size,
            port=args.vllm_port,
            extra_args=vllm_extra or None,
        ) as (base_url, model_name):
            llm_config = LlmConfig(base_url=base_url, model=model_name)
            rollout = Rollout(config, llm_config)
            records = asyncio.run(rollout.run(samples, sampler, checkpoint_path=ckpt_path if use_checkpoint else None))
    else:
        llm_config = LlmConfig()
        if args.model:
            llm_config = dataclasses.replace(llm_config, model=args.model)
        if not llm_config.available:
            print(
                "error: LLM endpoint not configured — set SPECTUNE_LLM_BASE_URL and "
                "SPECTUNE_LLM_MODEL, or pass --local-model (see secrets.env.example)",
                file=sys.stderr,
            )
            sys.exit(1)
        rollout = Rollout(config, llm_config)
        records = asyncio.run(rollout.run(samples, sampler, checkpoint_path=ckpt_path if use_checkpoint else None))

    records = prior_records + records

    accepted = len(records)
    total = len(samples)
    print(f"accepted {accepted}/{total} samples ({100 * accepted / total:.1f}%)")

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
