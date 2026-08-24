"""Shared CLI building blocks for ``spectune.rollout``'s two entry points.

``python -m spectune.rollout`` (SFT rollout / rejection sampling) and
``python -m spectune.rollout.eval`` (hit@k evaluation) both need to run a
:class:`~spectune.rollout.rollout.Rollout` against one of three LLM backends
(local vLLM, litellm, or a plain HTTP endpoint). Centralizing the argument
definitions and backend dispatch here keeps both mains thin and prevents the
sampling defaults from drifting apart.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys

from spectune.llm import BACKENDS, LlmConfig, create_llm_client, vllm_server
from spectune.tools.config import NMR_GENERATE_MAX_TOPK

from .config import DEFAULT_MAX_ASSISTANT_TURNS, RolloutConfig
from .rollout import Rollout, RolloutRecord
from .sampling.base import BaseSampler

JsonDict = dict


def add_llm_args(p: argparse.ArgumentParser) -> None:
    """Register the sampling / tool / backend flags shared by both mains."""
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Max rejection-sampling rounds per sample.  "
            "When omitted, each sample is called once and accepted unconditionally.  "
            "Only has an effect when the caller passes a sampler to run_with_backend() "
            "(e.g. `python -m spectune.rollout`); `python -m spectune.rollout.eval` "
            "ignores it so evaluation measures the model's unassisted output."
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
        "--skills",
        nargs="+",
        metavar="PATH",
        default=None,
        help=(
            "Skill file(s) (default: .md) whose contents are appended to the end of "
            "the system prompt, one per file separated by a newline.  "
            "Off by default."
        ),
    )
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default=os.getenv("SPECTUNE_LLM_BACKEND", "http"),
        help=(
            "LLM backend: 'http' talks directly to SPECTUNE_LLM_BASE_URL (default), "
            "'litellm' routes through litellm using LITELLM_API_KEY/LITELLM_API_BASE, "
            "'local' auto-starts a vLLM server using --model as the model path.  "
            "Defaults to SPECTUNE_LLM_BACKEND env var, then 'http'."
        ),
    )
    p.add_argument(
        "--model",
        metavar="NAME_OR_PATH",
        default=None,
        help=(
            "Model name or path.  "
            "For 'http': overrides SPECTUNE_LLM_MODEL (e.g. 'qwen3-max').  "
            "For 'litellm': overrides LITELLM_MODEL (e.g. 'openai/gpt-4o').  "
            "For 'local': path to a HuggingFace model directory "
            "(e.g. /fs_mol/liujiarun/models/qwen3-32b); required, no env var fallback -- "
            "which local model to spin up a vLLM server for is a per-run choice, "
            "not something to bake into secrets.env."
        ),
    )

    local_group = p.add_argument_group(
        "local vLLM (--backend local)",
        "Options for the auto-started vLLM server when --backend local is used.",
    )
    local_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        metavar="N",
        help="Number of GPUs for vLLM tensor parallelism (default: 1).",
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

    p.add_argument("--no-progress", action="store_true", help="Suppress progress bar.")


def build_rollout_config(args: argparse.Namespace) -> RolloutConfig:
    """Build a :class:`RolloutConfig` from the flags registered by :func:`add_llm_args`."""
    return RolloutConfig(
        max_rounds=args.rounds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        nmr_gen_topk=args.nmr_gen_topk,
        tool_names=tuple(args.tools) if args.tools else (),
        max_assistant_turns=args.max_assistant_turns,
        max_concurrency=args.max_concurrency,
        show_progress=not args.no_progress,
        skills=tuple(args.skills) if args.skills else (),
    )


def run_with_backend(
    args: argparse.Namespace,
    config: RolloutConfig,
    samples: list[JsonDict],
    *,
    sampler: BaseSampler | None = None,
    checkpoint_path=None,
) -> list[RolloutRecord]:
    """Dispatch a rollout run onto whichever backend ``--backend`` selected.

    Shared by ``python -m spectune.rollout`` and ``python -m spectune.rollout.eval``
    so both read the exact same backend-selection and error-message logic.
    """
    if args.backend == "local":
        model_path = args.model
        if not model_path:
            print("error: --backend local requires --model <path>", file=sys.stderr)
            sys.exit(1)
        vllm_extra: list[str] = []
        if args.max_model_len is not None:
            vllm_extra += ["--max-model-len", str(args.max_model_len)]
        if args.gpu_memory_utilization is not None:
            vllm_extra += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        with vllm_server(
            model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            port=args.vllm_port,
            extra_args=vllm_extra or None,
        ) as (base_url, model_name):
            llm_config = LlmConfig(base_url=base_url, model=model_name)
            rollout = Rollout(config, llm_config)
            return asyncio.run(rollout.run(samples, sampler, checkpoint_path=checkpoint_path))

    if args.backend == "litellm":
        overrides: dict[str, object] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "max_concurrency": config.max_concurrency,
        }
        if args.model:
            overrides["model"] = args.model
        llm = create_llm_client("litellm", **overrides)
        if not llm.available:
            print(
                "error: litellm endpoint not configured — set LITELLM_API_KEY, "
                "LITELLM_API_BASE, and LITELLM_MODEL (see secrets.env.example)",
                file=sys.stderr,
            )
            sys.exit(1)
        rollout = Rollout(config, llm=llm)
        return asyncio.run(rollout.run(samples, sampler, checkpoint_path=checkpoint_path))

    llm_config = LlmConfig()
    if args.model:
        llm_config = dataclasses.replace(llm_config, model=args.model)
    if not llm_config.available:
        print(
            "error: LLM endpoint not configured — set SPECTUNE_LLM_BASE_URL and "
            "SPECTUNE_LLM_MODEL, or use --backend local with --model <path> "
            "(see secrets.env.example)",
            file=sys.stderr,
        )
        sys.exit(1)
    rollout = Rollout(config, llm_config)
    return asyncio.run(rollout.run(samples, sampler, checkpoint_path=checkpoint_path))


__all__ = ["add_llm_args", "build_rollout_config", "run_with_backend"]
