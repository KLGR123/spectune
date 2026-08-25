"""CLI: ``python -m spectune.artifacts ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectune.artifacts.compile import ArtifactCompileConfig, compile_jsonl_file, write_interaction_config
from spectune.format.v1 import DEFAULT_PROMPT_VERSION, FORMAT_SPEC_VERSION, SYSTEM_PROMPT_REGISTRY, get_system_prompt
from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Spectune JSONL datasets into verl-ready training artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="JSONL to parquet/jsonl RLHF rows")
    compile_parser.add_argument(
        "--input",
        "-i",
        required=True,
        nargs="+",
        help=(
            "One or more Spectune JSONL paths. When multiple paths are given, "
            "their samples are merged and shuffled together before compiling."
        ),
    )
    compile_parser.add_argument("--output", "-o", required=True, help="Output .parquet or .jsonl")
    compile_parser.add_argument("--split", default="train", help="Split label stored in extra_info")
    compile_parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Preserve input order instead of shuffling merged samples (default: shuffle)",
    )
    compile_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the merge shuffle, for reproducible ordering",
    )
    compile_parser.add_argument(
        "--data-source",
        nargs="+",
        default=["spectune/nmrexp"],
        metavar="SOURCE",
        help=(
            "data_source field written into each row. Pass one value to apply it to every "
            "--input path, or exactly one value per --input path to tag each source "
            "differently (e.g. --data-source spectune/nmrexp spectune/others)."
        ),
    )
    compile_parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        metavar="TOOL",
        help=(
            "Tool names to expose via extra_info.tools_kwargs. "
            f"Defaults to the standard RL set: {', '.join(DEFAULT_RL_TOOL_NAMES)}"
        ),
    )
    compile_parser.add_argument(
        "--reward-config-json",
        default=None,
        help="Optional JSON object merged into extra_info.reward_config",
    )
    compile_parser.add_argument(
        "--include-tool-schemas",
        action="store_true",
        help="Embed full OpenAI schemas in every row (default: store tool_names only)",
    )
    compile_parser.add_argument(
        "--prompt",
        choices=list(SYSTEM_PROMPT_REGISTRY),
        default=DEFAULT_PROMPT_VERSION,
        metavar="VERSION",
        help=(
            f"System prompt version baked into the compiled rows (default: "
            f"{DEFAULT_PROMPT_VERSION}). Choices: {', '.join(SYSTEM_PROMPT_REGISTRY)}."
        ),
    )
    compile_parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to write a small JSON compile manifest",
    )
    compile_parser.add_argument(
        "--interaction-config",
        default=None,
        metavar="PATH",
        help="If given, write the verl interaction config YAML for ScriptedFollowupInteraction to this path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command != "compile":  # pragma: no cover
        raise SystemExit(f"unknown command: {args.command}")

    reward_config = {}
    if args.reward_config_json:
        loaded = json.loads(args.reward_config_json)
        if not isinstance(loaded, dict):
            raise SystemExit("--reward-config-json must decode to an object")
        reward_config = loaded

    tool_names = tuple(args.tools) if args.tools else tuple(DEFAULT_RL_TOOL_NAMES)

    if len(args.data_source) not in (1, len(args.input)):
        raise SystemExit(
            f"--data-source got {len(args.data_source)} values for {len(args.input)} --input paths; "
            "pass exactly one (broadcast to all) or one per --input path"
        )

    config = ArtifactCompileConfig(
        data_source=args.data_source[0],
        system_prompt=get_system_prompt(args.prompt),
        tool_names=tool_names,
        reward_config=reward_config,
        include_tool_schemas=bool(args.include_tool_schemas),
    )
    count = compile_jsonl_file(
        args.input,
        args.output,
        split=args.split,
        config=config,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        data_sources=args.data_source,
    )
    print(f"wrote {count} rows to {args.output} (format_spec={FORMAT_SPEC_VERSION})")

    if args.manifest:
        manifest = {
            "input": [str(Path(p)) for p in args.input],
            "output": str(Path(args.output)),
            "rows": count,
            "split": args.split,
            "format_spec": FORMAT_SPEC_VERSION,
            "data_source": list(args.data_source) if len(args.data_source) > 1 else config.data_source,
            "tool_names": list(config.tool_names),
        }
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.interaction_config:
        write_interaction_config(args.interaction_config)
        print(f"wrote interaction config to {args.interaction_config}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
