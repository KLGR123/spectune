"""Entry point for ``python -m spectune.tools``.

Writes a verl tool-config YAML listing the Spectune tools that should be
exposed during rollout.

Usage::

    python -m spectune.tools write-config \
        --output examples/grpo/tools_config.yaml

    # restrict to a specific subset of tools
    python -m spectune.tools write-config \
        --output examples/grpo/tools_config.yaml \
        --tools nmr_generate nmr_repair nmr_forward_predict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES
from spectune.tools.verl import write_tools_config


def _cmd_write_config(args: argparse.Namespace) -> None:
    tool_names = tuple(args.tools) if args.tools else DEFAULT_RL_TOOL_NAMES
    dest = write_tools_config(Path(args.output), tool_names=tool_names)
    print(f"wrote {dest}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectune.tools",
        description="Spectune tool utilities.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    wc = sub.add_parser(
        "write-config",
        help="Write a verl tool_config_path YAML for Spectune tools.",
    )
    wc.add_argument(
        "--output", required=True, metavar="PATH",
        help="Destination YAML file (parent dirs are created automatically).",
    )
    wc.add_argument(
        "--tools", nargs="+", metavar="TOOL",
        default=None,
        help=(
            "Tool names to include (default: DEFAULT_RL_TOOL_NAMES = "
            + ", ".join(DEFAULT_RL_TOOL_NAMES)
            + ")."
        ),
    )
    wc.set_defaults(func=_cmd_write_config)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
