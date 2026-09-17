"""Entry point for ``python -m spectune.tools``.

Writes a verl tool-config YAML listing the Spectune tools that should be
exposed during rollout.

Usage::

    python -m spectune.tools write-config \
        --output outputs/datasets/verl/tools_config.yaml

    # restrict to a specific subset of tools
    python -m spectune.tools write-config \
        --output outputs/datasets/verl/tools_config.yaml \
        --tools nmr_generate nmr_repair nmr_forward_predict

    # raise the nmr_generate candidate budget
    python -m spectune.tools write-config \
        --output outputs/datasets/verl/tools_config.yaml \
        --nmr-gen-topk 30

Also provides ``build-reaction-index``, a one-time offline step that builds
prebuilt parquet indexes for ``reaction_local_index_search``::

    python -m spectune.tools build-reaction-index \
        --output-dir outputs/reaction_index \
        --sources uspto chempile pistachio \
        --workers 16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES
from spectune.tools.config import NMR_GENERATE_MAX_TOPK, ReactionLocalIndexSearchConfig
from spectune.tools.reaction_index_build import build_reaction_index
from spectune.tools.verl import write_tools_config


def _cmd_write_config(args: argparse.Namespace) -> None:
    tool_names = tuple(args.tools) if args.tools else DEFAULT_RL_TOOL_NAMES
    dest = write_tools_config(
        Path(args.output),
        tool_names=tool_names,
        nmr_gen_topk=args.nmr_gen_topk,
    )
    print(f"wrote {dest}")


def _cmd_build_reaction_index(args: argparse.Namespace) -> None:
    # ReactionLocalIndexSearchConfig() reads the RXN_LOCAL_INDEX_* raw paths from
    # the environment; --uspto-csv/--chempile-parquet/--pistachio-smi override them.
    config = ReactionLocalIndexSearchConfig()
    workers = args.workers if args.workers is not None else min(16, os.cpu_count() or 1)
    summaries = build_reaction_index(
        output_dir=args.output_dir,
        sources=args.sources,
        uspto_csv_path=args.uspto_csv or config.uspto_csv_path,
        chempile_parquet_path=args.chempile_parquet or config.chempile_parquet_path,
        pistachio_smi_path=args.pistachio_smi or config.pistachio_smi_path,
        max_records=args.max_records,
        workers=workers,
    )
    for summary in summaries:
        print(summary)


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
        "--output",
        required=True,
        metavar="PATH",
        help="Destination YAML file (parent dirs are created automatically).",
    )
    wc.add_argument(
        "--tools",
        nargs="+",
        metavar="TOOL",
        default=None,
        help=("Tool names to include (default: DEFAULT_RL_TOOL_NAMES = " + ", ".join(DEFAULT_RL_TOOL_NAMES) + ")."),
    )
    wc.add_argument(
        "--nmr-gen-topk",
        type=int,
        default=None,
        metavar="K",
        help=(f"Default topk for nmr_generate candidate generation (default: 10, max: {NMR_GENERATE_MAX_TOPK})."),
    )
    wc.set_defaults(func=_cmd_write_config)

    bi = sub.add_parser(
        "build-reaction-index",
        help="Build prebuilt reaction_local_index_search parquet indexes (one-time offline step).",
    )
    bi.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to write {source}.parquet into; point RXN_LOCAL_INDEX_PREBUILT_DIR here to use it.",
    )
    bi.add_argument(
        "--sources",
        nargs="+",
        choices=["uspto", "chempile", "pistachio"],
        default=None,
        metavar="SOURCE",
        help="Sources to build (default: all three that have a configured raw path).",
    )
    bi.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel RDKit canonicalization workers (default: min(16, cpu_count)).",
    )
    bi.add_argument(
        "--max-records",
        type=int,
        default=0,
        metavar="N",
        help="Cap records read per source (0 = unlimited, default: 0).",
    )
    bi.add_argument(
        "--uspto-csv",
        default="",
        metavar="PATH",
        help="Override RXN_LOCAL_INDEX_USPTO_CSV for this build.",
    )
    bi.add_argument(
        "--chempile-parquet",
        default="",
        metavar="PATH",
        help="Override RXN_LOCAL_INDEX_CHEMPILE_PARQUET for this build.",
    )
    bi.add_argument(
        "--pistachio-smi",
        default="",
        metavar="PATH",
        help="Override RXN_LOCAL_INDEX_PISTACHIO_SMI for this build.",
    )
    bi.set_defaults(func=_cmd_build_reaction_index)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
