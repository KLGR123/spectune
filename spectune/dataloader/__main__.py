"""Entry point for ``python -m spectune.dataloader``.

Two sub-commands:

  preprocess   Read raw NMRexp exports, merge rows with the same canonical
               SMILES, and write cached JSONL truth files.

  info         Print dataset sizes without rebuilding.
"""

from __future__ import annotations

import argparse
import json
import sys

from spectune.dataloader import NmrExpDataLoader
from spectune.dataloader.config import NmrExpDataLoaderConfig


def _build_preprocess_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "preprocess",
        help="Build (or rebuild) cached JSONL truth files from raw NMRexp exports.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        metavar="SPLIT",
        default=None,
        help="Which splits to build (default: all configured splits, usually train and test).",
    )
    p.add_argument(
        "--raw-dir",
        metavar="DIR",
        default=None,
        help="Directory containing raw NMRexp exports (overrides NMREXP_RAW_DIR env var).",
    )
    p.add_argument(
        "--datasets-dir",
        metavar="DIR",
        default=None,
        help="Output directory for processed JSONL files (overrides SPECTUNE_DATASETS_DIR env var).",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=0,
        metavar="N",
        help="Cap total kept records per split (0 = unlimited, default: 0).",
    )
    p.add_argument(
        "--allowed-nmr-types",
        nargs="+",
        metavar="TYPE",
        default=None,
        help="Whitelist of NMR types to keep, e.g. '1H NMR' '13C NMR' (default: all types).",
    )
    p.add_argument(
        "--min-quality",
        choices=("any", "same_skeleton", "same_molecule"),
        default="any",
        help="Minimum QA quality gate for checked CSV sources (default: any).",
    )
    p.add_argument(
        "--no-drop-qc-wrong",
        action="store_true",
        help="Keep rows flagged as QC-wrong by human reviewers (dropped by default).",
    )
    p.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Skip RDKit SMILES canonicalization (not recommended).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild even if the cached JSONL already exists.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-row progress output.",
    )


def _build_info_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "info",
        help="Print the size of each processed truth split (builds cache if missing).",
    )
    p.add_argument(
        "--raw-dir",
        metavar="DIR",
        default=None,
        help="Directory containing raw NMRexp exports.",
    )
    p.add_argument(
        "--datasets-dir",
        metavar="DIR",
        default=None,
        help="Directory for processed JSONL files.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        metavar="SPLIT",
        default=None,
        help="Splits to inspect (default: all).",
    )


def _make_config(args: argparse.Namespace) -> NmrExpDataLoaderConfig:
    kwargs: dict = {}
    if getattr(args, "raw_dir", None):
        kwargs["raw_dir"] = args.raw_dir
    if getattr(args, "datasets_dir", None):
        kwargs["processed_dir"] = args.datasets_dir
    if getattr(args, "max_records", None):
        kwargs["max_records"] = args.max_records
    if getattr(args, "allowed_nmr_types", None):
        kwargs["allowed_nmr_types"] = tuple(args.allowed_nmr_types)
    if getattr(args, "min_quality", None) and args.min_quality != "any":
        kwargs["min_quality"] = args.min_quality
    if getattr(args, "no_drop_qc_wrong", False):
        kwargs["drop_qc_wrong"] = False
    if getattr(args, "no_canonicalize", False):
        kwargs["canonicalize_smiles"] = False
    if getattr(args, "no_progress", False):
        kwargs["show_progress"] = False
    return NmrExpDataLoaderConfig(**kwargs)


def _cmd_preprocess(args: argparse.Namespace) -> None:
    config = _make_config(args)
    loader = NmrExpDataLoader(config)
    summaries = loader.preprocess(args.splits, overwrite=args.overwrite)
    for split, summary in summaries.items():
        print(json.dumps({"split": split, **summary}, ensure_ascii=False))


def _cmd_info(args: argparse.Namespace) -> None:
    config = _make_config(args)
    loader = NmrExpDataLoader(config)
    splits = list(args.splits) if args.splits else list(config.truth_splits)
    for split in splits:
        dataset = loader.load_truth(split)
        print(f"{split}: {len(dataset):,} records  →  {loader._processed_path(split)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectune.dataloader",
        description="Preprocess NMRexp raw exports into merged JSONL truth files.",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    _build_preprocess_parser(sub)
    _build_info_parser(sub)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        _cmd_preprocess(args)
    elif args.command == "info":
        _cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
