"""Entry point for ``python -m spectune.augmentor``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectune.augmentor.augmentor import Augmentor
from spectune.augmentor.config import (
    INFORMATION_TYPES,
    AugmentorConfig,
)


def _parse_information_mix(value: str) -> dict[str, float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f'--information-mix must be a valid JSON object, e.g. \'{{"none":0.2,"formula":0.2,...}}\': {exc}'
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--information-mix must be a JSON object")
    unknown = [k for k in parsed if k not in INFORMATION_TYPES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown information type(s) {unknown}; expected subset of {list(INFORMATION_TYPES)}"
        )
    return {k: float(v) for k, v in parsed.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectune.augmentor",
        description="Build an augmented NMR structure-elucidation dataset.",
    )

    # Output
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help=(
            "Destination JSONL file path.  When the value contains '{name}', "
            "three disjoint splits (train/test/sft) are written in one pass; "
            "{size} is also replaced with the corresponding size.  "
            "Example: 'datasets/nmrexp_{name}_{size}.jsonl'."
        ),
    )
    p.add_argument(
        "--sizes",
        nargs=3,
        type=int,
        default=[20000, 200, 2000],
        metavar=("TRAIN", "TEST", "SFT"),
        help="Sizes for train/test/sft splits when --output uses {name} template (default: 20000 200 2000).",
    )

    # Sampling
    p.add_argument("--dataset-name", default="nmrexp", metavar="NAME", help="Source dataset name (default: nmrexp).")
    p.add_argument("--split", default="train", metavar="SPLIT", help="Dataset split to draw from (default: train).")
    p.add_argument(
        "--sample-size", type=int, default=0, metavar="N", help="Number of rows to sample (0 = full split, default: 0)."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--no-progress", action="store_true", help="Suppress progress bars.")

    # Information mix
    p.add_argument(
        "--information-mix",
        type=_parse_information_mix,
        metavar="JSON",
        default=None,
        help=(
            "JSON object mapping information type to share (must sum to 1). "
            "Types: " + ", ".join(INFORMATION_TYPES) + ". "
            'Default: {"none":0.30,"formula":0.20,"structure":0.20,'
            '"reaction":0.15,"fragment":0.15}.'
        ),
    )
    p.add_argument(
        "--followup-probability",
        type=float,
        default=0.4,
        metavar="P",
        help="Probability of adding information as a follow-up turn (default: 0.4).",
    )

    # Construction
    p.add_argument(
        "--max-fragment-items", type=int, default=2, metavar="N", help="Max fragment hints per sample (default: 2)."
    )
    p.add_argument(
        "--max-reaction-items",
        type=int,
        default=2,
        metavar="N",
        help="Max reaction context items per sample (default: 2).",
    )

    # Noise
    p.add_argument(
        "--nmr-noise-ratio",
        type=float,
        default=0.0,
        metavar="P",
        help="Share of rows with corrupted NMR spectrum (default: 0.0).",
    )
    p.add_argument(
        "--nmr-noise-strength",
        type=float,
        default=0.3,
        metavar="S",
        help="Corruption intensity in (0, 1] (default: 0.3).",
    )
    p.add_argument(
        "--formula-noise-ratio",
        type=float,
        default=0.0,
        metavar="P",
        help="Share of formula-condition rows with corrupted formula (default: 0.0).",
    )
    p.add_argument(
        "--modal-drop-ratio",
        type=float,
        default=0.5,
        metavar="P",
        help=(
            "Probability that a multi-modality row drops some of its spectra before "
            "query construction; at least one spectrum is always kept (default: 0.0)."
        ),
    )

    # LLM rewrite
    p.add_argument("--use-llm-rewrite", action="store_true", help="Rewrite queries with an LLM (disabled by default).")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_kwargs: dict = dict(
        dataset_name=args.dataset_name,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        show_progress=not args.no_progress,
        followup_probability=args.followup_probability,
        max_fragment_items=args.max_fragment_items,
        max_reaction_items=args.max_reaction_items,
        nmr_noise_ratio=args.nmr_noise_ratio,
        nmr_noise_strength=args.nmr_noise_strength,
        formula_noise_ratio=args.formula_noise_ratio,
        modal_drop_ratio=args.modal_drop_ratio,
        use_llm_rewrite=args.use_llm_rewrite,
    )
    if args.information_mix is not None:
        config_kwargs["information_mix"] = args.information_mix

    config = AugmentorConfig(**config_kwargs)
    augmentor = Augmentor(config)
    if "{name}" in args.output:
        names = ("train", "test", "sft")
        sizes = tuple(args.sizes)
        output_paths = {
            name: Path(args.output.format(name=name, size=size)) for name, size in zip(names, sizes, strict=True)
        }
        augmentor.build_splits(sizes=sizes, output_paths=output_paths)
    else:
        augmentor.build(output_path=args.output)


if __name__ == "__main__":
    main()
