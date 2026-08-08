"""Entry point for ``python -m spectune.classifier``.

Assigns MiniBatchKMeans cluster labels to a preprocessed NMRexp truth JSONL
file, rewriting it in place. Optionally produces a 2D/3D scatter-plot PNG.
"""

from __future__ import annotations

import argparse
import json

from spectune.classifier import Classifier, ClassifierConfig
from spectune.dataloader.base import JsonlDataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectune.classifier",
        description=(
            "Cluster an NMRexp truth JSONL file by molecular structure and annotate each record with a 'cluster' label."
        ),
    )
    p.add_argument(
        "input",
        metavar="JSONL",
        help="Path to the NMRexp truth JSONL file produced by 'python -m spectune.dataloader preprocess'.",
    )
    p.add_argument(
        "--clusters",
        type=int,
        default=256,
        metavar="N",
        help="Number of KMeans clusters (default: 256).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel RDKit featurization workers (default: min(16, cpu_count)).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans and visualization (default: 42).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        metavar="N",
        help="MiniBatchKMeans batch size (default: 2048).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="KMeans training passes over the data (default: 2).",
    )
    p.add_argument(
        "--property-weight",
        type=float,
        default=0.35,
        metavar="W",
        help="Weight of standardized molecular properties relative to fingerprint (default: 0.35).",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory for temporary files and visualization PNG (default: SPECTUNE_OUTPUT_DIR or ./outputs).",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate a cluster scatter-plot PNG (requires matplotlib and scikit-learn).",
    )
    p.add_argument(
        "--visualization-path",
        metavar="PATH",
        default=None,
        help="Override the visualization PNG output path.",
    )
    p.add_argument(
        "--visualization-dim",
        type=int,
        choices=(2, 3),
        default=2,
        help="Dimensionality of the cluster scatter-plot (default: 2).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress progress output.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_kwargs: dict = dict(
        nmr_n_clusters=args.clusters,
        random_state=args.seed,
        batch_size=args.batch_size,
        kmeans_epochs=args.epochs,
        property_weight=args.property_weight,
        visualize=args.visualize,
        show_progress=not args.no_progress,
        visualization_dim=args.visualization_dim,
    )
    if args.output_dir:
        config_kwargs["output_dir"] = args.output_dir
    if args.workers is not None:
        config_kwargs["nmr_num_workers"] = args.workers

    config = ClassifierConfig(**config_kwargs)
    classifier = Classifier(config)
    dataset = JsonlDataset(args.input)
    classifier.classify(dataset, visualization_path=args.visualization_path)

    summary = classifier.last_summary or {}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
