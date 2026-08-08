"""Configuration for downstream dataset clustering."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_output_dir() -> str:
    configured = os.getenv("SPECTUNE_OUTPUT_DIR") or os.getenv("OUTPUT_PATH")
    return configured or str(Path(__file__).resolve().parents[2] / "outputs")


def _default_nmr_num_workers() -> int:
    configured = os.getenv("SPECTUNE_NMR_WORKERS")
    if configured:
        return int(configured)
    return min(16, os.cpu_count() or 1)


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Settings for :class:`~spectune.classifier.classifier.Classifier`.

    NMR structures use a Morgan fingerprint plus standardized, chemically
    meaningful molecular descriptors. MiniBatchKMeans keeps the method
    practical for multi-million-row truth files.

    Classification always recomputes representations and labels, replacing any
    existing ``cluster`` values. Set ``visualize=False`` for headless runs;
    callers can also override visualization per call.
    """

    output_dir: str = field(default_factory=_default_output_dir)
    nmr_n_clusters: int = 256
    random_state: int = 42
    batch_size: int = 2_048
    kmeans_epochs: int = 2
    visualize: bool = True
    show_progress: bool = True
    visualization_dpi: int = 400
    visualization_max_points: int = 20_000
    visualization_dim: int = 2
    cluster_key: str = "cluster"

    fingerprint_radius: int = 2
    fingerprint_bits: int = 2_048
    fingerprint_use_chirality: bool = True
    property_weight: float = 0.35
    nmr_num_workers: int = field(default_factory=_default_nmr_num_workers)
    nmr_worker_chunk_size: int = 512

    def __post_init__(self) -> None:
        positive = {
            "nmr_n_clusters": self.nmr_n_clusters,
            "batch_size": self.batch_size,
            "kmeans_epochs": self.kmeans_epochs,
            "visualization_dpi": self.visualization_dpi,
            "visualization_max_points": self.visualization_max_points,
            "fingerprint_radius": self.fingerprint_radius,
            "fingerprint_bits": self.fingerprint_bits,
            "nmr_num_workers": self.nmr_num_workers,
            "nmr_worker_chunk_size": self.nmr_worker_chunk_size,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"ClassifierConfig values must be positive: {invalid}")
        if self.property_weight < 0:
            raise ValueError("property_weight must be non-negative")
        if self.visualization_dim not in (2, 3):
            raise ValueError("visualization_dim must be 2 or 3")


__all__ = ["ClassifierConfig"]
