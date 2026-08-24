"""Configuration objects for spectune dataset loaders.

Kept separate from :mod:`spectune.tools.config` (which holds *tool* configs):
dataset loaders are a distinct concern from tools, each new data source gets
its own config here, and none of this is consumed by :class:`~spectune.tools.manager.ToolManager`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal


def default_datasets_dir() -> str:
    """Resolve the shared dataset cache/output directory.

    Every stage that reads or writes files under this directory (dataloader
    truth caches, the augmentor's enrichment cache and JSONL output, ...)
    resolves the same plain default through this one function, so overriding
    it once (via ``--datasets-dir`` on any entry point, or the
    ``processed_dir``/``datasets_dir`` config field directly) moves all of
    them together instead of each stage guessing its own default.
    """
    return "outputs/datasets"


_DEFAULT_NMREXP_SOURCES: Mapping[str, str] = MappingProxyType(
    {
        "raw": "NMRexp_10to24_1_1004.parquet",
        "checked": "test_300_checked.csv",
        "checked_boron": "B_50_checked.csv",
        "checked_fluorine": "F_50_checked.csv",
        "checked_phosphorus": "P_50_checked.csv",
        "checked_silicon": "Si_50_checked.csv",
        "checked_heteronuclei": "hetero_200_checked.csv",
    }
)
_DEFAULT_NMREXP_TRUTH_SPLITS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        # Named after data *provenance*, not the eventual RL-train/eval-test
        # role a sampled row ends up playing -- ``spectune.augmentor`` draws
        # all of its train/test/sft output splits from the same truth split
        # (by default "bulk"), so a truth-split name of "train"/"test" would
        # bleed into every downstream ``sample_id`` and be read as "this row
        # is training data", regardless of which output split it lands in.
        "bulk": ("raw",),
        "verified": (
            "checked",
            "checked_boron",
            "checked_fluorine",
            "checked_phosphorus",
            "checked_silicon",
            "checked_heteronuclei",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class NmrExpDataLoaderConfig:
    """Settings for :class:`~spectune.dataloader.nmrexp.NmrExpDataLoader`.

    ``raw_dir`` holds the raw NMRexp exports (the large unchecked ``raw``
    parquet plus the small human-checked CSV exports); ``processed_dir`` is
    where ``preprocess()`` caches normalized JSON-Lines truth files. Both have
    plain placeholder defaults; pass explicit values (or ``--raw-dir``/
    ``--datasets-dir`` on the CLI entry points) for anything but a quick local test.

    ``sources`` maps source names to raw filenames under ``raw_dir``.
    ``truth_splits`` assigns those sources to output truth splits. By default,
    the raw parquet becomes ``nmrexp_truth_bulk.jsonl`` and all checked CSV
    sources are merged into ``nmrexp_truth_verified.jsonl``. These remain
    *truth* splits, named after provenance (bulk/unverified vs. small/human-
    verified) rather than train/test role: downstream code (the augmentor)
    independently samples and slices whichever truth split it is pointed at
    to produce the final train/test/sft datasets.

    Add entries to both mappings (or pass replacements) to ingest additional
    sources. ``min_quality``/``drop_qc_wrong`` only affect checked CSV sources,
    since the raw export carries no verification columns.

    ``show_progress`` prints row-processed/rate/ETA to stderr while
    ``preprocess()`` runs; disable it for quiet/non-interactive runs (e.g. CI).

    Rows sharing the same canonical ``gt_smiles`` are always merged into a
    single record: ``nmr`` holds the first block (backwards-compatible),
    ``nmr_list`` carries all blocks for that molecule, and multi-row groups
    additionally expose ``merged_sample_ids``.
    """

    raw_dir: str = "/root/data/NMRexp"
    processed_dir: str = field(default_factory=default_datasets_dir)
    dataset_name: str = "NMRexp"
    sources: Mapping[str, str] = field(default_factory=lambda: _DEFAULT_NMREXP_SOURCES)
    truth_splits: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: _DEFAULT_NMREXP_TRUTH_SPLITS)
    canonicalize_smiles: bool = True
    drop_qc_wrong: bool = True
    min_quality: Literal["any", "same_skeleton", "same_molecule"] = "any"
    allowed_nmr_types: tuple[str, ...] = ()
    max_records: int = 0
    show_progress: bool = True


__all__ = ["NmrExpDataLoaderConfig", "default_datasets_dir"]
