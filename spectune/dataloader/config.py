"""Configuration objects for spectune dataset loaders.

Kept separate from :mod:`spectune.tools.config` (which holds *tool* configs):
dataset loaders are a distinct concern from tools, each new data source gets
its own config here, and none of this is consumed by :class:`~spectune.tools.manager.ToolManager`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

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
        "train": ("raw",),
        "test": (
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
    where ``preprocess()`` caches normalized JSON-Lines truth files. Both
    default to this cluster's layout but are fully overridable.

    ``sources`` maps source names to raw filenames under ``raw_dir``.
    ``truth_splits`` assigns those sources to output truth splits. By default,
    the raw parquet becomes ``nmrexp_truth_train.jsonl`` and all checked CSV
    sources are merged into ``nmrexp_truth_test.jsonl``. These remain *truth*
    splits: downstream code will independently combine each with sampled seeds
    to produce the final train/test datasets.

    Add entries to both mappings (or pass replacements) to ingest additional
    sources. ``min_quality``/``drop_qc_wrong`` only affect checked CSV sources,
    since the raw export carries no verification columns.

    ``show_progress`` prints row-processed/rate/ETA to stderr while
    ``preprocess()`` runs; disable it for quiet/non-interactive runs (e.g. CI).
    """

    raw_dir: str = field(default_factory=lambda: os.getenv("NMREXP_RAW_DIR", "/root/data/NMRexp"))
    processed_dir: str = field(default_factory=lambda: os.getenv("SPECTUNE_DATASETS_DIR", "./datasets"))
    dataset_name: str = "NMRexp"
    sources: Mapping[str, str] = field(default_factory=lambda: _DEFAULT_NMREXP_SOURCES)
    truth_splits: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: _DEFAULT_NMREXP_TRUTH_SPLITS)
    canonicalize_smiles: bool = True
    drop_qc_wrong: bool = True
    min_quality: Literal["any", "same_skeleton", "same_molecule"] = "any"
    allowed_nmr_types: tuple[str, ...] = ()
    max_records: int = 0
    show_progress: bool = True


@dataclass(frozen=True, slots=True)
class SpecXMasterDataLoaderConfig:
    """Settings for :class:`~spectune.dataloader.specxmaster.SpecXMasterDataLoader`.

    SpecXMaster is raw production traffic (zipped per-conversation exports),
    not a labeled dataset, so unlike :class:`NmrExpDataLoaderConfig` there is
    no ``sources``/split mapping: every ``*.zip`` under ``raw_dir`` is read and
    merged into one ``queries`` pool, cached as a single JSON-Lines file under
    ``processed_dir``. Queries are meant to be clustered into seed questions
    downstream, then matched independently against
    :class:`NmrExpDataLoaderConfig`'s train/test truth splits to build the final
    train/test datasets.

    ``zip_glob`` selects which archives to read (relative to ``raw_dir``).
    ``show_progress`` prints per-archive progress to stderr while
    ``preprocess()`` runs; disable it for quiet/non-interactive runs (e.g. CI).
    """

    raw_dir: str = field(default_factory=lambda: os.getenv("SPECXMASTER_RAW_DIR", "/fs_mol/liujiarun/data/SpecXMaster"))
    processed_dir: str = field(default_factory=lambda: os.getenv("SPECTUNE_DATASETS_DIR", "./datasets"))
    dataset_name: str = "SpecXMaster"
    zip_glob: str = "*.zip"
    show_progress: bool = True


__all__ = ["NmrExpDataLoaderConfig", "SpecXMasterDataLoaderConfig"]
