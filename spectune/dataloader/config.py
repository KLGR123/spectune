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


@dataclass(frozen=True, slots=True)
class NmrExpDataLoaderConfig:
    """Settings for :class:`~spectune.dataloader.nmrexp.NmrExpDataLoader`.

    ``raw_dir`` holds the raw NMRexp exports (the large unchecked ``raw``
    parquet plus the small human-checked CSV exports); ``processed_dir`` is
    where ``preprocess()`` caches the normalized, merged JSON-Lines ``truth``
    file that ``load()`` reads back. Both default to this cluster's layout but
    are fully overridable, e.g. for a different filesystem or a scratch
    directory.

    ``sources`` maps a source name to a raw filename under ``raw_dir``; every
    configured source is merged into one ``truth`` pool by ``preprocess()`` --
    there is no train/test split at this stage (that happens later, once
    ``truth`` is combined with clustered ``queries``-derived seeds into an
    augmented, trainable dataset). Add entries here (or pass a replacement
    mapping) to preprocess additional checked subsets. ``min_quality``/
    ``drop_qc_wrong`` only affect the checked CSV sources, since the raw
    export carries no verification columns.

    ``show_progress`` prints row-processed/rate/ETA to stderr while
    ``preprocess()`` runs; disable it for quiet/non-interactive runs (e.g. CI).
    """

    raw_dir: str = field(default_factory=lambda: os.getenv("NMREXP_RAW_DIR", "/root/data/NMRexp"))
    processed_dir: str = field(default_factory=lambda: os.getenv("SPECTUNE_DATASETS_DIR", "./datasets"))
    dataset_name: str = "NMRexp"
    sources: Mapping[str, str] = field(default_factory=lambda: _DEFAULT_NMREXP_SOURCES)
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
    ``processed_dir``. ``queries`` pairs with :class:`NmrExpDataLoaderConfig`'s
    ``truth``: queries are meant to be clustered into seed questions downstream,
    which then get matched against ``truth`` to build an augmented dataset.

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
