"""Dataset loading and preprocessing for spectune.

Every data source gets a two-stage loader: a ``preprocess()`` step that reads
raw exports and caches normalized JSON-Lines files, and a ``load()`` step that
returns an iterable/indexable :class:`~spectune.dataloader.base.Dataset` over
those caches. NMRexp preserves separate truth pools by provenance:

- :mod:`spectune.dataloader.nmrexp` -- a labeled dataset (ground-truth SMILES
  paired with NMR evidence), split into ``truth_bulk`` (unverified) and
  ``truth_verified`` (human-checked).
"""

from .base import Dataset, InMemoryDataset, JsonlDataset, Subset, write_jsonl
from .config import NmrExpDataLoaderConfig
from .nmrexp import NmrExpDataLoader

__all__ = [
    "Dataset",
    "InMemoryDataset",
    "JsonlDataset",
    "NmrExpDataLoader",
    "NmrExpDataLoaderConfig",
    "Subset",
    "write_jsonl",
]
