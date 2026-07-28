"""Dataset loading and preprocessing for spectune.

Every data source gets a two-stage loader: a ``preprocess()`` step that reads
raw exports and caches normalized JSON-Lines files, and a ``load()`` step that
returns an iterable/indexable :class:`~spectune.dataloader.base.Dataset` over
those caches. SpecXMaster produces one query pool; NMRexp preserves separate
train/test truth pools so each can later be combined independently with seeds
sampled from those queries:

- :mod:`spectune.dataloader.nmrexp` -- a labeled dataset (ground-truth SMILES
  paired with NMR evidence), split into ``truth_train`` and ``truth_test``.
- :mod:`spectune.dataloader.specxmaster` -- raw unlabeled production traffic
  (real user conversations), merged into one ``queries`` pool.
"""

from .base import Dataset, InMemoryDataset, JsonlDataset, Subset, write_jsonl
from .config import NmrExpDataLoaderConfig, SpecXMasterDataLoaderConfig
from .nmrexp import NmrExpDataLoader
from .specxmaster import SpecXMasterDataLoader

__all__ = [
    "Dataset",
    "InMemoryDataset",
    "JsonlDataset",
    "NmrExpDataLoader",
    "NmrExpDataLoaderConfig",
    "SpecXMasterDataLoader",
    "SpecXMasterDataLoaderConfig",
    "Subset",
    "write_jsonl",
]
