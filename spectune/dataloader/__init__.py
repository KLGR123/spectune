"""Dataset loading and preprocessing for spectune.

Every data source gets a two-stage loader: a ``preprocess()`` step that reads
raw exports and caches a normalized, merged JSON-Lines file, and a ``load()``
step that returns an iterable/indexable
:class:`~spectune.dataloader.base.Dataset` over that cache. Neither loader
produces train/test splits at this stage -- that happens downstream, after
:mod:`spectune.dataloader.specxmaster`'s ``queries`` are clustered into seed
questions and matched against :mod:`spectune.dataloader.nmrexp`'s ``truth`` to
build an augmented, trainable dataset:

- :mod:`spectune.dataloader.nmrexp` -- a labeled dataset (ground-truth SMILES
  paired with NMR evidence), merged into one ``truth`` pool.
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
