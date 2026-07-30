"""Framework-agnostic, ``torch.utils.data.Dataset``-inspired container types.

Downstream stages (rollout, SFT/RL post-training, evaluation) all want the same
narrow interface -- ``len()``, integer/slice indexing, and repeatable iteration
-- regardless of whether records live in memory or are streamed off disk.
This intentionally does not depend on torch: spectune's core dependency list is
empty, and nothing here needs collation, batching, or tensors yet.
"""

from __future__ import annotations

import json
import random
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

JsonDict = dict[str, Any]
_T = TypeVar("_T")


class Dataset(ABC):
    """Minimal analogue of ``torch.utils.data.Dataset``: sized, indexable, iterable.

    Subclasses only need to implement ``__len__`` and ``__getitem__``; iteration
    and the composable helpers below (``take``, ``shuffle``, ``filter``, ...) are
    provided in terms of those two primitives.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> JsonDict: ...

    def __iter__(self) -> Iterator[JsonDict]:
        for i in range(len(self)):
            yield self[i]

    def head(self, n: int) -> list[JsonDict]:
        return [self[i] for i in range(min(n, len(self)))]

    def to_list(self) -> list[JsonDict]:
        return list(self)

    def take(self, n: int) -> Subset:
        return Subset(self, range(min(n, len(self))))

    def select(self, indices: Iterable[int]) -> Subset:
        return Subset(self, list(indices))

    def shuffle(self, seed: int | None = None) -> Subset:
        indices = list(range(len(self)))
        random.Random(seed).shuffle(indices)
        return Subset(self, indices)

    def sample(
        self,
        num: int,
        *,
        seed: int | None = None,
        cluster_key: str = "cluster",
        progress: bool = True,
    ) -> Subset:
        """Sample at most ``num`` records, balancing clusters when available.

        If every record has ``cluster_key``, the requested size is distributed
        as evenly as possible across clusters; spare capacity from small
        clusters is redistributed to larger ones. Sampling uses a two-pass
        reservoir algorithm, so memory is ``O(num + clusters)`` even for a
        multi-million-row :class:`JsonlDataset`. If any record lacks a cluster
        label, the method falls back to uniform random sampling over the whole
        dataset without scanning record contents a second time.
        """
        if num < 0:
            raise ValueError("num must be non-negative")
        total = len(self)
        target = min(num, total)
        if target == 0:
            return Subset(self, [])

        counts: dict[Any, int] = {}
        all_clustered = True
        for record in progress_iter(
            self,
            total=total,
            label="sample/scan",
            enabled=progress,
        ):
            if cluster_key not in record or record[cluster_key] is None:
                all_clustered = False
                break
            cluster_label = record[cluster_key]
            counts[cluster_label] = counts.get(cluster_label, 0) + 1

        rng = random.Random(seed)
        if not all_clustered or not counts:
            return Subset(self, rng.sample(range(total), target))

        quotas = _balanced_cluster_quotas(counts, target, rng)
        reservoirs: dict[Any, list[int]] = {lbl: [] for lbl in counts}
        seen: dict[Any, int] = {lbl: 0 for lbl in counts}
        for index, record in enumerate(
            progress_iter(
                self,
                total=total,
                label="sample/reservoir",
                enabled=progress,
            )
        ):
            cluster_label = record[cluster_key]
            quota = quotas[cluster_label]
            if quota == 0:
                continue
            seen[cluster_label] += 1
            reservoir = reservoirs[cluster_label]
            if len(reservoir) < quota:
                reservoir.append(index)
                continue
            replacement = rng.randrange(seen[cluster_label])
            if replacement < quota:
                reservoir[replacement] = index

        indices = [index for reservoir in reservoirs.values() for index in reservoir]
        rng.shuffle(indices)
        return Subset(self, indices)

    def filter(self, predicate: Callable[[JsonDict], bool]) -> InMemoryDataset:
        """Materialize the subset of records matching ``predicate`` in memory.

        Unlike ``take``/``select``/``shuffle`` (index views over the source
        dataset), the result size is unknown ahead of time, so this reads
        through the whole dataset once and builds a plain in-memory list.
        """
        return InMemoryDataset([record for record in self if predicate(record)])

    def map(self, fn: Callable[[JsonDict], JsonDict]) -> InMemoryDataset:
        """Materialize ``fn`` applied to every record in memory."""
        return InMemoryDataset([fn(record) for record in self])

    def save_jsonl(self, path: str | Path) -> int:
        return write_jsonl(path, self)


class Subset(Dataset):
    """A view over another dataset restricted/reordered to ``indices``.

    Cheap by construction: it stores only the index list, and defers all record
    access to the wrapped dataset. This backs ``take``/``select``/``shuffle`` so
    they work the same way over an in-memory list or a multi-million-row
    :class:`JsonlDataset` without copying record content.
    """

    def __init__(self, dataset: Dataset, indices: Iterable[int]) -> None:
        self._dataset = dataset
        self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self._dataset[self._indices[_normalize_index(index, len(self))]]


class InMemoryDataset(Dataset):
    """A dataset backed by a plain Python list already resident in memory."""

    def __init__(self, records: Iterable[JsonDict]) -> None:
        self._records = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return self._records[index]
        return self._records[_normalize_index(index, len(self._records))]


class JsonlDataset(Dataset):
    """A dataset backed by a JSON-Lines file, indexed by byte offset.

    Only a list of per-line byte offsets is kept in memory, so this scales to
    multi-million-row files (e.g. the raw NMRexp training split) without ever
    materializing every record at once. Each access reopens the file, so
    concurrent readers and repeated epochs over the same instance are both safe.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offsets = _index_jsonl_offsets(self.path)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = _normalize_index(index, len(self))
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            line = handle.readline()
        return json.loads(line)

    def __iter__(self) -> Iterator[JsonDict]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)


def _normalize_index(index: int, length: int) -> int:
    if index < 0:
        index += length
    if not (0 <= index < length):
        raise IndexError(index)
    return index


def _balanced_cluster_quotas(counts: dict[Any, int], target: int, rng: random.Random) -> dict[Any, int]:
    labels = list(counts)
    rng.shuffle(labels)
    quotas = {label: min(counts[label], target // len(labels)) for label in labels}
    remaining = target - sum(quotas.values())
    while remaining:
        eligible = [label for label in labels if quotas[label] < counts[label]]
        if not eligible:
            break
        rng.shuffle(eligible)
        for label in eligible:
            quotas[label] += 1
            remaining -= 1
            if remaining == 0:
                break
    return quotas


def _index_jsonl_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    offset = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(offset)
            offset += len(line)
    return offsets


def write_jsonl(path: str | Path, records: Iterable[JsonDict]) -> int:
    """Write ``records`` to ``path`` as JSON-Lines, returning the row count written."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[JsonDict]:
    """Fully materialize a JSON-Lines file into a list; convenience for small files/tests."""
    records: list[JsonDict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def has_pandas() -> bool:
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False
    return True


def progress_iter(
    iterable: Iterable[_T],
    *,
    total: int | None = None,
    label: str = "",
    enabled: bool = True,
    min_interval: float = 0.5,
) -> Iterator[_T]:
    """Wrap ``iterable``, printing periodic progress to stderr as items are consumed.

    A dependency-free stand-in for ``tqdm`` (spectune's core dependency list is
    empty): prints ``processed/total (pct%) rate elapsed eta`` at most once per
    ``min_interval`` seconds, plus a final line, so long-running preprocessing
    (e.g. NMRexp's ~1M-row training parquet) stays visible instead of silently
    running for minutes. Pass ``enabled=False`` to disable it entirely, e.g.
    from a config flag or to keep test output quiet.
    """
    if not enabled:
        yield from iterable
        return

    start = time.monotonic()
    last_print = 0.0
    count = 0
    try:
        for item in iterable:
            count += 1
            now = time.monotonic()
            if now - last_print >= min_interval:
                _print_progress(label, count, total, start, now)
                last_print = now
            yield item
    finally:
        _print_progress(label, count, total, start, time.monotonic())
        sys.stderr.write("\n")
        sys.stderr.flush()


def _print_progress(label: str, count: int, total: int | None, start: float, now: float) -> None:
    elapsed = max(now - start, 1e-9)
    rate = count / elapsed
    prefix = f"[{label}] " if label else ""
    if total:
        pct = min(count / total * 100.0, 100.0)
        eta = (total - count) / rate if rate > 0 else 0.0
        line = f"{prefix}{count}/{total} ({pct:5.1f}%) {rate:,.0f} rows/s elapsed={elapsed:,.0f}s eta={eta:,.0f}s"
    else:
        line = f"{prefix}{count} rows {rate:,.0f} rows/s elapsed={elapsed:,.0f}s"
    sys.stderr.write("\r" + line + " " * 4)
    sys.stderr.flush()


__all__ = [
    "Dataset",
    "InMemoryDataset",
    "JsonlDataset",
    "Subset",
    "has_pandas",
    "progress_iter",
    "read_jsonl",
    "write_jsonl",
]
