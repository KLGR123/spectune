"""Loader for SpecXMaster's raw online conversation logs.

SpecXMaster is fundamentally different from NMRexp: it is not a labeled
dataset (no ground-truth structure to train or evaluate against) but raw
production traffic -- zipped exports where each folder holds one real
conversation between a user and the interpretation agent, as a
``conversation.json`` (plus a human-readable ``conversation.txt`` this loader
ignores). Each conversation is one JSON object with keys ``id``, ``user_id``,
``title``, ``modality``, ``agent``, ``created_at``, and ``messages`` (a list of
``{role, content: [...]}`` turns); every observed ``modality`` so far is
``"nmr"``, but it is read from the data rather than assumed.

This loader keeps only what a query pool needs: the free-text of every
user-authored turn (a conversation may have one or several, e.g. a spectrum
submission followed by clarifying replies) plus the conversation's
``modality``. The result is a flat collection of records called ``queries`` --
meant to be clustered/deduplicated downstream into a smaller set of
representative "seed" questions, which then get matched against
:class:`~spectune.dataloader.nmrexp.NmrExpDataLoader`'s train/test truth splits
to build the final train/test datasets. Both the clustering and the
augmentation are out of scope for this loader.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .base import JsonDict, JsonlDataset, progress_iter
from .config import SpecXMasterDataLoaderConfig


class SpecXMasterDataLoader:
    """Loads SpecXMaster's zipped conversation logs into a flat ``queries`` pool.

    1. :meth:`preprocess` unzips every archive matching ``config.zip_glob``
       under ``config.raw_dir`` into a scratch directory, reads every
       ``conversation.json`` found inside, and emits one record per
       user-authored message. The same conversation is sometimes exported into
       more than one zip shard, so conversations are deduplicated by their
       ``id`` (first occurrence, in zip-name order, wins). Output is cached as
       a single JSON-Lines file under ``config.processed_dir``.
    2. :meth:`load` (aliased :meth:`load_queries`) returns a
       :class:`~spectune.dataloader.base.JsonlDataset` over that cache --
       sized, indexable, repeatably iterable -- building it on first use if
       the cache is missing.

    Unlike :class:`~spectune.dataloader.nmrexp.NmrExpDataLoader`, there is no
    ``sources``/split concept: SpecXMaster is one undifferentiated pool of real
    user queries, not a train/test-labeled dataset. These queries are clustered
    into seeds later, then sampled separately against NMRexp's train/test truth
    to build the final datasets.
    """

    def __init__(self, config: SpecXMasterDataLoaderConfig | None = None) -> None:
        self.config = config or SpecXMasterDataLoaderConfig()

    def preprocess(self, *, overwrite: bool = False) -> JsonDict:
        """Build (or reuse) the cached ``queries`` JSON-Lines file.

        Returns a summary dict describing whether the cache was reused
        (``status: "cached"``) or rebuilt (``status: "built"``, with
        conversation/query counts).
        """
        config = self.config
        out_path = self._processed_path()
        if out_path.exists() and not overwrite:
            return {"status": "cached", "output_path": str(out_path)}

        raw_dir = Path(config.raw_dir)
        zip_paths = sorted(raw_dir.glob(config.zip_glob))
        if not zip_paths:
            raise FileNotFoundError(
                f"no SpecXMaster archives found under {raw_dir} (glob {config.zip_glob!r}); "
                "set SPECXMASTER_RAW_DIR or pass a SpecXMasterDataLoaderConfig(raw_dir=...)"
            )

        stats: JsonDict = {
            "zip_files": len(zip_paths),
            "conversations_seen": 0,
            "conversations_duplicate": 0,
            "conversations_kept": 0,
            "conversations_unparseable": 0,
            "queries": 0,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        seen_ids: set[Any] = set()
        zips = progress_iter(zip_paths, total=len(zip_paths), label="SpecXMaster/unzip", enabled=config.show_progress)
        with tempfile.TemporaryDirectory(prefix="spectune_specxmaster_") as scratch_dir:
            with tmp_path.open("w", encoding="utf-8") as handle:
                for zip_path in zips:
                    extract_dir = Path(scratch_dir) / zip_path.stem
                    with zipfile.ZipFile(zip_path) as archive:
                        archive.extractall(extract_dir)
                    for convo_path in sorted(extract_dir.rglob("conversation.json")):
                        conversation = _load_conversation(convo_path)
                        if conversation is None:
                            stats["conversations_unparseable"] += 1
                            continue
                        stats["conversations_seen"] += 1
                        conv_id = conversation.get("id")
                        if conv_id is not None and conv_id in seen_ids:
                            stats["conversations_duplicate"] += 1
                            continue
                        seen_ids.add(conv_id)
                        stats["conversations_kept"] += 1
                        for record in _extract_queries(conversation, source_zip=zip_path.name, config=config):
                            handle.write(json.dumps(record, ensure_ascii=False))
                            handle.write("\n")
                            stats["queries"] += 1
        tmp_path.replace(out_path)
        return {"status": "built", "output_path": str(out_path), **stats}

    def load(self, *, force_reprocess: bool = False) -> JsonlDataset:
        """Return the processed ``queries`` :class:`JsonlDataset`, building it if needed."""
        out_path = self._processed_path()
        if force_reprocess or not out_path.exists():
            self.preprocess(overwrite=force_reprocess)
        return JsonlDataset(out_path)

    def load_queries(self, **kwargs: Any) -> JsonlDataset:
        """Alias for :meth:`load`, named after the dataset it returns."""
        return self.load(**kwargs)

    def _processed_path(self) -> Path:
        return Path(self.config.processed_dir) / f"{self.config.dataset_name.lower()}_queries.jsonl"


def _load_conversation(path: Path) -> JsonDict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _extract_queries(
    conversation: JsonDict,
    *,
    source_zip: str,
    config: SpecXMasterDataLoaderConfig,
) -> Iterator[JsonDict]:
    conv_id = conversation.get("id")
    modality = conversation.get("modality") or None
    created_at = conversation.get("created_at")
    user_turns = [message for message in conversation.get("messages") or [] if message.get("role") == "user"]
    num_turns = len(user_turns)
    for turn_index, message in enumerate(user_turns):
        text = _message_text(message)
        if not text:
            continue
        yield {
            "sample_id": f"{config.dataset_name}:{conv_id}:{turn_index}",
            "query": text,
            "modality": modality,
            "provenance": {
                "dataset": config.dataset_name,
                "conversation_id": conv_id,
                "user_id": conversation.get("user_id"),
                "turn_index": turn_index,
                "num_turns": num_turns,
                "msg_id": message.get("msg_id"),
                "chat_id": message.get("chat_id"),
                "timestamp": message.get("timestamp") or created_at,
                "source_zip": source_zip,
            },
        }


def _message_text(message: JsonDict) -> str:
    """Join every ``text``-typed content block of a message into one string.

    User turns in practice carry exactly one ``text`` block each, but this
    joins multiple defensively rather than silently dropping content if that
    ever changes.
    """
    parts = [
        str(item.get("text") or "").strip()
        for item in message.get("content") or []
        if item.get("type") == "text" and str(item.get("text") or "").strip()
    ]
    return "\n".join(parts)


__all__ = ["SpecXMasterDataLoader"]
