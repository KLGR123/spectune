"""Fact-gathering pass that runs before any query is constructed.

:class:`Enricher` turns a truth row (SMILES + spectrum) into the same row plus
an ``enrichment`` block holding everything the query builder might want to
mention: molecular properties, Chinese/English names, structural fragments, and
reaction context. It is a separate submodule because it is the expensive,
network- and corpus-bound half of augmentation, while query construction itself
is pure local sampling.

Two properties keep it usable on real dataset sizes:

- **Batching by cost class.** The local reaction corpora are scanned exactly
  once per run for *all* molecules at once (a per-molecule scan of a 6.9 GB
  file would be absurd), while per-molecule work that is network-bound (name
  search, ASKCOS verification) runs concurrently under its own limit.
- **Caching by canonical SMILES.** Results are appended to a JSON-Lines cache,
  so re-running with a larger sample, or after changing only the construction
  knobs, re-does no lookups. An entry that covers only some of the requested
  stages is kept and topped up, so enabling name lookup later does not re-run
  the reaction search.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from spectune.dataloader import Dataset, InMemoryDataset
from spectune.dataloader.base import progress_iter
from spectune.tools import AskcosReactionForwardPredictTool, ReactionLocalIndexSearchTool
from spectune.tools.utils import canonical_smiles, has_rdkit

from .config import EnrichmentConfig
from .fragments import detect_fragments, molecule_properties
from .naming import empty_names, resolve_names
from .reactions import (
    polymer_context,
    retro_routes,
    scan_product_precedents,
    search_reactant_precedents,
    verify_routes_with_askcos,
)

JsonDict = dict[str, Any]

# Which payload field each stage owns, so a partially cached entry can be
# topped up field by field.
_STAGE_FIELDS = {
    "properties": "properties",
    "names": "names",
    "fragments": "fragments",
    "reactions": "reaction",
}


class Enricher:
    """Attach names, fragments, reaction context, and properties to truth rows."""

    def __init__(
        self,
        config: EnrichmentConfig | None = None,
        *,
        askcos_tool: AskcosReactionForwardPredictTool | None = None,
        local_index_tool: ReactionLocalIndexSearchTool | None = None,
    ) -> None:
        self.config = config or EnrichmentConfig()
        self.askcos = askcos_tool or AskcosReactionForwardPredictTool(self.config.askcos)
        self.local_index = local_index_tool or ReactionLocalIndexSearchTool(self.config.reaction_index)
        self.last_summary: JsonDict | None = None

    @property
    def enabled_stages(self) -> tuple[str, ...]:
        stages = []
        if self.config.compute_properties:
            stages.append("properties")
        if self.config.resolve_names:
            stages.append("names")
        if self.config.detect_fragments:
            stages.append("fragments")
        if self.config.resolve_reactions:
            stages.append("reactions")
        return tuple(stages)

    def enrich(self, dataset: Dataset | Sequence[JsonDict]) -> InMemoryDataset:
        """Synchronous wrapper around :meth:`enrich_async`."""
        return asyncio.run(self.enrich_async(dataset))

    async def enrich_async(self, dataset: Dataset | Sequence[JsonDict]) -> InMemoryDataset:
        started = time.monotonic()
        records = [dict(record) for record in dataset]
        keys = [canonical_smiles(str(record.get("gt_smiles") or "")) for record in records]

        cache = self._load_cache()
        stages = set(self.enabled_stages)
        pending = {
            key: cache.get(key)
            for key in sorted({key for key in keys if key})
            if not (key in cache and stages <= set(cache[key].get("stages") or ()))
        }

        computed = await self._compute(pending)
        if computed:
            self._append_cache(computed)
            cache.update(computed)

        for record, key in zip(records, keys, strict=True):
            record["enrichment"] = cache.get(key) or self._empty_payload(key)

        self.last_summary = {
            "records": len(records),
            "unique_molecules": len({key for key in keys if key}),
            "cache_hits": len({key for key in keys if key}) - len(pending),
            "computed": len(computed),
            "stages": list(self.enabled_stages),
            "with_names": sum(1 for record in records if (record["enrichment"].get("names") or {}).get("complete")),
            "with_reaction_precedents": sum(
                1 for record in records if (record["enrichment"].get("reaction") or {}).get("precedents")
            ),
            "with_retro_routes": sum(
                1 for record in records if (record["enrichment"].get("reaction") or {}).get("routes")
            ),
            "with_fragments": sum(1 for record in records if record["enrichment"].get("fragments")),
            "cache_completion": _cache_completion(cache),
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        return InMemoryDataset(records)

    async def _compute(self, pending: dict[str, JsonDict | None]) -> dict[str, JsonDict]:
        """Fill in the stages each molecule is still missing, reusing the rest."""
        if not pending:
            return {}
        payloads: dict[str, JsonDict] = {}
        todo: dict[str, dict[str, JsonDict]] = {stage: {} for stage in self.enabled_stages}
        for key, cached in pending.items():
            payload = self._empty_payload(key)
            covered = set((cached or {}).get("stages") or ())
            for stage, field in _STAGE_FIELDS.items():
                if cached and stage in covered and cached.get(field) is not None:
                    payload[field] = cached[field]
            payload["stages"] = sorted(covered | set(self.enabled_stages))
            payloads[key] = payload
            for stage in self.enabled_stages:
                if stage not in covered:
                    todo[stage][key] = payload

        if todo.get("properties") or todo.get("fragments"):
            self._add_local_chemistry(todo.get("properties", {}), todo.get("fragments", {}))
        if todo.get("reactions"):
            await self._add_reactions(todo["reactions"])
        if todo.get("names"):
            await self._add_names(todo["names"])
        return payloads

    def _empty_payload(self, key: str) -> JsonDict:
        return {
            "canonical_smiles": key,
            "stages": list(self.enabled_stages),
            "properties": {},
            "names": empty_names("not_requested"),
            "fragments": [],
            "reaction": {
                "precedents": [],
                "routes": [],
                "reactant_precedents": [],
                "polymer": {"monomer_classes": [], "is_probable_monomer": False},
                "status": "not_requested",
            },
        }

    def _add_local_chemistry(
        self,
        properties: dict[str, JsonDict],
        fragments: dict[str, JsonDict],
    ) -> None:
        if not has_rdkit():
            return
        keys = sorted(set(properties) | set(fragments))
        items = progress_iter(
            keys,
            total=len(keys),
            label="Augmentor/structure analysis",
            enabled=self.config.show_progress,
        )
        for key in items:
            if key in properties:
                properties[key]["properties"] = molecule_properties(key)
            if key in fragments:
                fragments[key]["fragments"] = detect_fragments(key, max_fragments=self.config.max_fragments)

    async def _add_reactions(self, payloads: dict[str, JsonDict]) -> None:
        if not has_rdkit():
            for payload in payloads.values():
                payload["reaction"]["status"] = "rdkit_unavailable"
            return

        precedents = scan_product_precedents(
            list(payloads),
            self.config,
            progress=self.config.show_progress,
        )
        for key, payload in payloads.items():
            reaction = payload["reaction"]
            reaction["precedents"] = precedents.get(key, [])
            reaction["routes"] = retro_routes(key, max_routes=self.config.max_retro_routes)
            reaction["polymer"] = polymer_context(key)
            reaction["status"] = "ok"

        semaphore = asyncio.Semaphore(self.config.askcos_max_concurrency)

        async def finish(payload: JsonDict) -> None:
            reaction = payload["reaction"]
            async with semaphore:
                reaction["reactant_precedents"] = await search_reactant_precedents(
                    reaction["routes"],
                    self.config,
                    tool=self.local_index,
                )
            await verify_routes_with_askcos(
                payload["canonical_smiles"],
                reaction["routes"],
                self.config,
                tool=self.askcos,
            )

        if self.config.askcos_verify_top_routes or self.config.local_index_topk:
            await _gather_with_progress(
                [finish(payload) for payload in payloads.values()],
                label="Augmentor/reaction verification",
                enabled=self.config.show_progress,
            )

    async def _add_names(self, payloads: dict[str, JsonDict]) -> None:
        semaphore = asyncio.Semaphore(self.config.name_max_concurrency)

        async def resolve(payload: JsonDict) -> None:
            async with semaphore:
                payload["names"] = await resolve_names(payload["canonical_smiles"], self.config)

        await _gather_with_progress(
            [resolve(payload) for payload in payloads.values()],
            label="Augmentor/name lookup",
            enabled=self.config.show_progress,
        )

    def _cache_path(self) -> Path:
        return Path(self.config.cache_path)

    def _load_cache(self) -> dict[str, JsonDict]:
        path = self._cache_path()
        if not self.config.use_cache or not path.exists():
            return {}
        cache: dict[str, JsonDict] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                key = entry.get("canonical_smiles")
                if key:
                    cache[key] = entry
        return cache

    def _append_cache(self, payloads: dict[str, JsonDict]) -> None:
        if not self.config.use_cache:
            return
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for payload in payloads.values():
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")


async def _gather_with_progress(
    coroutines: Iterable[Any],
    *,
    label: str,
    enabled: bool,
) -> None:
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    if not tasks:
        return
    completed = progress_iter(
        asyncio.as_completed(tasks),
        total=len(tasks),
        label=label,
        enabled=enabled,
    )
    for future in completed:
        await future


def _cache_completion(cache: dict[str, JsonDict]) -> JsonDict:
    """Summarize usable enrichment values across the whole current cache."""
    payloads = list(cache.values())
    total = len(payloads)
    checks = {
        "properties": lambda payload: bool(payload.get("properties")),
        "names": lambda payload: bool((payload.get("names") or {}).get("complete")),
        "reactions": lambda payload: bool(
            (payload.get("reaction") or {}).get("precedents") or (payload.get("reaction") or {}).get("routes")
        ),
        "fragments": lambda payload: bool(payload.get("fragments")),
    }
    summary: JsonDict = {}
    for name, check in checks.items():
        completed = sum(1 for payload in payloads if check(payload))
        summary[name] = {
            "completed": completed,
            "total": total,
            "percentage": round(100.0 * completed / total, 1) if total else 0.0,
        }
    return summary


__all__ = ["Enricher"]
