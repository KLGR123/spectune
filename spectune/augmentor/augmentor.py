"""Query construction: turn enriched truth rows into user-style training samples.

The scenario grid the augmentor covers is a product of two independent draws
per row. ``information_mix`` picks *what* extra knowledge the user has beyond
the spectrum (nothing, a molecular formula, a hedged structure guess, reaction
background, or a fragment hint), and ``followup_probability`` picks *when* they
say it (first message, or a second message after the spectrum). Two more knobs
control presentation rather than content: ``raw_query_ratio`` leaves a share of
rows as plain template text instead of LLM-rewritten prose, and
``nmr_noise_ratio`` corrupts a share of the spectra on purpose.
``formula_noise_ratio`` similarly replaces a share of formula conditions with
a nearby composition.

Every draw, fallback, and rejection is written back onto the output record, so
downstream analysis can slice the dataset by scenario without re-deriving
anything. In particular, when a molecule turns out not to *have* the drawn
information (no published name, no reaction precedent), the row degrades to a
spectrum-only query and says so via ``information_available: false`` rather
than inventing the fact.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from spectune.dataloader import Dataset, InMemoryDataset, NmrExpDataLoader, NmrExpDataLoaderConfig
from spectune.dataloader.base import write_jsonl
from spectune.llm import LlmClient

from .config import AugmentorConfig
from .enrichment import Enricher
from .formulas import perturb_formula
from .prompts import (
    FOLLOWUP_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    render_first_turn,
    render_followup,
    render_formula_block,
    render_fragment_block,
    render_polymer_block,
    render_reaction_block,
    render_structure_block,
)
from .reactions import describe_agents
from .spectra import apply_nmr_noise, build_multimodal_spectrum_text, build_spectrum_text

JsonDict = dict[str, Any]


class Augmentor:
    """Sample truth data, enrich it, and construct annotated multi-turn queries."""

    def __init__(
        self,
        config: AugmentorConfig | None = None,
        *,
        enricher: Enricher | None = None,
        llm: LlmClient | None = None,
        loader: NmrExpDataLoader | None = None,
    ) -> None:
        self.config = config or AugmentorConfig()
        self.llm = llm or LlmClient(self.config.llm)
        self.enricher = enricher or Enricher(self.config.enrichment)
        self.loader = loader or NmrExpDataLoader(
            NmrExpDataLoaderConfig(
                processed_dir=self.config.datasets_dir,
                show_progress=self.config.show_progress,
            )
        )
        self.last_summary: JsonDict | None = None

    def load_truth(self, split: str | None = None) -> Dataset:
        """Load one clustered truth split (``train`` or ``test``)."""
        if self.config.dataset_name.lower() != "nmrexp":
            raise ValueError(f"unsupported dataset {self.config.dataset_name!r}; only NMRexp truth is wired up so far")
        return self.loader.load_truth(split or self.config.split)

    def sample(self, dataset: Dataset | None = None, *, size: int | None = None) -> Dataset:
        """Draw ``size`` rows (0/None means the whole split), balanced across clusters."""
        dataset = dataset if dataset is not None else self.load_truth()
        target = self.config.sample_size if size is None else size
        if not target:
            return dataset
        return dataset.sample(target, seed=self.config.seed, cluster_key=self.config.cluster_key)

    def build(
        self,
        dataset: Dataset | Sequence[JsonDict] | None = None,
        *,
        output_path: str | Path | None = None,
    ) -> InMemoryDataset:
        """Synchronous wrapper around :meth:`build_async`."""
        return asyncio.run(self.build_async(dataset, output_path=output_path))

    async def build_async(
        self,
        dataset: Dataset | Sequence[JsonDict] | None = None,
        *,
        output_path: str | Path | None = None,
    ) -> InMemoryDataset:
        started = time.monotonic()
        base = InMemoryDataset(dataset) if isinstance(dataset, Sequence) else dataset
        source = self.sample(base)
        enriched = await self.enricher.enrich_async(source)

        plans = [self._plan(record, index) for index, record in enumerate(enriched)]
        rewrites = await self._rewrite(plans)
        records = [self._assemble(plan, rewrite) for plan, rewrite in zip(plans, rewrites, strict=True)]

        path = Path(output_path) if output_path else self._default_output_path()
        written = write_jsonl(path, records) if path else 0
        self.last_summary = {
            "records": len(records),
            "output_path": str(path) if path else None,
            "written": written,
            "split": self.config.split,
            "information_types": _counter(record["augmentation"]["information_type"] for record in records),
            "requested_information_types": _counter(
                record["augmentation"]["requested_information_type"] for record in records
            ),
            "placements": _counter(record["augmentation"]["information_placement"] for record in records),
            "information_degraded": sum(1 for record in records if record["augmentation"]["information_degraded"]),
            "llm_rewritten": sum(1 for record in records if record["augmentation"]["llm_rewritten"]),
            "llm_rejected": sum(
                1 for record in records if record["augmentation"]["llm_status"] == "rejected_information_loss"
            ),
            "noised": _counter(
                record["augmentation"]["nmr_noise_mode"]
                for record in records
                if record["augmentation"]["nmr_noise_applied"]
            ),
            "formula_noise_eligible": sum(
                1 for record in records if record["augmentation"]["information_type"] == "formula"
            ),
            "formula_noised": sum(1 for record in records if record["augmentation"]["formula_noise_applied"]),
            "multi_turn": sum(1 for record in records if record["num_of_queries"] > 1),
            "modal_dropped": sum(1 for record in records if record["augmentation"]["modal_drop_applied"]),
            "enrichment": self.enricher.last_summary,
            "llm": dict(self.llm.stats),
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        return InMemoryDataset(records)

    def build_splits(
        self,
        dataset: Dataset | Sequence[JsonDict] | None = None,
        *,
        sizes: tuple[int, int, int] = (20000, 200, 2000),
        output_paths: dict[str, Path],
    ) -> dict[str, InMemoryDataset]:
        """Synchronous wrapper around :meth:`build_splits_async`."""
        return asyncio.run(self.build_splits_async(dataset, sizes=sizes, output_paths=output_paths))

    async def build_splits_async(
        self,
        dataset: Dataset | Sequence[JsonDict] | None = None,
        *,
        sizes: tuple[int, int, int] = (20000, 200, 2000),
        output_paths: dict[str, Path],
    ) -> dict[str, InMemoryDataset]:
        """Build three disjoint splits (train/test/sft) from one shuffle of the truth pool.

        Disjointness is guaranteed by construction: the pool is shuffled once with the
        configured seed, then sliced into three contiguous windows.  Enrichment runs over
        the full union in a single pass so the SMILES cache is warmed only once.
        """
        base = InMemoryDataset(dataset) if isinstance(dataset, Sequence) else dataset
        if base is None:
            base = self.load_truth()

        n_train, n_test, n_sft = sizes
        total_needed = n_train + n_test + n_sft
        if len(base) < total_needed:
            raise ValueError(
                f"truth pool has {len(base)} records but {total_needed} are needed "
                f"({n_train}+{n_test}+{n_sft}); reduce --sizes or use a larger split"
            )

        shuffled = list(base.shuffle(seed=self.config.seed).take(total_needed))
        enriched = list(await self.enricher.enrich_async(shuffled))

        split_defs = [
            ("train", enriched[:n_train]),
            ("test", enriched[n_train : n_train + n_test]),
            ("sft", enriched[n_train + n_test :]),
        ]

        results: dict[str, InMemoryDataset] = {}
        for name, raw in split_defs:
            plans = [self._plan(r, i) for i, r in enumerate(raw)]
            rewrites = await self._rewrite(plans)
            records = [self._assemble(p, rw) for p, rw in zip(plans, rewrites, strict=True)]
            write_jsonl(output_paths[name], records)
            results[name] = InMemoryDataset(records)

        return results

    # Scenario planning

    def _plan(self, record: JsonDict, index: int) -> JsonDict:
        config = self.config
        rng = random.Random(f"{config.seed}:{record.get('sample_id') or index}")
        enrichment = record.get("enrichment") or {}

        # A merged record carries every modality in ``nmr_list``; a plain
        # record's single ``nmr`` block is treated as a one-element list so
        # both shapes flow through the same multi-modal path below.
        nmr_list = record.get("nmr_list")
        if not nmr_list and record.get("nmr"):
            nmr_list = [record["nmr"]]
        nmr_list = [nmr for nmr in (nmr_list or []) if nmr]

        # Randomly drop some modalities of multi-spectrum rows, keeping at
        # least one, so the model also sees partial-evidence queries.
        dropped_modalities: list[int] = []
        if len(nmr_list) > 1 and config.modal_drop_ratio and rng.random() < config.modal_drop_ratio:
            keep_count = rng.randint(1, len(nmr_list) - 1)
            kept_indices = sorted(rng.sample(range(len(nmr_list)), keep_count))
            dropped_modalities = [i for i in range(len(nmr_list)) if i not in kept_indices]
            nmr_list = [nmr_list[i] for i in kept_indices]

        clean_text = build_multimodal_spectrum_text(nmr_list, rng)
        spectrum_text = clean_text
        noise: JsonDict = {"mode": None, "applied": False, "changed": False}
        if spectrum_text and config.nmr_noise_ratio and rng.random() < config.nmr_noise_ratio:
            mode = rng.choice(list(config.nmr_noise_modes))
            noised_parts: list[str] = []
            noise_details: list[JsonDict] = []
            for nmr in nmr_list:
                noised_text, detail = apply_nmr_noise(
                    nmr,
                    mode,
                    rng,
                    strength=config.nmr_noise_strength,
                )
                noised_parts.append(noised_text or build_spectrum_text(nmr))
                noise_details.append(detail)
            noised_nmr_proxy = [
                {"shift_text": part, "type": (nmr or {}).get("type")}
                for nmr, part in zip(nmr_list, noised_parts, strict=True)
            ]
            spectrum_text = build_multimodal_spectrum_text(noised_nmr_proxy, rng)
            noise = {"mode": mode, "applied": True, "per_spectrum": noise_details}
            # A row can be selected for noising and still come out identical
            # (every peak survived a random drop); analysis wants both facts.
            noise["changed"] = spectrum_text != clean_text

        information_type = _weighted_choice(config.information_mix, rng)
        uncertain = information_type == "structure" or rng.random() < config.uncertain_tone_probability
        text, detail = self._information_text(information_type, record, enrichment, rng, uncertain=uncertain)
        available = bool(text)
        formula_noise = detail.get("formula_noise") or {"applied": False, "changed": False}
        if not available:
            information_type_effective = "none"
            placement = "none"
        else:
            information_type_effective = information_type
            placement = "followup" if rng.random() < config.followup_probability else "first_turn"

        return {
            "record": record,
            "rng": rng,
            "nmr_list": nmr_list,
            "dropped_modalities": dropped_modalities,
            "spectrum_text": spectrum_text,
            "noise": noise,
            "formula_noise": formula_noise,
            "requested_information_type": information_type,
            "information_type": information_type_effective,
            "information_available": available,
            # Spectrum-only rows are drawn on purpose; a *degraded* row wanted
            # information the molecule turned out not to have.
            "information_degraded": information_type != "none" and not available,
            "information_text": text,
            "information_detail": detail,
            "information_placement": placement,
            "uncertain": uncertain and available,
            "llm_rewrite": config.use_llm_rewrite and self.llm.available and rng.random() >= config.raw_query_ratio,
        }

    def _information_text(
        self,
        information_type: str,
        record: JsonDict,
        enrichment: JsonDict,
        rng: random.Random,
        *,
        uncertain: bool,
    ) -> tuple[str, JsonDict]:
        if information_type == "none":
            return "", {}
        if information_type == "formula":
            formula = record.get("molecular_formula") or (enrichment.get("properties") or {}).get("molecular_formula")
            if not formula:
                return "", {}
            original_formula = str(formula)
            presented_formula = original_formula
            formula_noise: JsonDict = {
                "applied": False,
                "changed": False,
                "original_formula": original_formula,
                "perturbed_formula": original_formula,
            }
            if self.config.formula_noise_ratio and rng.random() < self.config.formula_noise_ratio:
                presented_formula, formula_noise = perturb_formula(original_formula, rng)
            return render_formula_block(presented_formula), {
                "formula": presented_formula,
                "formula_original": original_formula,
                "formula_noise": formula_noise,
            }
        if information_type == "structure":
            return self._structure_text(record, enrichment, rng)
        if information_type == "reaction":
            return self._reaction_text(enrichment, rng, uncertain=uncertain)
        if information_type == "fragment":
            return self._fragment_text(enrichment, rng, uncertain=uncertain)
        raise ValueError(f"unknown information type {information_type!r}")

    def _structure_text(
        self,
        record: JsonDict,
        enrichment: JsonDict,
        rng: random.Random,
    ) -> tuple[str, JsonDict]:
        names = enrichment.get("names") or {}
        options: list[tuple[str, str]] = []
        smiles = str(record.get("gt_smiles") or "")
        if smiles:
            options.append(("smiles", smiles))
        names_complete = bool(names.get("complete") and names.get("name_zh") and names.get("name_en"))
        if names_complete:
            options.append(("name_zh", str(names["name_zh"])))
            options.append(("name_en", str(names["name_en"])))
        if not options:
            return "", {}
        kind, value = rng.choice(options)
        return render_structure_block(value, kind, rng), {"structure_kind": kind, "structure_value": value}

    def _reaction_text(
        self,
        enrichment: JsonDict,
        rng: random.Random,
        *,
        uncertain: bool,
    ) -> tuple[str, JsonDict]:
        reaction = enrichment.get("reaction") or {}
        items: list[JsonDict] = []
        # Retrieved precedents are real chemistry for this exact product, so they
        # are quoted first and stated with confidence; template routes are only
        # proposals and are phrased as such.
        for precedent in (reaction.get("precedents") or [])[: self.config.max_reaction_items]:
            items.append(
                {
                    "reactants": precedent.get("reactants") or [],
                    "conditions": "、".join(describe_agents(precedent.get("agents") or [])),
                    "name": precedent.get("reaction_class") or "",
                    "origin": "precedent",
                    "reference": precedent.get("reference") or "",
                }
            )
        if not items:
            routes = sorted(
                reaction.get("routes") or [],
                key=lambda route: bool((route.get("askcos") or {}).get("verified")),
                reverse=True,
            )
            # Two disconnections of the same kind read as repetition rather than
            # as extra evidence, so quote at most one route per template.
            seen_templates: set[str] = set()
            routes = [
                route
                for route in routes
                if not (route.get("template") in seen_templates or seen_templates.add(str(route.get("template"))))
            ]
            for route in routes[: self.config.max_reaction_items]:
                items.append(
                    {
                        "reactants": route.get("reactants") or [],
                        "conditions": route.get("conditions_zh") or "",
                        "name": route.get("zh") or "",
                        "origin": "retro_template",
                        "askcos_verified": bool((route.get("askcos") or {}).get("verified")),
                    }
                )
        proposed = bool(items) and items[0]["origin"] == "retro_template"
        text = render_reaction_block(items, rng, uncertain=uncertain or proposed)
        polymer = reaction.get("polymer") or {}
        if text and polymer.get("monomer_classes") and rng.random() < 0.5:
            polymer_text = render_polymer_block(polymer["monomer_classes"])
            if polymer_text:
                text = f"{text}。{polymer_text}"
        return text, {"reaction_items": items, "reaction_origin": items[0]["origin"] if items else None}

    def _fragment_text(
        self,
        enrichment: JsonDict,
        rng: random.Random,
        *,
        uncertain: bool,
    ) -> tuple[str, JsonDict]:
        candidates = [
            fragment
            for fragment in enrichment.get("fragments") or []
            if fragment.get("key") != "murcko_scaffold" and fragment.get("zh")
        ]
        if not candidates:
            return "", {}
        count = min(len(candidates), rng.randint(1, max(self.config.max_fragment_items, 1)))
        chosen = candidates[:count]
        return render_fragment_block(chosen, rng, uncertain=uncertain), {
            "fragments": [fragment["key"] for fragment in chosen]
        }

    # Rendering

    async def _rewrite(self, plans: Sequence[JsonDict]) -> list[JsonDict]:
        prompts: list[tuple[str, str]] = []
        routing: list[tuple[int, str]] = []
        for index, plan in enumerate(plans):
            if not plan["llm_rewrite"]:
                continue
            first_block, followup_block = _information_blocks(plan)
            prompts.append((QUERY_REWRITE_SYSTEM_PROMPT, first_block))
            routing.append((index, "first"))
            if followup_block:
                prompts.append((FOLLOWUP_REWRITE_SYSTEM_PROMPT, followup_block))
                routing.append((index, "followup"))

        completions = await self.llm.complete_many(prompts)
        rewrites: list[JsonDict] = [{"first": None, "followup": None} for _ in plans]
        for (index, slot), completion in zip(routing, completions, strict=True):
            rewrites[index][slot] = completion.strip() if completion else None
        return rewrites

    def _assemble(self, plan: JsonDict, rewrite: JsonDict) -> JsonDict:
        record = plan["record"]
        rng: random.Random = plan["rng"]
        first_block, followup_block = _information_blocks(plan)

        first_template = render_first_turn(
            plan["spectrum_text"],
            plan["information_text"] if plan["information_placement"] == "first_turn" else "",
            rng,
        )
        followup_template = render_followup(plan["information_text"], rng) if followup_block else ""

        must_keep_first = _required_literals(plan, include_spectrum=True)
        must_keep_followup = _required_literals(plan, include_spectrum=False)
        first_turn, first_status = _select_text(rewrite["first"], first_template, must_keep_first)
        followup, followup_status = (
            _select_text(rewrite["followup"], followup_template, must_keep_followup)
            if followup_block
            else ("", "not_applicable")
        )

        turns = [{"turn_index": 0, "role": "user", "content": first_turn}]
        if followup:
            turns.append({"turn_index": 1, "role": "user", "content": followup})

        llm_status = first_status if followup_status in ("not_applicable", first_status) else "mixed"
        record_nmr_list = record.get("nmr_list") or ([record.get("nmr")] if record.get("nmr") else None)
        augmentation = {
            "requested_information_type": plan["requested_information_type"],
            "information_type": plan["information_type"],
            "information_placement": plan["information_placement"],
            "information_available": plan["information_available"],
            "information_degraded": plan["information_degraded"],
            "information_text": plan["information_text"],
            "information_detail": plan["information_detail"],
            "uncertain_tone": plan["uncertain"],
            "llm_requested": plan["llm_rewrite"],
            "llm_rewritten": llm_status == "llm",
            "llm_status": llm_status,
            "nmr_noise_applied": bool(plan["noise"].get("applied")),
            "nmr_noise_mode": plan["noise"].get("mode"),
            "nmr_noise_detail": plan["noise"],
            "formula_noise_applied": bool(plan["formula_noise"].get("applied")),
            "formula_noise_detail": plan["formula_noise"],
            "reaction_noise_applied": False,
            "num_active_modalities": len(plan["nmr_list"]),
            "dropped_modalities": plan["dropped_modalities"],
            "modal_drop_applied": bool(plan["dropped_modalities"]),
            "has_name_zh": bool(((record.get("enrichment") or {}).get("names") or {}).get("name_zh")),
            "has_name_en": bool(((record.get("enrichment") or {}).get("names") or {}).get("name_en")),
            "has_reaction_precedent": bool(((record.get("enrichment") or {}).get("reaction") or {}).get("precedents")),
            "num_retro_routes": len(((record.get("enrichment") or {}).get("reaction") or {}).get("routes") or []),
            "askcos_verified": any(
                (route.get("askcos") or {}).get("verified")
                for route in ((record.get("enrichment") or {}).get("reaction") or {}).get("routes") or []
            ),
            "has_polymer_context": bool(
                (((record.get("enrichment") or {}).get("reaction") or {}).get("polymer") or {}).get("monomer_classes")
            ),
            "num_fragments": len((record.get("enrichment") or {}).get("fragments") or []),
            "seed": self.config.seed,
        }
        return {
            "sample_id": f"{record.get('sample_id')}:aug",
            "source_sample_id": record.get("sample_id"),
            "modality": record.get("modality", "nmr"),
            "cluster": record.get(self.config.cluster_key),
            "turns": turns,
            "num_of_queries": len(turns),
            "gt_smiles": record.get("gt_smiles"),
            "molecular_formula": record.get("molecular_formula"),
            "molecular_weight": ((record.get("enrichment") or {}).get("properties") or {}).get("molecular_weight"),
            "nmr": record.get("nmr"),
            "nmr_list": record_nmr_list,
            "active_nmr_list": plan["nmr_list"],
            "ms": record.get("ms"),
            "nmr_text": plan["spectrum_text"],
            "nmr_text_clean": build_multimodal_spectrum_text(plan["nmr_list"]),
            "augmentation": augmentation,
            "enrichment": record.get("enrichment"),
            "provenance": {**(record.get("provenance") or {}), "augmented_from": record.get("sample_id")},
            "quality": record.get("quality"),
        }

    def _default_output_path(self) -> Path:
        name = f"{self.config.dataset_name.lower()}_augmented_{self.config.split}.jsonl"
        return Path(self.config.datasets_dir) / name


def _information_blocks(plan: JsonDict) -> tuple[str, str]:
    """Return the raw fact blocks handed to the model for each turn."""
    information = plan["information_text"]
    if plan["information_placement"] == "first_turn" and information:
        return f"{plan['spectrum_text']}\n{information}", ""
    if plan["information_placement"] == "followup" and information:
        return plan["spectrum_text"], information
    return plan["spectrum_text"], ""


def _required_literals(plan: JsonDict, *, include_spectrum: bool) -> list[str]:
    """Substrings an LLM rewrite must preserve verbatim to be accepted.

    Only load-bearing values are checked -- the spectrum itself, a formula, a
    SMILES, a fragment or reaction name -- because the model is expected to
    rephrase the connective language around them.
    """
    literals: list[str] = []
    if include_spectrum and plan["spectrum_text"]:
        literals.append(plan["spectrum_text"])
    detail = plan["information_detail"]
    placement_matches = (plan["information_placement"] == "first_turn") == include_spectrum
    if plan["information_available"] and placement_matches:
        if detail.get("formula"):
            literals.append(str(detail["formula"]))
        if detail.get("structure_value"):
            literals.append(str(detail["structure_value"]))
        for item in detail.get("reaction_items") or []:
            literals.extend(item.get("reactants") or [])
    return literals


def _select_text(candidate: str | None, fallback: str, must_keep: Sequence[str]) -> tuple[str, str]:
    """Prefer the model rewrite, but only when every required literal survived."""
    if not candidate:
        return fallback, "template"
    if any(literal and literal not in candidate for literal in must_keep):
        return fallback, "rejected_information_loss"
    return candidate, "llm"


def _weighted_choice(weights: dict[str, float] | Any, rng: random.Random) -> str:
    items = [(name, float(weight)) for name, weight in weights.items() if weight > 0]
    total = sum(weight for _, weight in items)
    threshold = rng.random() * total
    cumulative = 0.0
    for name, weight in items:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return items[-1][0]


def _counter(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


__all__ = ["Augmentor"]
