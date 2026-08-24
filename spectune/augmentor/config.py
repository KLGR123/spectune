"""Configuration objects for the spectune data augmentor.

Two layers, each independently overridable:

- :class:`EnrichmentConfig` -- what extra facts
  :class:`~spectune.augmentor.enrichment.Enricher` attaches to each truth row
  before any query is built (names, fragments, reaction context, properties).
  It embeds the *tool* configs it drives, so credentials/paths stay defined in
  exactly one place (:mod:`spectune.tools.config`).
- :class:`AugmentorConfig` -- the sampling plan and the construction mixture:
  which extra information a row carries, whether that information arrives in
  the first turn or a follow-up, how often the LLM rewrite is skipped, and how
  often the NMR text or a molecular-formula condition is corrupted on purpose.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from spectune.dataloader.config import default_datasets_dir as _default_datasets_dir
from spectune.llm import LlmConfig
from spectune.tools.config import (
    AskcosReactionForwardPredictConfig,
    ReactionLocalIndexSearchConfig,
)

# Extra information a constructed sample can carry on top of the NMR text.
# "none" keeps the spectrum-only scenario inside the same mixture, so the
# shares of every scenario sum to 1 and are directly interpretable as dataset
# composition.
INFORMATION_TYPES = ("none", "formula", "structure", "reaction", "fragment")
NMR_NOISE_MODES = ("digits_only", "strip_punctuation", "perturb_values", "drop_peaks")

_DEFAULT_INFORMATION_MIX: Mapping[str, float] = MappingProxyType(
    {
        "none": 0.30,
        "formula": 0.20,
        "structure": 0.20,
        "reaction": 0.15,
        "fragment": 0.15,
    }
)


def _default_scan_workers() -> int:
    configured = os.getenv("SPECTUNE_REACTION_SCAN_WORKERS")
    if configured:
        return int(configured)
    return min(32, os.cpu_count() or 1)


@dataclass(frozen=True, slots=True)
class EnrichmentConfig:
    """What :class:`~spectune.augmentor.enrichment.Enricher` looks up per molecule.

    Every stage degrades to "absent" rather than failing: a molecule with no
    complete PubChem name pair simply has no usable names, one with no reaction precedent keeps only its
    template-derived routes, and the augmentor records which facts were
    actually available on each output row.

    Reaction context is assembled from three complementary sources, because
    measurement on this cluster showed no single one suffices (see the module
    docstring of :mod:`spectune.augmentor.reactions`): exact product lookup in
    the local corpora covers only a few percent of NMRexp molecules, while
    RDKit retro-templates propose plausible disconnections for nearly all of
    them and ASKCOS forward prediction says whether those disconnections
    actually reconstruct the target.

    Fragment names come exclusively from the curated bilingual SMARTS lexicon;
    arbitrary web results are not used as chemical labels.
    """

    cache_path: str = field(default_factory=lambda: str(Path(_default_datasets_dir()) / "enrichment.jsonl"))
    use_cache: bool = True
    show_progress: bool = True

    compute_properties: bool = True

    resolve_names: bool = True
    name_max_concurrency: int = 8
    name_max_candidates: int = 5
    pubchem_base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    pubchem_timeout_s: float = 30.0
    pubchem_max_retries: int = 2

    detect_fragments: bool = True
    max_fragments: int = 8

    resolve_reactions: bool = True
    reaction_sources: tuple[str, ...] = ("pistachio", "chempile", "uspto")
    reaction_index: ReactionLocalIndexSearchConfig = field(default_factory=ReactionLocalIndexSearchConfig)
    reaction_scan_workers: int = field(default_factory=_default_scan_workers)
    reaction_scan_max_lines: int = 0
    reaction_scan_chunk_size: int = 20_000
    max_precedents_per_molecule: int = 5
    max_retro_routes: int = 4
    local_index_topk: int = 0
    askcos: AskcosReactionForwardPredictConfig = field(default_factory=AskcosReactionForwardPredictConfig)
    askcos_verify_top_routes: int = 1
    askcos_max_concurrency: int = 4

    def __post_init__(self) -> None:
        unknown = [name for name in self.reaction_sources if name not in ("uspto", "chempile", "pistachio")]
        if unknown:
            raise ValueError(f"unknown reaction source(s) {unknown}; expected uspto/chempile/pistachio")
        non_negative = {
            "pubchem_max_retries": self.pubchem_max_retries,
            "max_fragments": self.max_fragments,
            "reaction_scan_max_lines": self.reaction_scan_max_lines,
            "max_precedents_per_molecule": self.max_precedents_per_molecule,
            "max_retro_routes": self.max_retro_routes,
            "local_index_topk": self.local_index_topk,
            "askcos_verify_top_routes": self.askcos_verify_top_routes,
        }
        invalid = [name for name, value in non_negative.items() if value < 0]
        if invalid:
            raise ValueError(f"EnrichmentConfig values must be non-negative: {invalid}")
        positive = {
            "name_max_concurrency": self.name_max_concurrency,
            "name_max_candidates": self.name_max_candidates,
            "reaction_scan_workers": self.reaction_scan_workers,
            "reaction_scan_chunk_size": self.reaction_scan_chunk_size,
            "askcos_max_concurrency": self.askcos_max_concurrency,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"EnrichmentConfig values must be positive: {invalid}")
        if self.pubchem_timeout_s <= 0:
            raise ValueError("EnrichmentConfig.pubchem_timeout_s must be positive")
        if not self.pubchem_base_url:
            raise ValueError("EnrichmentConfig.pubchem_base_url must not be empty")


@dataclass(frozen=True, slots=True)
class AugmentorConfig:
    """Sampling plan and construction mixture for :class:`~spectune.augmentor.augmentor.Augmentor`.

    ``information_mix`` is a categorical distribution over
    :data:`INFORMATION_TYPES` and must sum to 1; each sampled row draws exactly
    one extra information type from it. ``followup_probability`` then decides
    whether that information is stated up front or asked/added in a second
    user turn, so the two knobs together describe the whole scenario grid the
    dataset is meant to cover.

    ``raw_query_ratio`` is the share of rows deliberately left as plain
    template text (no LLM rewriting), and ``nmr_noise_ratio`` is the share
    whose spectrum text is corrupted; a corrupted row picks one mode from
    ``nmr_noise_modes`` at random. ``formula_noise_ratio`` applies only to
    rows whose selected condition is a molecular formula and replaces it with
    a nearby formula. ``reaction_noise_ratio`` is reserved for a later
    experiment and must stay at 0 for now.
    """

    dataset_name: str = "nmrexp"
    split: str = "bulk"
    sample_size: int = 0
    seed: int = 42
    cluster_key: str = "cluster"
    datasets_dir: str = field(default_factory=_default_datasets_dir)
    show_progress: bool = True

    information_mix: Mapping[str, float] = field(default_factory=lambda: _DEFAULT_INFORMATION_MIX)
    followup_probability: float = 0.4
    uncertain_tone_probability: float = 0.5
    max_reaction_items: int = 2
    max_fragment_items: int = 2

    use_llm_rewrite: bool = False  # when False all rows use the template; raw_query_ratio is ignored
    raw_query_ratio: float = 0.3
    nmr_noise_ratio: float = 0.0
    nmr_noise_modes: tuple[str, ...] = NMR_NOISE_MODES
    nmr_noise_strength: float = 0.3
    formula_noise_ratio: float = 0.0
    reaction_noise_ratio: float = 0.0  # TODO
    # Share of multi-modality rows (rows whose record carries more than one NMR
    # spectrum, e.g. merged 1H/13C entries) that randomly drop some modalities
    # before the query is built. At least one spectrum is always kept.
    modal_drop_ratio: float = 0.5

    llm: LlmConfig = field(default_factory=LlmConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)

    def __post_init__(self) -> None:
        unknown = [name for name in self.information_mix if name not in INFORMATION_TYPES]
        if unknown:
            raise ValueError(f"unknown information type(s) {unknown}; expected a subset of {list(INFORMATION_TYPES)}")
        negative = [name for name, value in self.information_mix.items() if value < 0]
        if negative:
            raise ValueError(f"information_mix shares must be non-negative: {negative}")
        total = sum(self.information_mix.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"information_mix must sum to 1, got {total:.6f}")

        for name in (
            "followup_probability",
            "uncertain_tone_probability",
            "raw_query_ratio",
            "nmr_noise_ratio",
            "formula_noise_ratio",
            "modal_drop_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"AugmentorConfig.{name} must be between 0 and 1, got {value}")
        if not 0.0 < self.nmr_noise_strength <= 1.0:
            raise ValueError("AugmentorConfig.nmr_noise_strength must be in (0, 1]")
        if self.reaction_noise_ratio:
            raise ValueError("reaction noise is reserved for a future experiment; keep reaction_noise_ratio at 0")

        unknown_modes = [mode for mode in self.nmr_noise_modes if mode not in NMR_NOISE_MODES]
        if unknown_modes:
            raise ValueError(f"unknown NMR noise mode(s) {unknown_modes}; expected a subset of {list(NMR_NOISE_MODES)}")
        if self.nmr_noise_ratio and not self.nmr_noise_modes:
            raise ValueError("nmr_noise_ratio is set but nmr_noise_modes is empty")
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative (0 means the whole split)")

        # EnrichmentConfig.cache_path defaults independently (it can be used
        # standalone), so a caller who only overrides datasets_dir -- e.g. via
        # `--datasets-dir` -- would otherwise still cache enrichment under the
        # unrelated default directory. Re-derive it from datasets_dir whenever
        # the caller left the nested config on its own plain default.
        default_cache_path = str(Path(_default_datasets_dir()) / "enrichment.jsonl")
        if self.enrichment.cache_path == default_cache_path and self.datasets_dir != _default_datasets_dir():
            object.__setattr__(
                self,
                "enrichment",
                dataclasses.replace(self.enrichment, cache_path=str(Path(self.datasets_dir) / "enrichment.jsonl")),
            )


__all__ = [
    "INFORMATION_TYPES",
    "NMR_NOISE_MODES",
    "AugmentorConfig",
    "EnrichmentConfig",
]
