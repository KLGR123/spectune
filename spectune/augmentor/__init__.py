"""Dataset construction for spectrum-interpretation training.

Two stages, deliberately separable:

- :class:`~spectune.augmentor.enrichment.Enricher` gathers facts about each
  truth molecule (names, fragments, reaction context, properties) from tools
  and local corpora, and caches them by canonical SMILES.
- :class:`~spectune.augmentor.augmentor.Augmentor` samples truth rows, runs the
  enricher, and constructs user-style single- or multi-turn queries according
  to a configurable scenario mixture, optionally rewritten by an LLM and
  optionally noised.
"""

from .augmentor import Augmentor
from .config import INFORMATION_TYPES, NMR_NOISE_MODES, AugmentorConfig, EnrichmentConfig
from .enrichment import Enricher
from .formulas import perturb_formula
from .fragments import FRAGMENT_LEXICON, detect_fragments, molecule_properties
from .reactions import RETRO_TEMPLATES, polymer_context, retro_routes, scan_product_precedents
from .spectra import apply_nmr_noise, build_spectrum_text

__all__ = [
    "FRAGMENT_LEXICON",
    "INFORMATION_TYPES",
    "NMR_NOISE_MODES",
    "RETRO_TEMPLATES",
    "Augmentor",
    "AugmentorConfig",
    "EnrichmentConfig",
    "Enricher",
    "apply_nmr_noise",
    "build_spectrum_text",
    "detect_fragments",
    "molecule_properties",
    "perturb_formula",
    "polymer_context",
    "retro_routes",
    "scan_product_precedents",
]
