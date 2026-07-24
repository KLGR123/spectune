"""Configuration objects for Spectune tools."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class WebSearchConfig:
    """Connection settings and request defaults for the web-search API.

    The ``default_*`` fields are applied whenever a tool call omits the
    corresponding argument; they also populate the ``default`` shown in the
    tool's JSON schema so callers see the actual configured behavior.
    """

    api_url: str = "https://open.feedcoopapi.com/search_api/web_search"
    api_key: str = field(default_factory=lambda: os.getenv("VOLCENGINE_WEBSEARCH_API_KEY", ""))
    timeout_s: float = 60.0
    default_count: int = 3
    default_search_type: str = "web"
    default_need_content: bool = False
    default_need_url: bool = True
    default_need_summary: bool = True
    default_sites: str = ""
    default_block_hosts: str = ""
    default_auth_info_level: int = 0
    default_time_range: str = ""
    default_query_rewrite: bool = False


@dataclass(frozen=True, slots=True)
class CodeInterpreterConfig:
    """Settings for remote or explicitly enabled local code execution.

    ``sandbox`` sends code to a SandboxFusion-compatible HTTP endpoint.
    ``local`` executes a subprocess and must also set ``allow_local_execution``.
    """

    backend: Literal["sandbox", "local"] = "sandbox"
    sandbox_url: str = field(
        default_factory=lambda: os.getenv("SANDBOX_FUSION_URL", os.getenv("sandbox_fusion_url", ""))
    )
    timeout_s: int = 30
    language: str = "python"
    memory_limit_mb: int = 1024
    python_executable: str = sys.executable
    allow_local_execution: bool = False
    max_output_chars: int = 200_000


@dataclass(frozen=True, slots=True)
class NmrGenerateConfig:
    """Settings for the ``nmr_generate`` structure-generation backend."""

    api_url: str = field(default_factory=lambda: os.getenv("NMR_GENERATE_API_URL", ""))
    timeout_s: float = 180.0
    default_topk: int = 10


@dataclass(frozen=True, slots=True)
class NmrRepairConfig:
    """Settings for the ``nmr_repair`` formula-repair backend."""

    api_url: str = field(default_factory=lambda: os.getenv("NMR_REPAIR_API_URL", ""))
    timeout_s: float = 180.0


@dataclass(frozen=True, slots=True)
class NmrRerankConfig:
    """Settings for the ``nmr_rerank`` candidate-scoring backend."""

    api_url: str = field(default_factory=lambda: os.getenv("NMR_RANK_API_URL", ""))
    timeout_s: float = 180.0
    default_topk: int = 10


@dataclass(frozen=True, slots=True)
class NmrForwardPredictConfig:
    """Settings for the ``nmr_forward_predict`` MCP backend.

    The backend is an MCP tool (not a plain HTTP endpoint), so it is reached
    through an MCP client rather than raw HTTP.
    """

    mcp_url: str = field(default_factory=lambda: os.getenv("NMR_PREDICT_MCP_URL", ""))
    mcp_tool_name: str = field(default_factory=lambda: os.getenv("NMR_PREDICT_MCP_TOOL_NAME", "NMR_predict"))
    timeout_s: float = 90.0


@dataclass(frozen=True, slots=True)
class NmrExpSearchConfig:
    """Settings for the ``nmrexp_search`` experimental-database search backend.

    The backend is reached over plain HTTP JSON POST at
    ``{mcp_base_url}{endpoint_path}``. The ``default_*`` fields mirror the legacy
    search-config knobs (sigma/iteration/mutation controls for the underlying
    structure-search algorithm) and populate the tool's JSON schema defaults.
    """

    mcp_base_url: str = field(default_factory=lambda: os.getenv("NMREXP_SEARCH_MCP_BASE_URL", ""))
    endpoint_path: str = "/sync_nmr_service_mcp"
    timeout_s: float = 180.0
    default_num_search: int = 1000
    default_topk: int = 1000
    default_allowed_elements: tuple[str, ...] = ("C", "H", "N", "O")
    default_sigma_h: float = 1.0
    default_sigma_c: float = 10.0
    default_use_h_split: bool = False
    default_split_coef: float = 0.8
    default_max_iter: int = 2
    default_num_pool: int = 1000
    default_num_filter_pair: int = 200_000
    default_num_filter_mol: int = 1000
    default_num_mutate_mol: int = 100
    default_use_stereo: bool = False
    default_optional_halogens: tuple[str, ...] = ("F", "Cl", "Br", "I")
    default_max_cycle_length: int = 6
    default_invalid_patterns: tuple[str, ...] = (
        "[O][O]",
        "[R]=[R]=[R]",
        "[r3,r4]=[r3,r4]",
        "[O][F,Cl,Br,I]",
    )
    default_include_active_hs: str = "yes"


@dataclass(frozen=True, slots=True)
class ReactionLocalIndexSearchConfig:
    """Settings for the ``reaction_local_index_search`` local precedent lookup.

    Each path defaults to ``""`` (disabled) since spectune must not assume any
    particular cluster's filesystem layout. The tool skips any source whose
    path is unset or does not exist on disk, and reports ``unavailable`` only
    if none of the three sources are usable.
    """

    uspto_csv_path: str = field(default_factory=lambda: os.getenv("RXN_LOCAL_INDEX_USPTO_CSV", ""))
    chempile_parquet_path: str = field(default_factory=lambda: os.getenv("RXN_LOCAL_INDEX_CHEMPILE_PARQUET", ""))
    pistachio_smi_path: str = field(default_factory=lambda: os.getenv("RXN_LOCAL_INDEX_PISTACHIO_SMI", ""))
    max_chempile_records: int = 50_000
    max_pistachio_records: int = 50_000
    default_topk: int = 20


@dataclass(frozen=True, slots=True)
class AskcosReactionForwardPredictConfig:
    """Settings for the ``askcos_reaction_forward_predict`` public ASKCOS forward-prediction API."""

    base_url: str = field(default_factory=lambda: os.getenv("ASKCOS_PUBLIC_BASE_URL", "https://askcos.mit.edu"))
    timeout_s: float = 90.0
    default_topk: int = 20
    default_api_backend: str = "wldn5"
    verify_ssl: bool = True


@dataclass(frozen=True, slots=True)
class Unimol3ReactionForwardPredictConfig:
    """Settings for the ``unimol3_reaction_forward_predict`` Uni-Mol3 forward-prediction backend.

    Uni-Mol3 is not run in-process (it needs its own conda env / checkout); this
    tool is a pure HTTP client placeholder until a Uni-Mol3 service is deployed
    at a stable URL. ``api_url`` defaults to ``""`` (disabled), in which case the
    tool reports ``status="unavailable"`` rather than attempting any local
    subprocess/conda execution.
    """

    api_url: str = field(default_factory=lambda: os.getenv("UNIMOL3_REACTION_FORWARD_PREDICT_API_URL", ""))
    timeout_s: float = 180.0
    default_topk: int = 20


@dataclass(frozen=True, slots=True)
class SemanticScholarSearchConfig:
    """Settings for the ``semantic_scholar_search`` literature-search tool.

    No API key is required for basic use; setting one raises Semantic Scholar's
    rate limits but is entirely optional.
    """

    api_url: str = "https://api.semanticscholar.org/graph/v1/paper/search"
    api_key: str = field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""))
    timeout_s: float = 30.0
    default_max_results: int = 5


@dataclass(frozen=True, slots=True)
class CrossrefSearchConfig:
    """Settings for the ``crossref_search`` literature-search tool.

    Crossref recommends identifying API clients via a ``mailto`` contact (their
    "polite pool") but does not require one.
    """

    api_url: str = "https://api.crossref.org/works"
    mailto: str = field(default_factory=lambda: os.getenv("CROSSREF_MAILTO", ""))
    timeout_s: float = 30.0
    default_max_results: int = 5


@dataclass(frozen=True, slots=True)
class WikipediaSearchConfig:
    """Settings for the ``wikipedia_search`` literature-search tool.

    ``search_base_url``/``summary_base_url`` default to Wikipedia's real public
    endpoints (templated by ``language``); overriding them points the tool at a
    different host entirely, which is how tests substitute a local mock server.
    """

    language: str = field(default_factory=lambda: os.getenv("WIKIPEDIA_LANGUAGE", "en"))
    search_base_url: str = ""
    summary_base_url: str = ""
    timeout_s: float = 20.0
    default_max_results: int = 5
    fetch_summaries: bool = True


@dataclass(frozen=True, slots=True)
class SpectuneConfig:
    """Top-level configuration used by the default tool manager."""

    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    code_interpreter: CodeInterpreterConfig = field(default_factory=CodeInterpreterConfig)
    nmr_generate: NmrGenerateConfig = field(default_factory=NmrGenerateConfig)
    nmr_repair: NmrRepairConfig = field(default_factory=NmrRepairConfig)
    nmr_rerank: NmrRerankConfig = field(default_factory=NmrRerankConfig)
    nmr_forward_predict: NmrForwardPredictConfig = field(default_factory=NmrForwardPredictConfig)
    nmrexp_search: NmrExpSearchConfig = field(default_factory=NmrExpSearchConfig)
    reaction_local_index_search: ReactionLocalIndexSearchConfig = field(default_factory=ReactionLocalIndexSearchConfig)
    askcos_reaction_forward_predict: AskcosReactionForwardPredictConfig = field(
        default_factory=AskcosReactionForwardPredictConfig
    )
    unimol3_reaction_forward_predict: Unimol3ReactionForwardPredictConfig = field(
        default_factory=Unimol3ReactionForwardPredictConfig
    )
    semantic_scholar_search: SemanticScholarSearchConfig = field(default_factory=SemanticScholarSearchConfig)
    crossref_search: CrossrefSearchConfig = field(default_factory=CrossrefSearchConfig)
    wikipedia_search: WikipediaSearchConfig = field(default_factory=WikipediaSearchConfig)
