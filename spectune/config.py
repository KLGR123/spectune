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
class SpectuneConfig:
    """Top-level configuration used by the default tool manager."""

    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    code_interpreter: CodeInterpreterConfig = field(default_factory=CodeInterpreterConfig)
    nmr_generate: NmrGenerateConfig = field(default_factory=NmrGenerateConfig)
    nmr_repair: NmrRepairConfig = field(default_factory=NmrRepairConfig)
    nmr_rerank: NmrRerankConfig = field(default_factory=NmrRerankConfig)
    nmr_forward_predict: NmrForwardPredictConfig = field(default_factory=NmrForwardPredictConfig)
