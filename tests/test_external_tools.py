"""Opt-in tool tests that connect to external services.

Unlike ``test_mock_tools.py`` / ``test_mock_nmr_tools.py``, these tests call
actual upstream services using credentials already present in the
environment:

- ``VOLCENGINE_WEBSEARCH_API_KEY`` for ``web_search``
- ``SANDBOX_FUSION_URL`` (or ``sandbox_fusion_url``) for ``code_interpreter``
- ``NMR_GENERATE_API_URL`` for ``nmr_generate``
- ``NMR_REPAIR_API_URL`` for ``nmr_repair``
- ``NMR_RANK_API_URL`` for ``nmr_rerank``
- ``NMR_PREDICT_MCP_URL`` (plus the ``fastmcp`` package) for ``nmr_forward_predict``

Each test is skipped automatically when the corresponding environment
variable is not set, so the suite stays runnable without credentials while
still exercising the real network path whenever they are available.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest

from spectune import (
    CodeInterpreterConfig,
    CodeInterpreterTool,
    NmrForwardPredictConfig,
    NmrForwardPredictTool,
    NmrGenerateConfig,
    NmrGenerateTool,
    NmrRepairConfig,
    NmrRepairTool,
    NmrRerankConfig,
    NmrRerankTool,
    WebSearchConfig,
    WebSearchTool,
)

_HAS_EXTERNAL_WEB_SEARCH = bool(os.getenv("VOLCENGINE_WEBSEARCH_API_KEY"))
_HAS_EXTERNAL_SANDBOX = bool(os.getenv("SANDBOX_FUSION_URL") or os.getenv("sandbox_fusion_url"))
_HAS_EXTERNAL_NMR_GENERATE = bool(os.getenv("NMR_GENERATE_API_URL"))
_HAS_EXTERNAL_NMR_REPAIR = bool(os.getenv("NMR_REPAIR_API_URL"))
_HAS_EXTERNAL_NMR_RERANK = bool(os.getenv("NMR_RANK_API_URL"))
_HAS_EXTERNAL_NMR_FORWARD_PREDICT = (
    bool(os.getenv("NMR_PREDICT_MCP_URL")) and importlib.util.find_spec("fastmcp") is not None
)


@pytest.mark.skipif(not _HAS_EXTERNAL_WEB_SEARCH, reason="VOLCENGINE_WEBSEARCH_API_KEY is not set")
def test_external_web_search_returns_hits():
    # WebSearchConfig() reads VOLCENGINE_WEBSEARCH_API_KEY from the environment.
    tool = WebSearchTool(WebSearchConfig())

    result = asyncio.run(tool.execute({"query": "pyridine ring SMILES", "count": 1}))

    assert result.completion == "success", result.warnings
    assert result.status in {"ok", "no_hits"}
    assert isinstance(result.data.get("parsed_hits"), list)


@pytest.mark.skipif(not _HAS_EXTERNAL_SANDBOX, reason="SANDBOX_FUSION_URL is not set")
def test_external_sandbox_code_execution():
    # CodeInterpreterConfig() reads SANDBOX_FUSION_URL from the environment
    # and defaults to the "sandbox" backend.
    tool = CodeInterpreterTool(CodeInterpreterConfig())

    result = asyncio.run(tool.execute({"code": "print(1 + 1)"}))

    assert result.completion == "success", result.warnings
    assert result.data["backend"] == "sandbox"
    assert result.data["stdout"] == "2\n"


@pytest.mark.skipif(not _HAS_EXTERNAL_NMR_GENERATE, reason="NMR_GENERATE_API_URL is not set")
def test_external_nmr_generate_returns_candidates():
    # NmrGenerateConfig() reads NMR_GENERATE_API_URL from the environment.
    tool = NmrGenerateTool(NmrGenerateConfig())

    result = asyncio.run(tool.execute({"formula": "C2H6O", "topk": 5}))

    assert result.completion == "success", result.warnings
    assert result.status in {"ok", "no_candidates"}
    assert isinstance(result.data.get("candidates"), list)


@pytest.mark.skipif(not _HAS_EXTERNAL_NMR_REPAIR, reason="NMR_REPAIR_API_URL is not set")
def test_external_nmr_repair_returns_candidate():
    # NmrRepairConfig() reads NMR_REPAIR_API_URL from the environment.
    tool = NmrRepairTool(NmrRepairConfig())

    result = asyncio.run(tool.execute({"input_smiles": "CCO", "target_formula": "C2H4O"}))

    assert result.completion == "success", result.warnings
    assert result.status in {"ok", "no_candidates"}


@pytest.mark.skipif(not _HAS_EXTERNAL_NMR_RERANK, reason="NMR_RANK_API_URL is not set")
def test_external_nmr_rerank_returns_ranking():
    # NmrRerankConfig() reads NMR_RANK_API_URL from the environment.
    tool = NmrRerankTool(NmrRerankConfig())

    result = asyncio.run(tool.execute({"smiles_list": ["CCO", "COC"]}))

    assert result.completion == "success", result.warnings
    assert result.status in {"ok", "no_candidates"}
    assert isinstance(result.data.get("candidates"), list)


@pytest.mark.skipif(
    not _HAS_EXTERNAL_NMR_FORWARD_PREDICT,
    reason="NMR_PREDICT_MCP_URL is not set or fastmcp is not installed",
)
def test_external_nmr_forward_predict_returns_shifts():
    # NmrForwardPredictConfig() reads NMR_PREDICT_MCP_URL from the environment.
    tool = NmrForwardPredictTool(NmrForwardPredictConfig())

    result = asyncio.run(tool.execute({"smiles_list": ["CCO"]}))

    assert result.completion == "success", result.warnings
    assert result.status in {"ok", "no_candidates"}
