"""Opt-in tool tests that connect to external services.

Unlike ``test_mock_tools.py`` and the other ``test_mock_*.py`` files, these
tests call actual upstream services. Most are gated on credentials/config
already present in the environment:

- ``VOLCENGINE_WEBSEARCH_API_KEY`` for ``web_search``
- ``SANDBOX_FUSION_URL`` (or ``sandbox_fusion_url``) for ``code_interpreter``
- ``NMR_GENERATE_API_URL`` for ``nmr_generate``
- ``NMR_REPAIR_API_URL`` for ``nmr_repair``
- ``NMR_RANK_API_URL`` for ``nmr_rerank``
- ``NMR_PREDICT_API_URL`` for ``nmr_forward_predict``
- ``NMREXP_SEARCH_MCP_BASE_URL`` for ``nmrexp_search``
- ``RXN_LOCAL_INDEX_USPTO_CSV`` / ``RXN_LOCAL_INDEX_CHEMPILE_PARQUET`` /
  ``RXN_LOCAL_INDEX_PISTACHIO_SMI`` (at least one) for ``reaction_local_index_search``
- ``UNIMOL3_REACTION_FORWARD_PREDICT_API_URL`` for ``unimol3_reaction_forward_predict``
  (unset until a Uni-Mol3 service is deployed, so this test currently always skips)

A few tools (``askcos_reaction_forward_predict``, ``semantic_scholar_search``,
``crossref_search``, ``wikipedia_search``) call genuinely public APIs that need
no credentials at all. Gating those the same way as the credential-based tests
above would mean they run unconditionally in any environment with network
access, including sandboxes with no internet; instead they are gated behind a
single opt-in flag, ``SPECTUNE_ENABLE_NETWORK_TESTS=1``.

Each test is skipped automatically when its prerequisite is not met, so the
suite stays runnable offline/without credentials while still exercising the
real network path whenever the prerequisite is available.
"""

from __future__ import annotations

import asyncio
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
    ReactionLocalIndexSearchConfig,
    ReactionLocalIndexSearchTool,
    WebSearchConfig,
    WebSearchTool,
)

pytestmark = pytest.mark.external

_HAS_EXTERNAL_WEB_SEARCH = bool(os.getenv("VOLCENGINE_WEBSEARCH_API_KEY"))
_HAS_EXTERNAL_SANDBOX = bool(os.getenv("SANDBOX_FUSION_URL") or os.getenv("sandbox_fusion_url"))
_HAS_EXTERNAL_NMR_GENERATE = bool(os.getenv("NMR_GENERATE_API_URL"))
_HAS_EXTERNAL_NMR_REPAIR = bool(os.getenv("NMR_REPAIR_API_URL"))
_HAS_EXTERNAL_NMR_RERANK = bool(os.getenv("NMR_RANK_API_URL"))
_HAS_EXTERNAL_NMR_FORWARD_PREDICT = bool(os.getenv("NMR_PREDICT_API_URL"))
_HAS_EXTERNAL_NMREXP_SEARCH = bool(os.getenv("NMREXP_SEARCH_MCP_BASE_URL"))
_HAS_LOCAL_REACTION_INDEX = bool(
    os.getenv("RXN_LOCAL_INDEX_USPTO_CSV")
    or os.getenv("RXN_LOCAL_INDEX_CHEMPILE_PARQUET")
    or os.getenv("RXN_LOCAL_INDEX_PISTACHIO_SMI")
)
_HAS_EXTERNAL_UNIMOL3_REACTION_FORWARD_PREDICT = bool(os.getenv("UNIMOL3_REACTION_FORWARD_PREDICT_API_URL"))
_NETWORK_TESTS_ENABLED = bool(os.getenv("SPECTUNE_ENABLE_NETWORK_TESTS"))


@pytest.mark.skipif(not _HAS_EXTERNAL_WEB_SEARCH, reason="VOLCENGINE_WEBSEARCH_API_KEY is not set")
def test_external_web_search_returns_hits():
    # WebSearchConfig() reads VOLCENGINE_WEBSEARCH_API_KEY from the environment.
    tool = WebSearchTool(WebSearchConfig())

    result = asyncio.run(tool.execute({"query": "pyridine ring SMILES", "count": 1}))
    print(result)
    assert result.completion == "success", result.warnings
    assert result.status in {"ok", "no_hits"}
    assert isinstance(result.data.get("parsed_hits"), list)


@pytest.mark.skipif(not _HAS_EXTERNAL_SANDBOX, reason="SANDBOX_FUSION_URL is not set")
def test_external_sandbox_code_execution():
    # CodeInterpreterConfig() reads SANDBOX_FUSION_URL from the environment
    # and defaults to the "sandbox" backend.
    tool = CodeInterpreterTool(CodeInterpreterConfig())

    result = asyncio.run(tool.execute({"code": "print(1 + 1)"}))
    print(result)
    assert result.completion == "success", result.warnings
    assert result.data["backend"] == "sandbox"
    assert result.data["stdout"] == "2\n"


@pytest.mark.skipif(not _HAS_EXTERNAL_NMR_GENERATE, reason="NMR_GENERATE_API_URL is not set")
def test_external_nmr_generate_returns_candidates():
    # NmrGenerateConfig() reads NMR_GENERATE_API_URL from the environment.
    tool = NmrGenerateTool(NmrGenerateConfig())

    result = asyncio.run(tool.execute({"h_shifts": [7.41, 7.34, 7.24, 7.12, 5.06, 2.89, 2.73], "topk": 5}))
    print(result)
    assert result.completion == "success", result.warnings
    assert result.status in {"ok", "no_candidates"}
    assert isinstance(result.data.get("candidates"), list)


@pytest.mark.skipif(not _HAS_EXTERNAL_NMR_REPAIR, reason="NMR_REPAIR_API_URL is not set")
def test_external_nmr_repair_returns_candidate():
    # NmrRepairConfig() reads NMR_REPAIR_API_URL from the environment.
    tool = NmrRepairTool(NmrRepairConfig())

    result = asyncio.run(tool.execute({"input_smiles": "CCO", "target_formula": "C2H4O"}))
    print(result)
    assert result.completion == "success", result.warnings
    assert result.status in {"ok", "no_candidates"}


# @pytest.mark.skipif(not _HAS_EXTERNAL_NMR_RERANK, reason="NMR_RANK_API_URL is not set")
# def test_external_nmr_rerank_returns_ranking():
#     # NmrRerankConfig() reads NMR_RANK_API_URL from the environment.
#     tool = NmrRerankTool(NmrRerankConfig())

#     result = asyncio.run(tool.execute({"smiles_list": ["CCO", "COC"]}))
#     print(result)
#     assert result.completion == "success", result.warnings
#     assert result.status in {"ok", "no_candidates"}
#     assert isinstance(result.data.get("candidates"), list)


@pytest.mark.skipif(
    not _HAS_EXTERNAL_NMR_FORWARD_PREDICT,
    reason="NMR_PREDICT_API_URL is not set or fastmcp is not installed",
)
def test_external_nmr_forward_predict_returns_shifts():
    # NmrForwardPredictConfig() reads NMR_PREDICT_API_URL from the environment.
    tool = NmrForwardPredictTool(NmrForwardPredictConfig())

    result = asyncio.run(tool.execute({"smiles": "CCO", "solvent": "CDCl3"}))
    print(result)
    assert result.completion == "success", result.warnings
    assert result.status == "ok"
    assert "atoms_shift" in result.data


# @pytest.mark.skipif(not _HAS_EXTERNAL_NMREXP_SEARCH, reason="NMREXP_SEARCH_MCP_BASE_URL is not set")
# def test_external_nmrexp_search_returns_candidates():
#     # NmrExpSearchConfig() reads NMREXP_SEARCH_MCP_BASE_URL from the environment.
#     tool = NmrExpSearchTool(NmrExpSearchConfig())

#     result = asyncio.run(tool.execute({"h_shifts": [1.2, 3.6], "c_shifts": [18.0, 58.0], "topk": 5}))

#     assert result.completion == "success", result.warnings
#     assert result.status in {"ok", "no_candidates"}
#     assert isinstance(result.data.get("candidates"), list)


# @pytest.mark.skipif(not _HAS_LOCAL_REACTION_INDEX, reason="no RXN_LOCAL_INDEX_* path is configured")
# def test_external_reaction_local_index_search_returns_candidates():
#     # ReactionLocalIndexSearchConfig() reads the RXN_LOCAL_INDEX_* paths from the environment.
#     tool = ReactionLocalIndexSearchTool(ReactionLocalIndexSearchConfig())

#     result = asyncio.run(tool.execute({"reactants": ["CCO", "CC(=O)Cl"], "topk": 1}))
#     print(result)
#     assert result.completion in {"success", "partial"}, result.warnings
#     assert result.status in {"ok", "no_candidates"}


# @pytest.mark.skipif(not _NETWORK_TESTS_ENABLED, reason="SPECTUNE_ENABLE_NETWORK_TESTS is not set")
# def test_external_askcos_reaction_forward_predict_returns_candidates():
#     # ASKCOS's public API needs no credentials; gated on the opt-in network flag
#     # instead so the suite does not require internet access by default.
#     tool = AskcosReactionForwardPredictTool(AskcosReactionForwardPredictConfig())

#     result = asyncio.run(tool.execute({"reactants": ["CCBr", "[OH-]"], "topk": 3}))

#     assert result.completion == "success", result.warnings
#     assert result.status in {"ok", "no_candidates"}
#     assert isinstance(result.data.get("candidates"), list)


# @pytest.mark.skipif(
#     not _HAS_EXTERNAL_UNIMOL3_REACTION_FORWARD_PREDICT,
#     reason="UNIMOL3_REACTION_FORWARD_PREDICT_API_URL is not set",
# )
# def test_external_unimol3_reaction_forward_predict_returns_candidates():
#     # Unimol3ReactionForwardPredictConfig() reads UNIMOL3_REACTION_FORWARD_PREDICT_API_URL
#     # from the environment; this test always skips until a Uni-Mol3 service is deployed there.
#     tool = Unimol3ReactionForwardPredictTool(Unimol3ReactionForwardPredictConfig())

#     result = asyncio.run(tool.execute({"reactants": ["CCBr", "[OH-]"], "topk": 3}))

#     assert result.completion == "success", result.warnings
#     assert result.status in {"ok", "no_candidates"}
#     assert isinstance(result.data.get("candidates"), list)


# @pytest.mark.skipif(not _NETWORK_TESTS_ENABLED, reason="SPECTUNE_ENABLE_NETWORK_TESTS is not set")
# def test_external_semantic_scholar_search_returns_hits():
#     tool = SemanticScholarSearchTool(SemanticScholarSearchConfig())

#     result = asyncio.run(tool.execute({"query": "Suzuki-Miyaura coupling mechanism", "max_results": 2}))

#     assert result.completion == "success", result.warnings
#     assert result.status in {"ok", "no_hits"}
#     assert isinstance(result.data.get("hits"), list)


# @pytest.mark.skipif(not _NETWORK_TESTS_ENABLED, reason="SPECTUNE_ENABLE_NETWORK_TESTS is not set")
# def test_external_crossref_search_returns_hits():
#     tool = CrossrefSearchTool(CrossrefSearchConfig())

#     result = asyncio.run(tool.execute({"query": "NMR structure elucidation", "max_results": 2}))

#     assert result.completion == "success", result.warnings
#     assert result.status in {"ok", "no_hits"}
#     assert isinstance(result.data.get("hits"), list)


# @pytest.mark.skipif(not _NETWORK_TESTS_ENABLED, reason="SPECTUNE_ENABLE_NETWORK_TESTS is not set")
# def test_external_wikipedia_search_returns_hits():
#     tool = WikipediaSearchTool(WikipediaSearchConfig())

#     result = asyncio.run(tool.execute({"query": "benzene", "max_results": 2}))

#     assert result.completion == "success", result.warnings
#     assert result.status in {"ok", "no_hits"}
#     assert isinstance(result.data.get("hits"), list)
