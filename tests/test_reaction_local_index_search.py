import asyncio
import importlib.util

import pytest

from spectune import ReactionLocalIndexSearchConfig, ReactionLocalIndexSearchTool

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is not installed")


@pytest.fixture
def uspto_csv(tmp_path):
    csv_path = tmp_path / "uspto_test.csv"
    csv_path.write_text(
        "reactant,product\nCC(=O)Cl.OCC,CC(=O)OCC\nc1ccccc1Br.OCC,c1ccccc1OCC\n",
        encoding="utf-8",
    )
    return csv_path


def test_reports_unavailable_without_rdkit(monkeypatch):
    monkeypatch.setattr("spectune.tools.reaction_local_index_search.has_rdkit", lambda: False)
    tool = ReactionLocalIndexSearchTool(ReactionLocalIndexSearchConfig(uspto_csv_path="/tmp/does-not-matter.csv"))

    result = asyncio.run(tool.execute({"reactants": ["CCO"]}))

    assert result.completion == "failure"
    assert result.status == "unavailable"


def test_reports_unavailable_without_any_configured_source():
    tool = ReactionLocalIndexSearchTool(ReactionLocalIndexSearchConfig())

    result = asyncio.run(tool.execute({"reactants": ["CCO"]}))

    assert result.completion == "failure"
    assert result.status == "unavailable"


def test_requires_valid_reactants(uspto_csv):
    config = ReactionLocalIndexSearchConfig(uspto_csv_path=str(uspto_csv))
    tool = ReactionLocalIndexSearchTool(config)

    result = asyncio.run(tool.execute({"reactants": ["not-a-smiles"]}))

    assert result.completion == "failure"
    assert result.status == "error"


def test_finds_precedent_by_reactant_overlap(uspto_csv):
    config = ReactionLocalIndexSearchConfig(uspto_csv_path=str(uspto_csv))
    tool = ReactionLocalIndexSearchTool(config)

    result = asyncio.run(tool.execute({"reactants": ["CC(=O)Cl", "OCC"], "topk": 5}))

    assert result.completion == "success"
    assert result.status == "ok"
    candidates = result.data["candidates"]
    assert candidates
    assert candidates[0]["source"] == "uspto_csv"
    assert candidates[0]["rank"] == 1
    assert "reaction_evidence" in candidates[0]


def test_accepts_reaction_smiles_shorthand(uspto_csv):
    config = ReactionLocalIndexSearchConfig(uspto_csv_path=str(uspto_csv))
    tool = ReactionLocalIndexSearchTool(config)

    result = asyncio.run(tool.execute({"reaction_smiles": "CC(=O)Cl.OCC>>CC(=O)OCC"}))

    assert result.completion == "success"
    assert result.status == "ok"


def test_target_formula_filters_low_scoring_mismatches(uspto_csv):
    config = ReactionLocalIndexSearchConfig(uspto_csv_path=str(uspto_csv))
    tool = ReactionLocalIndexSearchTool(config)

    result = asyncio.run(tool.execute({"reactants": ["CC(=O)Cl", "OCC"], "target_formula": "totally-wrong-formula"}))

    assert result.completion == "success"
    candidate = result.data["candidates"][0]
    assert candidate["constraint_evidence"]["formula_match"] is False
    assert "target_formula_mismatch" in candidate["warnings"]


def test_no_records_when_csv_missing(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    config = ReactionLocalIndexSearchConfig(uspto_csv_path=str(missing))
    tool = ReactionLocalIndexSearchTool(config)

    result = asyncio.run(tool.execute({"reactants": ["CCO"]}))

    assert result.status == "no_candidates"
