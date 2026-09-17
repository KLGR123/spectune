import asyncio
import importlib.util

import pytest

from spectune import ReactionLocalIndexSearchConfig, ReactionLocalIndexSearchTool

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is not installed")


@pytest.fixture(autouse=True)
def _clear_reaction_index_env(monkeypatch):
    # Tests build ReactionLocalIndexSearchConfig with only a subset of paths set and rely on
    # the rest defaulting to "" (disabled). Without this, a developer's real secrets.env (e.g.
    # RXN_LOCAL_INDEX_CHEMPILE_PARQUET/RXN_LOCAL_INDEX_PISTACHIO_SMI pointing at multi-GB files)
    # leaks into these tests via os.getenv() defaults, making them slow and non-hermetic.
    for var in (
        "RXN_LOCAL_INDEX_USPTO_CSV",
        "RXN_LOCAL_INDEX_CHEMPILE_PARQUET",
        "RXN_LOCAL_INDEX_PISTACHIO_SMI",
        "RXN_LOCAL_INDEX_PREBUILT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


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
    config = ReactionLocalIndexSearchConfig(uspto_csv_path="", chempile_parquet_path="", pistachio_smi_path="")
    tool = ReactionLocalIndexSearchTool(config)

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
    config = ReactionLocalIndexSearchConfig(
        uspto_csv_path=str(missing),
        chempile_parquet_path="",
        pistachio_smi_path="",
    )
    tool = ReactionLocalIndexSearchTool(config)

    result = asyncio.run(tool.execute({"reactants": ["CCO"]}))

    assert result.status == "no_candidates"


def test_prebuilt_index_is_preferred_and_skips_raw_parsing(uspto_csv, tmp_path, monkeypatch):
    from spectune.tools.reaction_index_build import build_reaction_index

    prebuilt_dir = tmp_path / "prebuilt"
    summary = build_reaction_index(
        output_dir=str(prebuilt_dir), sources=["uspto"], uspto_csv_path=str(uspto_csv), workers=1
    )
    assert summary == [
        {
            "source": "uspto",
            "status": "built",
            "raw_path": str(uspto_csv),
            "output_path": str(prebuilt_dir / "uspto.parquet"),
            "records_written": 2,
        }
    ]

    def _boom(*args, **kwargs):
        raise AssertionError("raw uspto CSV parsing should not run when a prebuilt index is available")

    monkeypatch.setattr("spectune.tools.reaction_local_index_search._load_uspto_csv", _boom)

    config = ReactionLocalIndexSearchConfig(uspto_csv_path=str(uspto_csv), prebuilt_index_dir=str(prebuilt_dir))
    tool = ReactionLocalIndexSearchTool(config)

    result = asyncio.run(tool.execute({"reactants": ["CC(=O)Cl", "OCC"], "topk": 5}))

    assert result.completion == "success"
    assert result.status == "ok"
    assert result.warnings == []
    candidates = result.data["candidates"]
    assert candidates
    assert candidates[0]["source"] == "uspto_csv"


def test_falls_back_to_raw_parsing_when_prebuilt_file_is_missing(uspto_csv, tmp_path):
    empty_prebuilt_dir = tmp_path / "empty_prebuilt"
    empty_prebuilt_dir.mkdir()
    config = ReactionLocalIndexSearchConfig(uspto_csv_path=str(uspto_csv), prebuilt_index_dir=str(empty_prebuilt_dir))
    tool = ReactionLocalIndexSearchTool(config)

    result = asyncio.run(tool.execute({"reactants": ["CC(=O)Cl", "OCC"], "topk": 5}))

    assert result.completion == "success"
    assert result.status == "ok"
    assert any("build-reaction-index" in warning for warning in result.warnings)


def test_constraint_evidence_skips_rdkit_when_no_constraints(monkeypatch):
    from spectune.tools.reaction_local_index_search import _constraint_evidence

    def _boom(*args, **kwargs):
        raise AssertionError("RDKit should not be touched when constraints is empty")

    monkeypatch.setattr("rdkit.Chem.MolFromSmiles", _boom)

    evidence = _constraint_evidence("CCO", "C2H6O", [], "C2H6O")

    assert evidence == {
        "formula": "C2H6O",
        "formula_match": True,
        "matched_constraints": [],
        "missing_constraints": [],
    }


def test_constraint_evidence_checks_functional_groups_when_present():
    from spectune.tools.reaction_local_index_search import _constraint_evidence

    evidence = _constraint_evidence("CC(=O)OCC", "C4H8O2", ["ester", "amine"], "")

    assert evidence["matched_constraints"] == ["ester"]
    assert evidence["missing_constraints"] == ["amine"]
