import importlib.util

import pandas as pd
import pytest

from spectune.tools.reaction_index_build import build_reaction_index

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit is not installed")


@pytest.fixture
def uspto_csv(tmp_path):
    csv_path = tmp_path / "uspto_test.csv"
    csv_path.write_text(
        "reactant,product\nCC(=O)Cl.OCC,CC(=O)OCC\nc1ccccc1Br.OCC,c1ccccc1OCC\nnot-a-smiles,alsobad\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def chempile_parquet(tmp_path):
    path = tmp_path / "chempile.parquet"
    pd.DataFrame({"text": ["some prefix CC(=O)Cl.OCC>>CC(=O)OCC trailing text", "no reaction here"]}).to_parquet(path)
    return path


@pytest.fixture
def pistachio_smi(tmp_path):
    path = tmp_path / "pistachio.smi"
    path.write_text(
        "CC(=O)Cl.OCC>>CC(=O)OCC\tUS1234\t\t\tacylation\nnot>>a>>reaction\tUS9999\n",
        encoding="utf-8",
    )
    return path


def test_build_uspto_index_writes_canonicalized_parquet(uspto_csv, tmp_path):
    output_dir = tmp_path / "prebuilt"

    summaries = build_reaction_index(output_dir=str(output_dir), sources=["uspto"], uspto_csv_path=str(uspto_csv))

    assert summaries == [
        {
            "source": "uspto",
            "status": "built",
            "raw_path": str(uspto_csv),
            "output_path": str(output_dir / "uspto.parquet"),
            "records_written": 2,
        }
    ]
    frame = pd.read_parquet(output_dir / "uspto.parquet")
    assert list(frame.columns) == [
        "source",
        "source_id",
        "reactant_components",
        "product",
        "canonical_product",
        "product_formula",
        "patent_id",
        "reaction_class_name",
    ]
    assert len(frame) == 2
    assert set(frame["source"]) == {"uspto_csv"}
    row = frame.iloc[0]
    assert row["canonical_product"] == "CCOC(C)=O"
    assert row["product_formula"] == "C4H8O2"
    assert sorted(row["reactant_components"]) == ["CCO", "CC(=O)Cl"] or sorted(row["reactant_components"]) == [
        "CC(=O)Cl",
        "CCO",
    ]


def test_build_chempile_index_extracts_embedded_reaction_smiles(chempile_parquet, tmp_path):
    output_dir = tmp_path / "prebuilt"

    summaries = build_reaction_index(
        output_dir=str(output_dir), sources=["chempile"], chempile_parquet_path=str(chempile_parquet)
    )

    assert summaries[0]["status"] == "built"
    assert summaries[0]["records_written"] == 1
    frame = pd.read_parquet(output_dir / "chempile.parquet")
    assert len(frame) == 1
    assert frame.iloc[0]["source"] == "chempile_lift_uspto"
    assert frame.iloc[0]["canonical_product"] == "CCOC(C)=O"


def test_build_pistachio_index_keeps_patent_metadata(pistachio_smi, tmp_path):
    output_dir = tmp_path / "prebuilt"

    summaries = build_reaction_index(
        output_dir=str(output_dir), sources=["pistachio"], pistachio_smi_path=str(pistachio_smi)
    )

    assert summaries[0]["records_written"] == 1
    frame = pd.read_parquet(output_dir / "pistachio.parquet")
    row = frame.iloc[0]
    assert row["source"] == "pistachio_smi"
    assert row["patent_id"] == "US1234"
    assert row["reaction_class_name"] == "acylation"


def test_build_uses_multiple_workers(uspto_csv, tmp_path):
    output_dir = tmp_path / "prebuilt"

    summaries = build_reaction_index(
        output_dir=str(output_dir), sources=["uspto"], uspto_csv_path=str(uspto_csv), workers=2
    )

    assert summaries[0]["records_written"] == 2


def test_skips_source_with_no_configured_raw_path(tmp_path):
    output_dir = tmp_path / "prebuilt"

    summaries = build_reaction_index(output_dir=str(output_dir), sources=["chempile"])

    assert summaries == [{"source": "chempile", "status": "skipped", "reason": "no raw path configured"}]


def test_reports_error_for_missing_raw_path(tmp_path):
    output_dir = tmp_path / "prebuilt"
    missing = tmp_path / "does_not_exist.csv"

    summaries = build_reaction_index(output_dir=str(output_dir), sources=["uspto"], uspto_csv_path=str(missing))

    assert summaries[0]["status"] == "error"


def test_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown reaction index source"):
        build_reaction_index(output_dir="/tmp/whatever", sources=["not-a-real-source"])
