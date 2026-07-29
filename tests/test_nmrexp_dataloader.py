import importlib.util

import pytest

from spectune import NmrExpDataLoader, NmrExpDataLoaderConfig

_HAS_PANDAS = importlib.util.find_spec("pandas") is not None
_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None

pytestmark = pytest.mark.skipif(not _HAS_PANDAS, reason="pandas is not installed")

_BASE_COLUMNS = [
    "Filename",
    "SMILES",
    "Page_in_file_mol",
    "Page_in_file_para",
    "Location_in_page_mol",
    "Location_in_page_para",
    "NMR_type",
    "NMR_frequency",
    "NMR_solvent",
    "NMR_shift_text",
    "NMR_note",
    "NMR_processed",
    "Atom_number",
    "Atom_number_diff_env",
    "Atom_number_abstract",
]
_CHECKED_EXTRA_COLUMNS = [
    "smiles_actual",
    "text_in_pdf",
    "nmr_frequency_right",
    "nmr_solvent_right",
    "nmr_processed_right",
    "is_same_molecule",
    "is_same_skeleton",
    "num_chiral_centers",
]


def _base_row(**overrides):
    row = {
        "Filename": "10.0000_test.0001",
        "SMILES": "CCO",
        "Page_in_file_mol": 1.0,
        "Page_in_file_para": 1,
        "Location_in_page_mol": "[0.1 0.2 0.3 0.4]",
        "Location_in_page_para": "[0.1 0.2 0.3 0.4]",
        "NMR_type": "1H NMR",
        "NMR_frequency": "400 MHz",
        "NMR_solvent": "CDCl3",
        "NMR_shift_text": "3.70 (q, 2H), 1.20 (t, 3H)",
        "NMR_note": None,
        "NMR_processed": "[('q', [], '2H', 3.70, 3.70), ('t', [], '3H', 1.20, 1.20)]",
        "Atom_number": 5,
        "Atom_number_diff_env": 2,
        "Atom_number_abstract": 2.0,
    }
    row.update(overrides)
    return row


def _checked_row(**overrides):
    row = _base_row()
    row.update(
        {
            "smiles_actual": row["SMILES"],
            "text_in_pdf": "1H NMR (400 MHz, CDCl3) d 3.70 (q, 2H), 1.20 (t, 3H)",
            "nmr_frequency_right": "right",
            "nmr_solvent_right": "right",
            "nmr_processed_right": "right",
            "is_same_molecule": True,
            "is_same_skeleton": True,
            "num_chiral_centers": 0,
        }
    )
    row.update(overrides)
    return row


@pytest.fixture
def raw_dir(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def processed_dir(tmp_path):
    return tmp_path / "processed"


def _write_checked_csv(path, rows):
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_BASE_COLUMNS + _CHECKED_EXTRA_COLUMNS).to_csv(path, index=False)


def _write_raw_csv(path, rows):
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_BASE_COLUMNS).to_csv(path, index=False)


def _config(raw_dir, processed_dir, **overrides) -> NmrExpDataLoaderConfig:
    # Tests default to a single checked test source so preprocessing with
    # no arguments don't need every entry of the real 7-source default mapping
    # to exist on disk; override both mappings to test train/test separation.
    overrides.setdefault("sources", {"checked": "test_300_checked.csv"})
    overrides.setdefault("truth_splits", {"test": ("checked",)})
    return NmrExpDataLoaderConfig(raw_dir=str(raw_dir), processed_dir=str(processed_dir), **overrides)


class TestPreprocessBuildsTruth:
    def test_resolves_smiles_actual_as_gt_when_it_differs(self, raw_dir, processed_dir):
        rows = [
            _checked_row(SMILES="c1ccccc1C", smiles_actual="Cc1ccccc1"),  # extraction differs from verified gt
        ]
        _write_checked_csv(raw_dir / "test_300_checked.csv", rows)
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        summary = loader.preprocess()["test"]
        dataset = loader.load_truth("test")

        assert summary["status"] == "built"
        assert summary["sources"]["checked"]["is_checked"] is True
        assert len(dataset) == 1
        record = dataset[0]
        assert record["gt_smiles"] == "Cc1ccccc1"
        assert record["provenance"]["smiles_raw"] == "c1ccccc1C"
        assert record["modality"] == "nmr"
        assert record["num_of_queries"] == 1
        assert record["ms"] is None
        assert record["quality"]["smiles_actual"] == "Cc1ccccc1"

    def test_falls_back_to_raw_smiles_when_source_is_unchecked_no_quality_block(self, raw_dir, processed_dir):
        _write_raw_csv(raw_dir / "raw.csv", [_base_row(SMILES="CCO")])
        config = _config(
            raw_dir,
            processed_dir,
            sources={"raw": "raw.csv"},
            truth_splits={"train": ("raw",)},
        )
        loader = NmrExpDataLoader(config)

        dataset = loader.load_truth("train")

        assert len(dataset) == 1
        record = dataset[0]
        assert record["gt_smiles"] == "CCO"
        assert record["quality"] is None

    def test_nmr_fields_and_processed_peaks_are_parsed(self, raw_dir, processed_dir):
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row()])
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        record = loader.load_truth("test")[0]

        assert record["nmr"]["type"] == "1H NMR"
        assert record["nmr"]["frequency"] == "400 MHz"
        assert record["nmr"]["solvent"] == "CDCl3"
        assert record["nmr"]["shift_text"] == "3.70 (q, 2H), 1.20 (t, 3H)"
        assert record["nmr"]["atom_number"] == 5
        assert record["nmr"]["processed"] == [["q", [], "2H", 3.70, 3.70], ["t", [], "3H", 1.20, 1.20]]

    def test_provenance_omits_task_irrelevant_page_and_location_fields(self, raw_dir, processed_dir):
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row()])
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        record = loader.load_truth("test")[0]

        assert set(record["provenance"]) == {
            "dataset",
            "split",
            "source",
            "source_file",
            "row_index",
            "filename",
            "smiles_raw",
        }
        assert record["provenance"]["split"] == "test"
        assert record["provenance"]["source"] == "checked"

    def test_drops_rows_with_empty_gt_smiles(self, raw_dir, processed_dir):
        _write_checked_csv(
            raw_dir / "test_300_checked.csv",
            [_checked_row(), _checked_row(SMILES="", smiles_actual="")],
        )
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        summary = loader.preprocess()["test"]

        assert summary["kept"] == 1
        assert summary["dropped_empty_gt_smiles"] == 1

    def test_drop_qc_wrong_default_drops_flagged_rows(self, raw_dir, processed_dir):
        _write_checked_csv(
            raw_dir / "test_300_checked.csv",
            [_checked_row(), _checked_row(nmr_solvent_right="wrong")],
        )
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        summary = loader.preprocess()["test"]

        assert summary["kept"] == 1
        assert summary["dropped_qc_wrong"] == 1

    def test_drop_qc_wrong_can_be_disabled(self, raw_dir, processed_dir):
        _write_checked_csv(
            raw_dir / "test_300_checked.csv",
            [_checked_row(), _checked_row(nmr_solvent_right="wrong")],
        )
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir, drop_qc_wrong=False))

        summary = loader.preprocess()["test"]

        assert summary["kept"] == 2

    def test_min_quality_same_skeleton_drops_mismatched_rows(self, raw_dir, processed_dir):
        _write_checked_csv(
            raw_dir / "test_300_checked.csv",
            [_checked_row(), _checked_row(is_same_skeleton=False)],
        )
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir, min_quality="same_skeleton"))

        summary = loader.preprocess()["test"]

        assert summary["kept"] == 1
        assert summary["dropped_quality"] == 1

    def test_allowed_nmr_types_filters_other_types(self, raw_dir, processed_dir):
        _write_checked_csv(
            raw_dir / "test_300_checked.csv",
            [_checked_row(), _checked_row(NMR_type="13C NMR")],
        )
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir, allowed_nmr_types=("1H NMR",)))

        summary = loader.preprocess()["test"]

        assert summary["kept"] == 1
        assert summary["dropped_nmr_type"] == 1

    def test_max_records_caps_output(self, raw_dir, processed_dir):
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row() for _ in range(5)])
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir, max_records=2))

        summary = loader.preprocess()["test"]

        assert summary["kept"] == 2
        assert len(loader.load_truth("test")) == 2

    def test_sample_ids_are_stable_across_reprocessing(self, raw_dir, processed_dir):
        _write_checked_csv(
            raw_dir / "test_300_checked.csv",
            [_checked_row(), _checked_row(nmr_solvent_right="wrong"), _checked_row(SMILES="CCN")],
        )
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        ids_before = [record["sample_id"] for record in loader.load_truth("test")]
        loader.preprocess(overwrite=True)
        ids_after = [record["sample_id"] for record in loader.load_truth("test")]

        assert ids_before == ids_after
        assert ids_before == ["NMRexp:test:checked:0", "NMRexp:test:checked:2"]


class TestTruthSplits:
    def test_preprocess_separates_raw_train_from_checked_test(self, raw_dir, processed_dir):
        _write_raw_csv(raw_dir / "raw.csv", [_base_row(SMILES="CCO"), _base_row(SMILES="CCN")])
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row(SMILES="c1ccccc1")])
        config = _config(
            raw_dir,
            processed_dir,
            sources={"raw": "raw.csv", "checked": "test_300_checked.csv"},
            truth_splits={"train": ("raw",), "test": ("checked",)},
        )
        loader = NmrExpDataLoader(config)

        summaries = loader.preprocess()
        train = loader.load_truth("train")
        test = loader.load_truth("test")

        assert set(summaries) == {"train", "test"}
        assert summaries["train"]["kept"] == 2
        assert summaries["train"]["sources"]["raw"]["is_checked"] is False
        assert summaries["test"]["kept"] == 1
        assert summaries["test"]["sources"]["checked"]["is_checked"] is True
        assert len(train) == 2
        assert len(test) == 1
        assert all(record["quality"] is None for record in train)
        assert test[0]["quality"] is not None
        assert {record["provenance"]["split"] for record in train} == {"train"}
        assert {record["provenance"]["split"] for record in test} == {"test"}
        assert (processed_dir / "nmrexp_truth_train.jsonl").exists()
        assert (processed_dir / "nmrexp_truth_test.jsonl").exists()

    def test_preprocess_can_select_one_truth_split(self, raw_dir, processed_dir):
        _write_raw_csv(raw_dir / "raw.csv", [_base_row()])
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row()])
        config = _config(
            raw_dir,
            processed_dir,
            sources={"raw": "raw.csv", "checked": "test_300_checked.csv"},
            truth_splits={"train": ("raw",), "test": ("checked",)},
        )
        loader = NmrExpDataLoader(config)

        summaries = loader.preprocess(["test"])

        assert set(summaries) == {"test"}
        assert summaries["test"]["kept"] == 1
        assert not (processed_dir / "nmrexp_truth_train.jsonl").exists()
        assert (processed_dir / "nmrexp_truth_test.jsonl").exists()


class TestPreprocessCachingAndErrors:
    def test_preprocess_reuses_cache_by_default(self, raw_dir, processed_dir):
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row()])
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        first = loader.preprocess()
        second = loader.preprocess()

        assert first["test"]["status"] == "built"
        assert second["test"]["status"] == "cached"

    def test_preprocess_overwrite_rebuilds(self, raw_dir, processed_dir):
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row()])
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))
        loader.preprocess()

        rebuilt = loader.preprocess(overwrite=True)

        assert rebuilt["test"]["status"] == "built"

    def test_load_auto_preprocesses_when_cache_missing(self, raw_dir, processed_dir):
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row()])
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        dataset = loader.load_truth("test")

        assert len(dataset) == 1

    def test_unknown_split_raises_key_error(self, raw_dir, processed_dir):
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        with pytest.raises(KeyError):
            loader.preprocess(["does-not-exist"])
        with pytest.raises(KeyError):
            loader.load_truth("does-not-exist")

    def test_split_referencing_unknown_source_raises_key_error(self, raw_dir, processed_dir):
        loader = NmrExpDataLoader(
            _config(
                raw_dir,
                processed_dir,
                truth_splits={"test": ("missing-source",)},
            )
        )

        with pytest.raises(KeyError, match="missing-source"):
            loader.preprocess()

    def test_missing_raw_file_raises_file_not_found(self, raw_dir, processed_dir):
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        with pytest.raises(FileNotFoundError):
            loader.load_truth("test")

    def test_missing_base_column_raises_value_error(self, raw_dir, processed_dir):
        import pandas as pd

        path = raw_dir / "test_300_checked.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [c for c in _BASE_COLUMNS + _CHECKED_EXTRA_COLUMNS if c != "NMR_solvent"]
        pd.DataFrame([_checked_row()], columns=columns).to_csv(path, index=False)
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        with pytest.raises(ValueError, match="NMR_solvent"):
            loader.load_truth("test")

    def test_partial_checked_columns_raises_value_error(self, raw_dir, processed_dir):
        import pandas as pd

        path = raw_dir / "test_300_checked.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [c for c in _BASE_COLUMNS + _CHECKED_EXTRA_COLUMNS if c != "is_same_skeleton"]
        pd.DataFrame([_checked_row()], columns=columns).to_csv(path, index=False)
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        with pytest.raises(ValueError, match="is_same_skeleton"):
            loader.load_truth("test")


class TestLoadTruthConvenienceMethods:
    def test_load_truth_is_an_alias_for_load(self, raw_dir, processed_dir):
        _write_checked_csv(raw_dir / "test_300_checked.csv", [_checked_row()])
        loader = NmrExpDataLoader(_config(raw_dir, processed_dir))

        assert [r["sample_id"] for r in loader.load_truth("test")] == [r["sample_id"] for r in loader.load("test")]


@pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow/fastparquet is not installed")
class TestRealParquetSource:
    def test_reads_real_parquet_and_uses_raw_smiles(self, raw_dir, processed_dir):
        import pandas as pd

        raw_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([_base_row(SMILES="CCN(CC)CC")], columns=_BASE_COLUMNS).to_parquet(
            raw_dir / "NMRexp_10to24_1_1004.parquet"
        )
        config = _config(
            raw_dir,
            processed_dir,
            sources={"raw": "NMRexp_10to24_1_1004.parquet"},
            truth_splits={"train": ("raw",)},
        )
        loader = NmrExpDataLoader(config)

        dataset = loader.load()

        assert len(dataset) == 1
        assert dataset[0]["gt_smiles"] == "CCN(CC)CC"
        assert dataset[0]["provenance"]["source_file"] == "NMRexp_10to24_1_1004.parquet"
