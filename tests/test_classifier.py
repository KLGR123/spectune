import importlib.util

import pytest

from spectune import Classifier, ClassifierConfig, InMemoryDataset, JsonlDataset
from spectune.dataloader import write_jsonl

_HAS_CLUSTER_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "sklearn"))
_HAS_CHEM_DEPS = _HAS_CLUSTER_DEPS and importlib.util.find_spec("rdkit") is not None
_HAS_VIS_DEPS = _HAS_CHEM_DEPS and importlib.util.find_spec("matplotlib") is not None


def _query_records():
    return [
        {
            "sample_id": "SpecXMaster:a:0",
            "query": "1H NMR spectrum for an alcohol",
            "modality": "nmr",
            "provenance": {"conversation_id": "a", "turn_index": 0},
        },
        {
            "sample_id": "SpecXMaster:a:1",
            "query": "continue with the carbon spectrum",
            "modality": "nmr",
            "provenance": {"conversation_id": "a", "turn_index": 1},
        },
        {
            "sample_id": "SpecXMaster:b:0",
            "query": "interpret an aromatic ketone",
            "modality": "nmr",
            "provenance": {"conversation_id": "b", "turn_index": 0},
        },
        {
            "sample_id": "SpecXMaster:b:1",
            "query": "the molecular formula is C8H8O",
            "modality": "nmr",
            "provenance": {"conversation_id": "b", "turn_index": 1},
        },
    ]


class _FakeTextClassifier(Classifier):
    def _encode_texts(self, texts):
        import numpy as np

        assert "[Next user turn]" in texts[0]
        assert len(texts) == 2
        return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


@pytest.mark.skipif(not _HAS_CLUSTER_DEPS, reason="numpy/scikit-learn are not installed")
class TestQueryClassifier:
    def test_multiturn_conversations_receive_one_shared_cluster(self, tmp_path):
        path = tmp_path / "queries.jsonl"
        write_jsonl(path, _query_records())
        config = ClassifierConfig(query_n_clusters=2, output_dir=str(tmp_path / "outputs"), visualize=False)
        classifier = _FakeTextClassifier(config)

        clustered = classifier.classify(JsonlDataset(path))

        assert len(clustered) == 4
        assert clustered[0]["cluster"] == clustered[1]["cluster"]
        assert clustered[2]["cluster"] == clustered[3]["cluster"]
        assert clustered[0]["cluster"] != clustered[2]["cluster"]
        assert classifier.last_summary["conversations"] == 2

    def test_existing_cluster_values_are_ignored_and_replaced(self, tmp_path):
        path = tmp_path / "queries.jsonl"
        write_jsonl(path, [{**record, "cluster": 99} for record in _query_records()])
        classifier = _FakeTextClassifier(
            ClassifierConfig(query_n_clusters=2, output_dir=str(tmp_path / "outputs"), visualize=False)
        )

        reclustered = classifier.classify(JsonlDataset(path))

        assert {record["cluster"] for record in reclustered} == {0, 1}
        assert classifier.last_summary["status"] == "built"


@pytest.mark.skipif(not _HAS_CHEM_DEPS, reason="RDKit/numpy/scikit-learn are not installed")
class TestNmrClassifier:
    def test_appends_structure_and_property_clusters(self, tmp_path):
        path = tmp_path / "truth.jsonl"
        records = [
            {"sample_id": "nmr:0", "gt_smiles": "CCO", "modality": "nmr"},
            {"sample_id": "nmr:1", "gt_smiles": "CCCO", "modality": "nmr"},
            {"sample_id": "nmr:2", "gt_smiles": "c1ccccc1", "modality": "nmr"},
            {"sample_id": "nmr:3", "gt_smiles": "Cc1ccccc1", "modality": "nmr"},
        ]
        write_jsonl(path, records)
        config = ClassifierConfig(
            nmr_n_clusters=2,
            nmr_num_workers=2,
            nmr_worker_chunk_size=2,
            batch_size=4,
            kmeans_epochs=1,
            output_dir=str(tmp_path / "outputs"),
            visualize=False,
        )
        classifier = Classifier(config)

        clustered = classifier.classify(JsonlDataset(path))

        assert len(clustered) == 4
        assert all(isinstance(record["cluster"], int) for record in clustered)
        assert len({record["cluster"] for record in clustered}) == 2
        assert classifier.last_summary["valid_records"] == 4

    @pytest.mark.skipif(not _HAS_VIS_DEPS, reason="matplotlib is not installed")
    @pytest.mark.parametrize("dim", (2, 3))
    def test_writes_400_dpi_visualization(self, tmp_path, dim):
        from PIL import Image

        path = tmp_path / "truth.jsonl"
        write_jsonl(
            path,
            [
                {"sample_id": "nmr:0", "gt_smiles": "CCO"},
                {"sample_id": "nmr:1", "gt_smiles": "CCCO"},
                {"sample_id": "nmr:2", "gt_smiles": "c1ccccc1"},
                {"sample_id": "nmr:3", "gt_smiles": "Cc1ccccc1"},
            ],
        )
        image_path = tmp_path / "custom" / f"clusters-{dim}d.png"
        classifier = Classifier(
            ClassifierConfig(
                nmr_n_clusters=2,
                nmr_num_workers=1,
                batch_size=4,
                kmeans_epochs=1,
                output_dir=str(tmp_path / "outputs"),
                visualization_dpi=400,
                visualization_dim=dim,
            )
        )

        classifier.classify(JsonlDataset(path), visualization_path=image_path)

        assert image_path.exists()
        with Image.open(image_path) as image:
            assert image.info["dpi"][0] >= 399


class TestClassifierValidation:
    def test_requires_jsonl_dataset(self):
        classifier = Classifier(ClassifierConfig(visualize=False))

        with pytest.raises(TypeError, match="JsonlDataset"):
            classifier.classify(InMemoryDataset([{"query": "test"}]))

    def test_rejects_unknown_record_schema(self, tmp_path):
        path = tmp_path / "unknown.jsonl"
        write_jsonl(path, [{"sample_id": "unknown:0"}])
        classifier = Classifier(ClassifierConfig(visualize=False))

        with pytest.raises(ValueError, match="unsupported record schema"):
            classifier.classify(JsonlDataset(path))
