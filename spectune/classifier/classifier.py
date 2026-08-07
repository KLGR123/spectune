"""Chemistry-aware clustering for NMRexp dataloader outputs."""

from __future__ import annotations

import json
import tempfile
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from spectune.dataloader import Dataset, JsonlDataset
from spectune.dataloader.base import JsonDict, progress_iter

from .config import ClassifierConfig
from .visualization import plot_clusters

_NMR_PROPERTY_NAMES = (
    "molecular_weight",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "fraction_csp3",
    "formal_charge",
)

_NMR_WORKER_GENERATOR: Any | None = None
_NMR_WORKER_BITS = 0


def _initialize_nmr_worker(radius: int, bits: int, use_chirality: bool) -> None:
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    global _NMR_WORKER_BITS, _NMR_WORKER_GENERATOR
    _NMR_WORKER_BITS = bits
    _NMR_WORKER_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=bits,
        includeChirality=use_chirality,
    )


def _nmr_feature_chunk(chunk: tuple[int, list[str]]) -> tuple[int, Any, Any, Any]:
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    start, smiles_values = chunk
    if _NMR_WORKER_GENERATOR is None:
        raise RuntimeError("NMR feature worker was not initialized")

    n_bytes = (_NMR_WORKER_BITS + 7) // 8
    fingerprints = np.zeros((len(smiles_values), n_bytes), dtype=np.uint8)
    properties = np.zeros((len(smiles_values), len(_NMR_PROPERTY_NAMES)), dtype=np.float32)
    valid = np.zeros(len(smiles_values), dtype=np.bool_)
    for offset, smiles in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        fingerprint = _NMR_WORKER_GENERATOR.GetFingerprint(molecule)
        packed = np.frombuffer(DataStructs.BitVectToBinaryText(fingerprint), dtype=np.uint8)
        fingerprints[offset, : len(packed)] = packed
        properties[offset] = (
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            rdMolDescriptors.CalcNumRings(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            Chem.GetFormalCharge(molecule),
        )
        valid[offset] = True
    return start, fingerprints, properties, valid


def _nmr_smiles_chunks(dataset: Dataset, chunk_size: int) -> Iterable[tuple[int, list[str]]]:
    start = 0
    smiles_values: list[str] = []
    for record in dataset:
        smiles_values.append(str(record.get("gt_smiles") or ""))
        if len(smiles_values) == chunk_size:
            yield start, smiles_values
            start += len(smiles_values)
            smiles_values = []
    if smiles_values:
        yield start, smiles_values


def _bounded_process_results(
    executor: ProcessPoolExecutor,
    chunks: Iterable[tuple[int, list[str]]],
    maximum_pending: int,
) -> Iterable[tuple[int, Any, Any, Any]]:
    iterator = iter(chunks)
    pending: deque[Any] = deque()
    for _ in range(maximum_pending):
        try:
            pending.append(executor.submit(_nmr_feature_chunk, next(iterator)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(_nmr_feature_chunk, next(iterator)))
        except StopIteration:
            pass


class Classifier:
    """Append meaningful cluster labels to NMRexp truth records.

    The source must be a :class:`JsonlDataset` returned by
    :class:`~spectune.dataloader.nmrexp.NmrExpDataLoader`; its JSONL file is
    rewritten atomically with a ``cluster`` field and a freshly indexed
    :class:`JsonlDataset` is returned.

    NMR clustering combines L2-normalized Morgan fingerprints (topological
    structure) with a separately standardized block of interpretable molecular
    properties. This deliberately excludes IDs, filenames, formula strings,
    and other provenance that would produce spurious clusters. MiniBatchKMeans
    keeps the method practical for multi-million-row truth files.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.last_summary: JsonDict | None = None

    def classify(
        self,
        dataset: Dataset,
        *,
        visualize: bool | None = None,
        visualization_path: str | Path | None = None,
    ) -> Dataset:
        """Recluster ``dataset`` and atomically replace ``config.cluster_key``."""
        if not isinstance(dataset, JsonlDataset):
            raise TypeError("Classifier requires a JsonlDataset returned by a spectune dataloader")
        if len(dataset) == 0:
            raise ValueError("cannot cluster an empty dataset")

        self._infer_kind(dataset[0])
        should_visualize = self.config.visualize if visualize is None else visualize
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="spectune_classifier_", dir=output_dir) as temporary_dir:
            result = self._classify_nmr(
                dataset,
                Path(temporary_dir),
                visualize=should_visualize,
                visualization_path=visualization_path,
            )
        return result

    def cluster(self, dataset: Dataset, **kwargs: Any) -> Dataset:
        """Alias for :meth:`classify`."""
        return self.classify(dataset, **kwargs)

    def visualize(
        self,
        dataset: Dataset,
        *,
        visualization_path: str | Path | None = None,
    ) -> Path:
        """Rebuild a visualization from existing cluster labels."""
        if not isinstance(dataset, JsonlDataset):
            raise TypeError("Classifier visualization requires a JsonlDataset")
        if len(dataset) < 2:
            raise ValueError("at least two records are required for cluster visualization")
        output_path = self._visualization_path(dataset, visualization_path)
        if not _all_records_clustered(dataset, self.config.cluster_key):
            raise ValueError("dataset has no complete cluster assignment; call classify() first")

        self._infer_kind(dataset[0])
        sampled = list(
            dataset.sample(
                self.config.visualization_max_points,
                seed=self.config.random_state,
                cluster_key=self.config.cluster_key,
            )
        )
        features, records = self._nmr_visual_features(sampled)
        labels = [int(record[self.config.cluster_key]) for record in records]
        representatives = _record_representatives(records, features, labels, "gt_smiles")
        return plot_clusters(
            features,
            labels,
            representatives,
            output_path,
            kind="nmr",
            dpi=self.config.visualization_dpi,
            random_state=self.config.random_state,
            dim=self.config.visualization_dim,
        )

    @staticmethod
    def _infer_kind(record: JsonDict) -> None:
        if not record.get("gt_smiles"):
            raise ValueError("unsupported record schema: expected NMRexp 'gt_smiles'")

    def _classify_nmr(
        self,
        dataset: JsonlDataset,
        work_dir: Path,
        *,
        visualize: bool,
        visualization_path: str | Path | None,
    ) -> JsonlDataset:
        np, MiniBatchKMeans, StandardScaler = _clustering_dependencies()
        prefix = work_dir / dataset.path.stem
        fingerprint_path = prefix.with_suffix(".fingerprints.npy")
        properties_path = prefix.with_suffix(".properties.npy")
        valid_path = prefix.with_suffix(".valid.npy")
        self._build_nmr_representation(dataset, fingerprint_path, properties_path, valid_path)

        fingerprints = np.load(fingerprint_path, mmap_mode="r")
        properties = np.load(properties_path, mmap_mode="r")
        valid = np.load(valid_path, mmap_mode="r")
        valid_count = int(valid.sum())
        if valid_count == 0:
            raise ValueError("no NMRexp SMILES could be parsed by RDKit")
        effective_clusters = min(self.config.nmr_n_clusters, valid_count)
        if effective_clusters < 1:
            raise ValueError("nmr_n_clusters must be at least 1")

        labels_path = prefix.with_suffix(".labels.npy")

        scaler = StandardScaler()
        for indices in _valid_index_batches(valid, self.config.batch_size):
            scaler.partial_fit(properties[indices])

        kmeans = MiniBatchKMeans(
            n_clusters=effective_clusters,
            random_state=self.config.random_state,
            batch_size=max(self.config.batch_size, effective_clusters),
            n_init="auto",
        )
        for _ in range(max(self.config.kmeans_epochs, 1)):
            for indices in _valid_index_batches(valid, max(self.config.batch_size, effective_clusters)):
                features = self._nmr_feature_batch(fingerprints[indices], properties[indices], scaler)
                kmeans.partial_fit(features)

        labels_memmap = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype=np.int32,
            shape=(len(dataset),),
        )
        labels_memmap[:] = -1
        for indices in _valid_index_batches(valid, self.config.batch_size):
            features = self._nmr_feature_batch(fingerprints[indices], properties[indices], scaler)
            labels_memmap[indices] = kmeans.predict(features)
        labels_memmap.flush()
        del labels_memmap
        labels = np.load(labels_path, mmap_mode="r")

        visualization = None
        if visualize and valid_count >= 2:
            sample_indices = _balanced_indices(labels, self.config.visualization_max_points, self.config.random_state)
            sample_features = self._nmr_feature_batch(
                fingerprints[sample_indices],
                properties[sample_indices],
                scaler,
            )
            sample_labels = np.asarray(labels[sample_indices], dtype=np.int32)
            representatives = _nmr_representatives(dataset, sample_indices, sample_features, sample_labels)
            visualization = plot_clusters(
                sample_features,
                sample_labels,
                representatives,
                self._visualization_path(dataset, visualization_path),
                kind="nmr",
                dpi=self.config.visualization_dpi,
                random_state=self.config.random_state,
                dim=self.config.visualization_dim,
            )
        _rewrite_jsonl_clusters(dataset.path, labels, self.config.cluster_key)

        self.last_summary = {
            "status": "built",
            "kind": "nmr",
            "records": len(dataset),
            "valid_records": valid_count,
            "clusters": effective_clusters,
            "structure_representation": {
                "type": "Morgan fingerprint",
                "radius": self.config.fingerprint_radius,
                "bits": self.config.fingerprint_bits,
                "use_chirality": self.config.fingerprint_use_chirality,
                "workers": self.config.nmr_num_workers,
            },
            "property_representation": list(_NMR_PROPERTY_NAMES),
            "output_path": str(dataset.path),
            "visualization_path": str(visualization) if visualization else None,
        }
        return JsonlDataset(dataset.path)

    def _build_nmr_representation(
        self,
        dataset: JsonlDataset,
        fingerprint_path: Path,
        properties_path: Path,
        valid_path: Path,
    ) -> None:
        try:
            import numpy as np
            import rdkit  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NMR clustering requires RDKit and numpy; install spectune[classifier]") from exc

        n_bytes = (self.config.fingerprint_bits + 7) // 8
        fingerprints = np.lib.format.open_memmap(
            fingerprint_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(dataset), n_bytes),
        )
        properties = np.lib.format.open_memmap(
            properties_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), len(_NMR_PROPERTY_NAMES)),
        )
        valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=(len(dataset),))
        chunk_size = self.config.nmr_worker_chunk_size
        total_chunks = (len(dataset) + chunk_size - 1) // chunk_size
        chunks = _nmr_smiles_chunks(dataset, chunk_size)
        initializer_args = (
            self.config.fingerprint_radius,
            self.config.fingerprint_bits,
            self.config.fingerprint_use_chirality,
        )

        if self.config.nmr_num_workers == 1:
            _initialize_nmr_worker(*initializer_args)
            results: Iterable[tuple[int, Any, Any, Any]] = map(_nmr_feature_chunk, chunks)
            self._store_nmr_feature_chunks(results, total_chunks, fingerprints, properties, valid)
        else:
            with ProcessPoolExecutor(
                max_workers=self.config.nmr_num_workers,
                initializer=_initialize_nmr_worker,
                initargs=initializer_args,
            ) as executor:
                results = _bounded_process_results(
                    executor,
                    chunks,
                    maximum_pending=self.config.nmr_num_workers * 2,
                )
                self._store_nmr_feature_chunks(results, total_chunks, fingerprints, properties, valid)
        fingerprints.flush()
        properties.flush()
        valid.flush()

    def _store_nmr_feature_chunks(
        self,
        results: Iterable[tuple[int, Any, Any, Any]],
        total_chunks: int,
        fingerprints: Any,
        properties: Any,
        valid: Any,
    ) -> None:
        chunks = progress_iter(
            results,
            total=total_chunks,
            label=f"Classifier/NMR representation ({self.config.nmr_num_workers} workers, chunks)",
            enabled=self.config.show_progress,
        )
        for start, chunk_fingerprints, chunk_properties, chunk_valid in chunks:
            stop = start + len(chunk_valid)
            fingerprints[start:stop] = chunk_fingerprints
            properties[start:stop] = chunk_properties
            valid[start:stop] = chunk_valid

    def _nmr_feature_batch(self, packed_fingerprints: Any, properties: Any, scaler: Any) -> Any:
        import numpy as np

        structural = np.unpackbits(
            np.asarray(packed_fingerprints, dtype=np.uint8),
            axis=1,
            count=self.config.fingerprint_bits,
        ).astype(np.float32)
        norms = np.linalg.norm(structural, axis=1, keepdims=True)
        structural /= np.maximum(norms, 1.0)
        property_features = scaler.transform(properties).astype(np.float32)
        property_features *= self.config.property_weight
        return np.concatenate((structural, property_features), axis=1)

    def _nmr_visual_features(self, records: Sequence[JsonDict]) -> tuple[Any, list[JsonDict]]:
        try:
            import numpy as np
            from rdkit import Chem, DataStructs
            from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "NMR visualization requires RDKit, numpy, and scikit-learn; install spectune[classifier]"
            ) from exc

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.config.fingerprint_radius,
            fpSize=self.config.fingerprint_bits,
            includeChirality=self.config.fingerprint_use_chirality,
        )
        fingerprints: list[Any] = []
        properties: list[tuple[float, ...]] = []
        valid_records: list[JsonDict] = []
        for record in records:
            molecule = Chem.MolFromSmiles(str(record.get("gt_smiles") or ""))
            if molecule is None:
                continue
            fingerprint = generator.GetFingerprint(molecule)
            unpacked = np.zeros(self.config.fingerprint_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fingerprint, unpacked)
            fingerprints.append(unpacked)
            properties.append(
                (
                    Descriptors.MolWt(molecule),
                    Crippen.MolLogP(molecule),
                    rdMolDescriptors.CalcTPSA(molecule),
                    Lipinski.NumHDonors(molecule),
                    Lipinski.NumHAcceptors(molecule),
                    Lipinski.NumRotatableBonds(molecule),
                    rdMolDescriptors.CalcNumRings(molecule),
                    rdMolDescriptors.CalcFractionCSP3(molecule),
                    Chem.GetFormalCharge(molecule),
                )
            )
            valid_records.append(record)
        if len(valid_records) < 2:
            raise ValueError("fewer than two sampled NMR records could be parsed by RDKit")
        structural = np.asarray(fingerprints, dtype=np.float32)
        structural /= np.maximum(np.linalg.norm(structural, axis=1, keepdims=True), 1.0)
        property_features = StandardScaler().fit_transform(properties).astype(np.float32)
        property_features *= self.config.property_weight
        return np.concatenate((structural, property_features), axis=1), valid_records

    def _visualization_path(
        self,
        dataset: JsonlDataset,
        requested_path: str | Path | None,
    ) -> Path:
        if requested_path is not None:
            return Path(requested_path)
        return Path(self.config.output_dir) / f"{dataset.path.stem}_clusters.png"


def _clustering_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("clustering requires numpy and scikit-learn; install spectune[classifier]") from exc
    return np, MiniBatchKMeans, StandardScaler



def _all_records_clustered(dataset: Dataset, cluster_key: str) -> bool:
    return all(cluster_key in record and record[cluster_key] is not None for record in dataset)


def _valid_index_batches(valid: Any, batch_size: int) -> Iterable[Any]:
    import numpy as np

    valid_indices = np.flatnonzero(valid)
    for start in range(0, len(valid_indices), batch_size):
        yield valid_indices[start : start + batch_size]


def _balanced_indices(labels: Any, maximum: int, seed: int) -> Any:
    import numpy as np

    labels_array = np.asarray(labels)
    valid_labels = sorted(int(label) for label in np.unique(labels_array) if label >= 0)
    if not valid_labels:
        return np.array([], dtype=np.int64)
    if len(labels_array) <= maximum:
        return np.arange(len(labels_array), dtype=np.int64)

    random = np.random.default_rng(seed)
    quota = max(maximum // len(valid_labels), 1)
    selected: list[Any] = []
    remaining: list[Any] = []
    for label in valid_labels:
        indices = np.flatnonzero(labels_array == label)
        random.shuffle(indices)
        selected.extend(indices[:quota])
        remaining.extend(indices[quota:])
    if len(selected) < maximum and remaining:
        remaining_array = np.asarray(remaining)
        random.shuffle(remaining_array)
        selected.extend(remaining_array[: maximum - len(selected)])
    return np.asarray(selected[:maximum], dtype=np.int64)


def _nmr_representatives(
    dataset: JsonlDataset,
    indices: Any,
    features: Any,
    labels: Any,
) -> dict[int, str]:
    import numpy as np

    representatives: dict[int, str] = {}
    for label in sorted(int(value) for value in np.unique(labels)):
        positions = np.flatnonzero(labels == label)
        cluster_features = features[positions]
        center = cluster_features.mean(axis=0)
        representative_position = positions[int(np.argmin(np.linalg.norm(cluster_features - center, axis=1)))]
        record = dataset[int(indices[representative_position])]
        representatives[label] = str(record.get("gt_smiles") or "")
    return representatives



def _record_representatives(
    records: Sequence[JsonDict],
    features: Any,
    labels: Any,
    value_key: str,
) -> dict[int, str]:
    import numpy as np

    labels = np.asarray(labels, dtype=np.int32)
    representatives: dict[int, str] = {}
    for label in sorted(int(value) for value in np.unique(labels)):
        positions = np.flatnonzero(labels == label)
        cluster_features = features[positions]
        center = cluster_features.mean(axis=0)
        representative_position = positions[int(np.argmin(np.linalg.norm(cluster_features - center, axis=1)))]
        representatives[label] = str(records[int(representative_position)].get(value_key) or "")
    return representatives


def _rewrite_jsonl_clusters(path: Path, labels: Sequence[int], cluster_key: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".cluster.tmp")
    count = 0
    with path.open("r", encoding="utf-8") as source, tmp_path.open("w", encoding="utf-8") as target:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            if count >= len(labels):
                tmp_path.unlink(missing_ok=True)
                raise ValueError(f"JSONL row count exceeds cluster label count {len(labels)}")
            record = json.loads(stripped)
            record[cluster_key] = int(labels[count])
            target.write(json.dumps(record, ensure_ascii=False))
            target.write("\n")
            count += 1
    if count != len(labels):
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"cluster label count {len(labels)} does not match JSONL row count {count}")
    tmp_path.replace(path)


__all__ = ["Classifier"]
