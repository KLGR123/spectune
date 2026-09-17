"""Local reaction-precedent search via canonical reactant-set overlap.

This is deterministic set-overlap scoring against flat local reaction tables
(USPTO, ChemPile-lift, Pistachio), not embedding/vector retrieval.
"""

from __future__ import annotations

import asyncio
import csv
import functools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import JsonDict, Tool, ToolResult
from .config import ReactionLocalIndexSearchConfig
from .utils import as_list, has_rdkit, molecular_formula, strict_canonical_smiles


@dataclass(frozen=True)
class IndexedReactionData:
    """Records with inverted index for fast candidate retrieval."""

    # Sequence rather than tuple[JsonDict, ...]: the prebuilt-parquet loader uses a
    # lazily-materializing columnar view (see _ColumnarRecords) so multi-million-row
    # corpora don't pay for a per-record dict upfront.
    records: Sequence[JsonDict]
    # inverted index: reactant_component -> list of record indices
    component_to_records: dict[str, list[int]]


_PREBUILT_SCALAR_COLUMNS = (
    "source",
    "source_id",
    "product",
    "canonical_product",
    "product_formula",
    "patent_id",
    "reaction_class_name",
)


class _ColumnarRecords(Sequence[JsonDict]):
    """Lazily builds per-row record dicts from column-oriented parquet data.

    ``DataFrame.to_dict(orient="records")`` takes tens of seconds at
    multi-million-row corpus scale (the whole point of the prebuilt index is to
    make full-corpus loading fast), so record dicts are only materialized for
    indices the scoring loop actually touches -- a small fraction of the corpus.
    """

    __slots__ = ("_reactant_components", "_scalar_columns", "_length")

    def __init__(self, *, reactant_components: list[Any], scalar_columns: dict[str, list[Any]]) -> None:
        self._reactant_components = reactant_components
        self._scalar_columns = scalar_columns
        self._length = len(reactant_components)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> JsonDict:  # type: ignore[override]
        columns = self._scalar_columns
        record: JsonDict = {
            "source": columns["source"][idx],
            "source_id": columns["source_id"][idx],
            "reactant_components": list(self._reactant_components[idx]),
            "product": columns["product"][idx],
            "canonical_product": columns["canonical_product"][idx],
            "product_formula": columns["product_formula"][idx],
        }
        patent_id = columns["patent_id"][idx]
        if patent_id:
            record["patent_id"] = patent_id
        reaction_class_name = columns["reaction_class_name"][idx]
        if reaction_class_name:
            record["reaction_class_name"] = reaction_class_name
        return record


_REACTION_SMILES_RE = re.compile(r"([^\s]+>>[^\s]+)")

# Common functional-group SMARTS used for post-hoc structural constraint checks.
# This is standard reference chemistry, not a per-example answer key.
_FUNCTIONAL_GROUP_SMARTS: dict[str, str] = {
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=O)[#6]",
    "phenyl": "c1ccccc1",
    "aromatic ring": "a1aaaaa1",
    "alcohol": "[OX2H]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "halide": "[F,Cl,Br,I]",
    "nitrile": "[CX2]#N",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfonyl": "S(=O)(=O)",
    "boronic acid": "B(O)O",
}


class ReactionLocalIndexSearchTool(Tool):
    name = "reaction_local_index_search"
    description = (
        "在本地反应先例表（USPTO / ChemPile-lift / Pistachio）中搜索与查询共享反应物的反应产物，"
        "通过标准化反应物集合的重合度打分。不读取 NMR / MS 谱图。"
    )

    def __init__(self, config: ReactionLocalIndexSearchConfig | None = None) -> None:
        self.config = config or ReactionLocalIndexSearchConfig()

    @property
    def parameters(self) -> JsonDict:
        return {
            "type": "object",
            "properties": {
                "reaction_smiles": {
                    "type": "string",
                    "description": "可选的反应 SMILES，格式为 'reactants>>products' 或 'reactants>agents>products'",
                },
                "reactants": {"type": "array", "items": {"type": "string"}, "description": "反应物 SMILES"},
                "reagents": {"type": "array", "items": {"type": "string"}, "description": "可选的试剂 SMILES"},
                "conditions": {
                    "type": "string",
                    "description": "自由文本形式的反应条件，仅作为溯源信息保留（不参与打分）",
                },
                "reaction_type": {
                    "type": "string",
                    "description": "自由文本形式的反应类型提示，仅作为溯源信息保留（不参与打分）",
                },
                "target_formula": {"type": "string", "description": "可选的目标产物分子式"},
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选的官能团提示，用于核对候选产物",
                },
                "topk": {"type": "integer", "minimum": 1, "maximum": 100, "default": self.config.default_topk},
                "sources": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["uspto", "chempile", "pistachio"]},
                    "description": "限定使用已配置本地数据源的一个子集（默认使用全部已配置的数据源）",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        return await asyncio.to_thread(_execute_sync, self.config, dict(arguments or {}))


def _execute_sync(config: ReactionLocalIndexSearchConfig, args: JsonDict) -> ToolResult:
    if not has_rdkit():
        return ToolResult(
            completion="failure",
            status="unavailable",
            warnings=["rdkit is not installed; install spectune[chem] to enable reaction_local_index_search"],
        )

    requested_sources = {str(s).strip().lower() for s in as_list(args.get("sources")) if str(s).strip()}

    available: list[tuple[str, str, str]] = []
    for source_name, raw_path in (
        ("uspto", config.uspto_csv_path),
        ("chempile", config.chempile_parquet_path),
        ("pistachio", config.pistachio_smi_path),
    ):
        if requested_sources and source_name not in requested_sources:
            continue
        prebuilt_path = _prebuilt_source_path(config.prebuilt_index_dir, source_name)
        if raw_path or prebuilt_path:
            available.append((source_name, raw_path, prebuilt_path))

    if not available:
        return ToolResult(
            completion="failure",
            status="unavailable",
            warnings=[
                "no local reaction index is configured; set RXN_LOCAL_INDEX_PREBUILT_DIR (preferred) and/or "
                "RXN_LOCAL_INDEX_USPTO_CSV, RXN_LOCAL_INDEX_CHEMPILE_PARQUET, RXN_LOCAL_INDEX_PISTACHIO_SMI"
            ],
        )

    reaction_smiles = str(args.get("reaction_smiles") or "").strip()
    rxn_reactants, rxn_agents, _rxn_products = _split_reaction_smiles(reaction_smiles)
    reactants = [str(x) for x in as_list(args.get("reactants"))] or rxn_reactants
    reagents = [str(x) for x in as_list(args.get("reagents"))] or rxn_agents
    constraints = [str(x) for x in as_list(args.get("constraints")) if str(x).strip()]
    target_formula = str(args.get("target_formula") or "").strip()
    conditions = str(args.get("conditions") or "").strip()
    reaction_type = str(args.get("reaction_type") or "").strip()
    topk = max(1, min(int(args.get("topk") or config.default_topk), 100))

    query_components, invalid_components = _canonical_components(reactants + reagents)
    warnings: list[str] = []
    if invalid_components:
        warnings.append("some_query_reactants_failed_smiles_parsing")
    if conditions or reaction_type:
        warnings.append("local_lookup_does_not_use_conditions_or_reaction_type")
    if not query_components:
        return ToolResult(
            completion="failure",
            status="error",
            warnings=[*warnings, "no valid reactant/reagent SMILES provided"],
        )

    indexed_datasets: list[IndexedReactionData] = []
    sources_used: list[str] = []
    source_errors: list[str] = []
    for source_name, path, prebuilt_path in available:
        try:
            if prebuilt_path:
                loaded = _load_prebuilt_parquet(prebuilt_path)
            elif source_name == "uspto":
                loaded = _load_uspto_csv(path)
            elif source_name == "chempile":
                loaded = _load_chempile_parquet(path, max_records=config.max_chempile_records)
            else:
                loaded = _load_pistachio_smi(path, max_records=config.max_pistachio_records)
        except Exception as exc:
            source_errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
            continue
        if not prebuilt_path:
            warnings.append(
                f"{source_name}: no prebuilt index configured; using slow, capped raw parsing -- run "
                "`python -m spectune.tools build-reaction-index` and set RXN_LOCAL_INDEX_PREBUILT_DIR "
                "for full coverage and fast loading"
            )
        if loaded:
            indexed_datasets.append(loaded)
            sources_used.append(source_name)

    if not indexed_datasets:
        return ToolResult(
            completion="success" if not source_errors else "partial",
            status="no_candidates",
            data={"provenance": {"sources_configured": [s for s, _, _ in available], "sources_loaded": []}},
            warnings=[*warnings, *source_errors, "no records loaded from any configured local source"],
        )

    query_set = set(query_components)
    best_by_product: dict[str, JsonDict] = {}
    total_records = 0

    # Use inverted index to find candidate records efficiently
    for dataset in indexed_datasets:
        total_records += len(dataset.records)
        candidate_indices: set[int] = set()

        # Gather all record indices that match any query component
        for component in query_components:
            if component in dataset.component_to_records:
                candidate_indices.update(dataset.component_to_records[component])

        # Only score candidate records that have at least one matching component
        for idx in candidate_indices:
            record = dataset.records[idx]
            candidate = _score_record(record, query_set, constraints=constraints, target_formula=target_formula)
            if candidate is None:
                continue
            product = candidate["canonical_smiles"]
            previous = best_by_product.get(product)
            if previous is None or candidate["score"] > previous["score"]:
                best_by_product[product] = candidate

    ranked = sorted(
        best_by_product.values(),
        key=lambda item: (item["score"], item["reaction_evidence"]["reactant_coverage"]),
        reverse=True,
    )[:topk]
    for rank, candidate in enumerate(ranked, 1):
        candidate["rank"] = rank

    return ToolResult(
        completion="success" if not source_errors else "partial",
        status="ok" if ranked else "no_candidates",
        data={
            "candidates": ranked,
            "provenance": {
                "sources_configured": [s for s, _, _ in available],
                "sources_loaded": sources_used,
                "num_records_indexed": total_records,
                "canonical_query_components": query_components,
            },
        },
        warnings=[*warnings, *source_errors],
    )


def _prebuilt_source_path(prebuilt_index_dir: str, source_name: str) -> str:
    """Return the prebuilt parquet path for *source_name* if it exists, else ``""``."""
    if not prebuilt_index_dir:
        return ""
    candidate = Path(prebuilt_index_dir) / f"{source_name}.parquet"
    return str(candidate) if candidate.exists() else ""


@functools.lru_cache(maxsize=8)
def _load_prebuilt_parquet(path: str) -> IndexedReactionData:
    """Load a prebuilt index parquet produced by ``python -m spectune.tools build-reaction-index``.

    Columns are already canonicalized ahead of time, so this skips RDKit entirely. Record dicts
    are built lazily via ``_ColumnarRecords`` (see its docstring) instead of eagerly via
    ``DataFrame.to_dict``, which is the dominant cost at full-corpus (multi-million-row) scale.
    ``build_reaction_index`` never writes rows with empty ``reactant_components``, so no
    empty-row filtering is needed here -- such rows simply never enter the inverted index.
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        return IndexedReactionData(records=(), component_to_records={})
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read the prebuilt index; install spectune[reaction]") from exc

    frame = pd.read_parquet(parquet_path)
    reactant_components: list[Any] = frame["reactant_components"].tolist()
    scalar_columns = {name: frame[name].tolist() for name in _PREBUILT_SCALAR_COLUMNS}

    component_to_records: dict[str, list[int]] = {}
    for idx, components in enumerate(reactant_components):
        for component in components:
            component_to_records.setdefault(component, []).append(idx)

    records = _ColumnarRecords(reactant_components=reactant_components, scalar_columns=scalar_columns)
    return IndexedReactionData(records=records, component_to_records=component_to_records)


def _split_reaction_smiles(reaction_smiles: str) -> tuple[list[str], list[str], list[str]]:
    text = str(reaction_smiles or "").strip()
    if not text:
        return [], [], []
    if ">>" in text:
        left, right = text.split(">>", 1)
        return _split_mixture(left), [], _split_mixture(right)
    parts = text.split(">")
    if len(parts) == 3:
        return _split_mixture(parts[0]), _split_mixture(parts[1]), _split_mixture(parts[2])
    return [], [], []


def _split_mixture(value: str) -> list[str]:
    return [token.strip() for token in str(value or "").split(".") if token.strip()]


def _canonical_components(values: Sequence[str]) -> tuple[list[str], list[str]]:
    canonical: list[str] = []
    invalid: list[str] = []
    for raw in values:
        for token in _split_mixture(raw):
            canon = strict_canonical_smiles(token)
            if canon:
                canonical.append(canon)
            else:
                invalid.append(token)
    return sorted(set(canonical)), invalid


def _record(
    *,
    source: str,
    source_id: str,
    reactant_mix: str,
    product_raw: str,
    metadata: JsonDict | None = None,
) -> JsonDict | None:
    product_canon = strict_canonical_smiles(product_raw)
    if not product_canon:
        return None
    components, _invalid = _canonical_components([reactant_mix])
    if not components:
        return None
    record: JsonDict = {
        "source": source,
        "source_id": source_id,
        "reactant_components": components,
        "product": product_raw,
        "canonical_product": product_canon,
        "product_formula": molecular_formula(product_canon),
    }
    if metadata:
        record.update(metadata)
    return record


def _build_index(records: list[JsonDict]) -> IndexedReactionData:
    component_to_records: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        for component in record.get("reactant_components") or []:
            component_to_records.setdefault(component, []).append(i)
    return IndexedReactionData(records=tuple(records), component_to_records=component_to_records)


@functools.lru_cache(maxsize=8)
def _load_uspto_csv(path: str) -> IndexedReactionData:
    """Load the ChemLLMBench-style USPTO CSV (columns: reactant, product)."""
    csv_path = Path(path)
    if not csv_path.exists():
        return IndexedReactionData(records=(), component_to_records={})
    records: list[JsonDict] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_id, row in enumerate(reader):
            record = _record(
                source="uspto_csv",
                source_id=f"uspto_csv:{row_id}",
                reactant_mix=str(row.get("reactant") or ""),
                product_raw=str(row.get("product") or ""),
            )
            if record is not None:
                records.append(record)
    return _build_index(records)


@functools.lru_cache(maxsize=8)
def _load_chempile_parquet(path: str, *, max_records: int) -> IndexedReactionData:
    """Load ChemPile-lift USPTO rows by extracting embedded 'reactants>>products' text."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        return IndexedReactionData(records=(), component_to_records={})
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read the ChemPile parquet; install spectune[reaction]") from exc

    frame = pd.read_parquet(parquet_path, columns=["text"])
    if max_records > 0:
        frame = frame.head(int(max_records))

    records: list[JsonDict] = []
    for row_id, text in enumerate(frame["text"].fillna("").astype(str)):
        match = _REACTION_SMILES_RE.search(text)
        if not match:
            continue
        reaction_smiles = match.group(1).strip().rstrip(".,;:")
        if ">>" not in reaction_smiles:
            continue
        reactant_mix, product_mix = reaction_smiles.split(">>", 1)
        products = _split_mixture(product_mix)
        if not products:
            continue
        record = _record(
            source="chempile_lift_uspto",
            source_id=f"chempile_lift_uspto:{row_id}",
            reactant_mix=reactant_mix,
            product_raw=products[0],
        )
        if record is not None:
            records.append(record)
    return _build_index(records)


@functools.lru_cache(maxsize=8)
def _load_pistachio_smi(path: str, *, max_records: int) -> IndexedReactionData:
    """Load a bounded prefix of a Pistachio-style tab-separated reaction SMILES file."""
    smi_path = Path(path)
    if not smi_path.exists():
        return IndexedReactionData(records=(), component_to_records={})
    records: list[JsonDict] = []
    with smi_path.open(encoding="utf-8", errors="replace") as handle:
        for row_id, line in enumerate(handle):
            if max_records and row_id >= int(max_records):
                break
            fields = line.rstrip("\n").split("\t")
            first_field = fields[0].strip() if fields else ""
            reaction_smiles = first_field.split(None, 1)[0] if first_field else ""
            if ">>" not in reaction_smiles:
                continue
            reactant_mix, product_mix = reaction_smiles.split(">>", 1)
            products = _split_mixture(product_mix)
            if not products:
                continue
            metadata = {
                "patent_id": fields[1] if len(fields) > 1 else "",
                "reaction_class_name": fields[4] if len(fields) > 4 else "",
            }
            record = _record(
                source="pistachio_smi",
                source_id=f"pistachio_smi:{row_id}",
                reactant_mix=reactant_mix,
                product_raw=products[0],
                metadata=metadata,
            )
            if record is not None:
                records.append(record)
    return _build_index(records)


def _constraint_evidence(smiles: str, formula: str, constraints: Sequence[str], target_formula: str) -> JsonDict:
    matched: list[str] = []
    missing: list[str] = []
    if constraints:
        from rdkit import Chem  # type: ignore[import-not-found]

        molecule = Chem.MolFromSmiles(smiles)
        for constraint in constraints:
            key = constraint.lower()
            smarts = next((pattern for name, pattern in _FUNCTIONAL_GROUP_SMARTS.items() if name in key), None)
            pattern = Chem.MolFromSmarts(smarts) if smarts else None
            if molecule is not None and pattern is not None and molecule.HasSubstructMatch(pattern):
                matched.append(constraint)
            else:
                missing.append(constraint)
    formula_match = formula == target_formula.strip() if target_formula.strip() else None
    return {
        "formula": formula,
        "formula_match": formula_match,
        "matched_constraints": matched,
        "missing_constraints": missing,
    }


def _score_record(
    record: JsonDict,
    query_set: set[str],
    *,
    constraints: Sequence[str],
    target_formula: str,
) -> JsonDict | None:
    record_components = set(record.get("reactant_components") or [])
    matched = sorted(query_set & record_components)
    if not matched:
        return None
    missing = sorted(query_set - record_components)
    coverage = len(matched) / len(query_set) if query_set else 0.0
    record_coverage = len(matched) / len(record_components) if record_components else 0.0
    union = len(query_set | record_components)
    jaccard = len(matched) / union if union else 0.0

    constraint_evidence = _constraint_evidence(
        record["canonical_product"], record.get("product_formula") or "", constraints, target_formula
    )
    score = 0.50 * coverage + 0.30 * jaccard + 0.12 * record_coverage
    if constraint_evidence["formula_match"] is True:
        score += 0.08
    elif constraint_evidence["formula_match"] is False:
        score -= 0.20
    checked = len(constraint_evidence["matched_constraints"]) + len(constraint_evidence["missing_constraints"])
    if checked:
        score += 0.08 * (len(constraint_evidence["matched_constraints"]) / checked)
    score = max(0.0, min(1.0, score))

    candidate_warnings: list[str] = []
    if constraint_evidence["formula_match"] is False:
        candidate_warnings.append("target_formula_mismatch")
    if constraint_evidence["missing_constraints"]:
        candidate_warnings.append("missing_structural_constraints")

    reaction_evidence = {
        "matched_reactants": matched,
        "missing_query_reactants": missing,
        "reactant_coverage": round(coverage, 4),
        "source_reactant_coverage": round(record_coverage, 4),
        "component_jaccard": round(jaccard, 4),
        "source_reactant_mixture": record.get("reactant_components", []),
        "patent_id": record.get("patent_id", ""),
        "reaction_class_name": record.get("reaction_class_name", ""),
    }
    confidence = "high" if score >= 0.80 and coverage >= 0.75 else "medium" if score >= 0.45 else "low"
    return {
        "rank": 0,
        "smiles": record["product"],
        "canonical_smiles": record["canonical_product"],
        "formula": record.get("product_formula"),
        "score": round(score, 6),
        "confidence": confidence,
        "source": record["source"],
        "source_id": record["source_id"],
        "reaction_evidence": reaction_evidence,
        "constraint_evidence": constraint_evidence,
        "warnings": candidate_warnings,
    }
