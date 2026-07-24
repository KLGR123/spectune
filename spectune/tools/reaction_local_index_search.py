"""Local reaction-precedent search via canonical reactant-set overlap.

This is deterministic set-overlap scoring against flat local reaction tables
(USPTO, ChemPile-lift, Pistachio), not embedding/vector retrieval.
"""

from __future__ import annotations

import csv
import functools
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spectune.config import ReactionLocalIndexSearchConfig

from .base import JsonDict, Tool, ToolResult
from .utils import as_list, has_rdkit, molecular_formula, strict_canonical_smiles

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
        "Search local reaction precedent tables (USPTO / ChemPile-lift / Pistachio) for "
        "products of reactions sharing reactants with the query, via canonical "
        "reactant-set overlap scoring. Does not read NMR/MS spectra."
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
                    "description": "Optional reaction SMILES, 'reactants>>products' or 'reactants>agents>products'.",
                },
                "reactants": {"type": "array", "items": {"type": "string"}, "description": "Reactant SMILES."},
                "reagents": {"type": "array", "items": {"type": "string"}, "description": "Optional reagent SMILES."},
                "conditions": {
                    "type": "string",
                    "description": "Free-text reaction conditions, kept as provenance only (unused by scoring).",
                },
                "reaction_type": {
                    "type": "string",
                    "description": "Free-text reaction type hint, kept as provenance only (unused by scoring).",
                },
                "target_formula": {"type": "string", "description": "Optional target product molecular formula."},
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional functional-group hints checked against candidate products.",
                },
                "topk": {"type": "integer", "minimum": 1, "maximum": 100, "default": self.config.default_topk},
                "sources": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["uspto", "chempile", "pistachio"]},
                    "description": "Restrict to a subset of the configured local sources (default: all configured).",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        if not has_rdkit():
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["rdkit is not installed; install spectune[chem] to enable reaction_local_index_search"],
            )

        config = self.config
        args = dict(arguments or {})
        requested_sources = {str(s).strip().lower() for s in as_list(args.get("sources")) if str(s).strip()}

        available: list[tuple[str, str]] = []
        if (not requested_sources or "uspto" in requested_sources) and config.uspto_csv_path:
            available.append(("uspto", config.uspto_csv_path))
        if (not requested_sources or "chempile" in requested_sources) and config.chempile_parquet_path:
            available.append(("chempile", config.chempile_parquet_path))
        if (not requested_sources or "pistachio" in requested_sources) and config.pistachio_smi_path:
            available.append(("pistachio", config.pistachio_smi_path))

        if not available:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=[
                    "no local reaction index is configured; set RXN_LOCAL_INDEX_USPTO_CSV, "
                    "RXN_LOCAL_INDEX_CHEMPILE_PARQUET, and/or RXN_LOCAL_INDEX_PISTACHIO_SMI"
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

        records: list[JsonDict] = []
        sources_used: list[str] = []
        source_errors: list[str] = []
        for source_name, path in available:
            try:
                if source_name == "uspto":
                    loaded = _load_uspto_csv(path)
                elif source_name == "chempile":
                    loaded = _load_chempile_parquet(path, max_records=config.max_chempile_records)
                else:
                    loaded = _load_pistachio_smi(path, max_records=config.max_pistachio_records)
            except Exception as exc:
                source_errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
                continue
            if loaded:
                records.extend(loaded)
                sources_used.append(source_name)

        if not records:
            return ToolResult(
                completion="success" if not source_errors else "partial",
                status="no_candidates",
                data={"provenance": {"sources_configured": [s for s, _ in available], "sources_loaded": []}},
                warnings=[*warnings, *source_errors, "no records loaded from any configured local source"],
            )

        query_set = set(query_components)
        best_by_product: dict[str, JsonDict] = {}
        for record in records:
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
                    "sources_configured": [s for s, _ in available],
                    "sources_loaded": sources_used,
                    "num_records_scanned": len(records),
                    "canonical_query_components": query_components,
                },
            },
            warnings=[*warnings, *source_errors],
        )


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


@functools.lru_cache(maxsize=8)
def _load_uspto_csv(path: str) -> tuple[JsonDict, ...]:
    """Load the ChemLLMBench-style USPTO CSV (columns: reactant, product)."""
    csv_path = Path(path)
    if not csv_path.exists():
        return ()
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
    return tuple(records)


@functools.lru_cache(maxsize=8)
def _load_chempile_parquet(path: str, *, max_records: int) -> tuple[JsonDict, ...]:
    """Load ChemPile-lift USPTO rows by extracting embedded 'reactants>>products' text."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        return ()
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
    return tuple(records)


@functools.lru_cache(maxsize=8)
def _load_pistachio_smi(path: str, *, max_records: int) -> tuple[JsonDict, ...]:
    """Load a bounded prefix of a Pistachio-style tab-separated reaction SMILES file."""
    smi_path = Path(path)
    if not smi_path.exists():
        return ()
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
    return tuple(records)


def _constraint_evidence(smiles: str, constraints: Sequence[str], target_formula: str) -> JsonDict:
    from rdkit import Chem  # type: ignore[import-not-found]

    molecule = Chem.MolFromSmiles(smiles)
    formula = molecular_formula(smiles) if molecule is not None else None
    matched: list[str] = []
    missing: list[str] = []
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

    constraint_evidence = _constraint_evidence(record["canonical_product"], constraints, target_formula)
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
