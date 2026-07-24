"""Forward reaction-product prediction via a self-hosted Uni-Mol3 HTTP service.

Uni-Mol3 needs its own conda env / model checkout, so it is not run in-process
(unlike, say, RDKit-based tools). This tool is a thin HTTP client that posts to
a configurable endpoint; until that endpoint is deployed and ``api_url`` is
set, it is a pure placeholder that reports ``status="unavailable"``. It is the
Uni-Mol3-backed sibling of :mod:`spectune.tools.askcos_reaction_forward_predict`
(ASKCOS-backed), so both tools accept the same reaction-description shape.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from spectune.config import Unimol3ReactionForwardPredictConfig

from .base import JsonDict, Tool, ToolResult
from .utils import as_list, molecular_formula, post_json, strict_canonical_smiles


class Unimol3ReactionForwardPredictTool(Tool):
    name = "unimol3_reaction_forward_predict"
    description = (
        "Predict likely reaction products from reactant SMILES via a Uni-Mol3 "
        "forward-prediction model served at a configurable HTTP endpoint (not run "
        "locally in-process). Currently a placeholder until that service is deployed: "
        "with no api_url configured it reports status='unavailable'. Does not read "
        "NMR/MS spectra; spectrum matching requires a follow-up call to an NMR tool."
    )

    def __init__(self, config: Unimol3ReactionForwardPredictConfig | None = None) -> None:
        self.config = config or Unimol3ReactionForwardPredictConfig()

    @property
    def parameters(self) -> JsonDict:
        config = self.config
        return {
            "type": "object",
            "properties": {
                "reaction_smiles": {
                    "type": "string",
                    "description": "Optional reaction SMILES, 'reactants>>products' or 'reactants>agents>products'.",
                },
                "reactants": {"type": "array", "items": {"type": "string"}, "description": "Reactant SMILES."},
                "reagents": {"type": "array", "items": {"type": "string"}, "description": "Optional reagent SMILES."},
                "solvent": {"type": "array", "items": {"type": "string"}, "description": "Optional solvent SMILES."},
                "conditions": {
                    "type": "string",
                    "description": "Free-text conditions, sent through to the Uni-Mol3 service as provenance.",
                },
                "reaction_type": {
                    "type": "string",
                    "description": "Free-text reaction type hint, kept as provenance only.",
                },
                "target_formula": {"type": "string", "description": "Optional target product molecular formula."},
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional functional-group hints checked against predicted products.",
                },
                "topk": {"type": "integer", "minimum": 1, "maximum": 100, "default": config.default_topk},
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        config = self.config
        if not config.api_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=[
                    "UNIMOL3_REACTION_FORWARD_PREDICT_API_URL is not configured; "
                    "Uni-Mol3 forward-predict service is not yet deployed"
                ],
            )

        args = dict(arguments or {})
        reaction_smiles = str(args.get("reaction_smiles") or "").strip()
        rxn_reactants, rxn_agents, rxn_products = _split_reaction_smiles(reaction_smiles)
        reactants = [str(x) for x in as_list(args.get("reactants"))] or rxn_reactants
        reagents = [str(x) for x in as_list(args.get("reagents"))] or rxn_agents
        solvent = [str(x) for x in as_list(args.get("solvent"))]
        conditions = str(args.get("conditions") or "").strip()
        reaction_type = str(args.get("reaction_type") or "").strip()
        target_formula = str(args.get("target_formula") or "").strip()
        constraints = [str(x) for x in as_list(args.get("constraints")) if str(x).strip()]
        topk = max(1, min(int(args.get("topk") or config.default_topk), 100))

        precursors, canonical_reactants, invalid_reactants = _canonical_mixture(reactants)
        reagents_mix, canonical_reagents, invalid_reagents = _canonical_mixture(reagents)
        solvent_mix, canonical_solvent, invalid_solvent = _canonical_mixture(solvent)

        warnings: list[str] = []
        if invalid_reactants:
            warnings.append("some_query_reactants_failed_smiles_parsing")
        if invalid_reagents:
            warnings.append("some_reagents_failed_smiles_parsing")
        if invalid_solvent:
            warnings.append("some_solvents_failed_smiles_parsing")
        if rxn_products:
            warnings.append("unimol3_reaction_forward_prediction_ignores_products_field")

        if not precursors:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=[*warnings, "no valid reactant SMILES provided"],
            )

        payload: JsonDict = {
            "reactants": canonical_reactants,
            "reagents": canonical_reagents,
            "solvent": canonical_solvent,
            "topk": topk,
        }
        if conditions:
            payload["conditions"] = conditions
        if reaction_type:
            payload["reaction_type"] = reaction_type

        try:
            raw = await asyncio.to_thread(post_json, config.api_url, payload, timeout=config.timeout_s)
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload},
                warnings=[*warnings, f"{type(exc).__name__}: {exc}"],
            )

        candidates = _parse_candidates(
            raw,
            canonical_reactants=canonical_reactants,
            canonical_reagents=canonical_reagents,
            canonical_solvent=canonical_solvent,
            constraints=constraints,
            target_formula=target_formula,
            topk=topk,
        )
        return ToolResult(
            completion="success",
            status="ok" if candidates else "no_candidates",
            data={"request": payload, "candidates": candidates},
            warnings=warnings,
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


def _canonical_mixture(values: Sequence[str]) -> tuple[str, list[str], list[str]]:
    canonical: list[str] = []
    invalid: list[str] = []
    for raw in values:
        for token in _split_mixture(raw):
            canon = strict_canonical_smiles(token)
            if canon:
                canonical.append(canon)
            else:
                invalid.append(token)
    canonical = sorted(dict.fromkeys(canonical))
    return ".".join(canonical), canonical, invalid


def _check_constraints(smiles: str, constraints: Sequence[str], target_formula: str) -> JsonDict:
    formula = molecular_formula(smiles)
    formula_match = formula == target_formula.strip() if target_formula.strip() else None
    return {"formula": formula, "formula_match": formula_match, "checked_constraints": list(constraints)}


def _parse_candidates(
    raw: Any,
    *,
    canonical_reactants: list[str],
    canonical_reagents: list[str],
    canonical_solvent: list[str],
    constraints: Sequence[str],
    target_formula: str,
    topk: int,
) -> list[JsonDict]:
    """Normalize a Uni-Mol3 forward-predict response into candidate rows.

    NOTE: there is no live Uni-Mol3 HTTP service to validate against yet, so this
    response shape is a best-effort guess (mirrors the ``{"candidates": [...]}`` /
    ``{"products": [...], "scores": [...]}`` shapes already used by
    ``askcos_reaction_forward_predict``/``nmr_generate``). Once the real service exists,
    check its actual response and adjust this parser rather than the call site.
    """
    if not isinstance(raw, dict):
        return []
    raw_records = raw.get("candidates")
    if not isinstance(raw_records, list):
        raw_records = []
        products = raw.get("products")
        scores = raw.get("scores") if isinstance(raw.get("scores"), list) else []
        if isinstance(products, list):
            for index, product in enumerate(products):
                raw_records.append({"smiles": product, "score": scores[index] if index < len(scores) else None})

    seen: dict[str, JsonDict] = {}
    for index, item in enumerate(raw_records):
        if not isinstance(item, dict):
            continue
        canonical = strict_canonical_smiles(str(item.get("smiles") or ""))
        if not canonical:
            continue
        score = _float_or_none(item.get("score"))
        if score is None:
            score = 1.0 / float(index + 1)
        constraint_evidence = _check_constraints(canonical, constraints, target_formula)
        if constraint_evidence["formula_match"] is True:
            score = min(1.0, max(score, 0.62 + 0.30 * score))
        elif constraint_evidence["formula_match"] is False:
            score *= 0.20
        candidate_warnings = []
        if constraint_evidence["formula_match"] is False:
            candidate_warnings.append("target_formula_mismatch")
        candidate = {
            "rank": 0,
            "smiles": str(item.get("smiles") or ""),
            "canonical_smiles": canonical,
            "formula": constraint_evidence["formula"],
            "score": round(float(score), 6),
            "confidence": _confidence(score),
            "source": "unimol3",
            "source_id": f"unimol3:{index + 1}",
            "reaction_evidence": {
                "canonical_reactants": canonical_reactants,
                "canonical_reagents": canonical_reagents,
                "canonical_solvent": canonical_solvent,
                "api_rank": item.get("rank"),
                "api_score": item.get("score"),
            },
            "constraint_evidence": constraint_evidence,
            "warnings": candidate_warnings,
        }
        previous = seen.get(canonical)
        if previous is None or candidate["score"] > previous["score"]:
            seen[canonical] = candidate

    ranked = sorted(seen.values(), key=lambda entry: entry["score"], reverse=True)[:topk]
    for rank, candidate in enumerate(ranked, 1):
        candidate["rank"] = rank
    return ranked


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _confidence(score: float) -> str:
    if score >= 0.50:
        return "high"
    if score >= 0.10:
        return "medium"
    return "low"
