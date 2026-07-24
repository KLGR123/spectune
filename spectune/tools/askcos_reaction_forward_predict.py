"""Forward reaction-product prediction via the public ASKCOS API."""

from __future__ import annotations

import asyncio
import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from spectune.config import AskcosReactionForwardPredictConfig

from .base import JsonDict, Tool, ToolResult
from .utils import as_list, molecular_formula, strict_canonical_smiles

_BACKENDS = {"wldn5", "augmented_transformer", "graph2smiles"}
_BACKEND_ALIASES = {
    "askcos": "wldn5",
    "at": "augmented_transformer",
    "augmented": "augmented_transformer",
    "g2s": "graph2smiles",
}
_FORWARD_ENDPOINT = "/api/forward/controller/call-sync"


class AskcosReactionForwardPredictTool(Tool):
    name = "askcos_reaction_forward_predict"
    description = (
        "Predict likely reaction products from reactant SMILES via the external ASKCOS "
        "forward-prediction model (MIT). Runs remotely (may be slow); predictions are references "
        "worth cross-checking. Does not read NMR/MS spectra, so spectrum matching needs "
        "a follow-up NMR tool."
    )

    def __init__(self, config: AskcosReactionForwardPredictConfig | None = None) -> None:
        self.config = config or AskcosReactionForwardPredictConfig()

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
                    "description": "Free-text conditions; only kept as provenance unless reagents/solvent are set.",
                },
                "reaction_type": {
                    "type": "string",
                    "description": "Free-text reaction type hint, kept as provenance only (not sent to ASKCOS).",
                },
                "target_formula": {"type": "string", "description": "Optional target product molecular formula."},
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional functional-group hints checked against predicted products.",
                },
                "topk": {"type": "integer", "minimum": 1, "maximum": 100, "default": config.default_topk},
                "api_backend": {
                    "type": "string",
                    "enum": sorted(_BACKENDS),
                    "default": config.default_api_backend,
                    "description": "ASKCOS forward-prediction model backend.",
                },
                "model_name": {"type": "string", "description": "Optional remote model name, e.g. 'pistachio'."},
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        config = self.config
        args = dict(arguments or {})

        reaction_smiles = str(args.get("reaction_smiles") or "").strip()
        rxn_reactants, rxn_agents, rxn_products = _split_reaction_smiles(reaction_smiles)
        reactants = [str(x) for x in as_list(args.get("reactants"))] or rxn_reactants
        reagents = [str(x) for x in as_list(args.get("reagents"))] or rxn_agents
        solvent = [str(x) for x in as_list(args.get("solvent"))]
        products = rxn_products
        conditions = str(args.get("conditions") or "").strip()
        reaction_type = str(args.get("reaction_type") or "").strip()
        target_formula = str(args.get("target_formula") or "").strip()
        constraints = [str(x) for x in as_list(args.get("constraints")) if str(x).strip()]
        topk = max(1, min(int(args.get("topk") or config.default_topk), 100))
        api_backend = _normalize_backend(str(args.get("api_backend") or config.default_api_backend))
        model_name = str(args.get("model_name") or _default_model_name(api_backend))

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
        if conditions and not (reagents or solvent):
            warnings.append("free_text_conditions_not_sent_to_askcos")
        if reaction_type:
            warnings.append("askcos_public_does_not_accept_reaction_type_field")
        if products:
            warnings.append("askcos_public_forward_prediction_ignores_products_field")

        if not precursors:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=[*warnings, "no valid reactant SMILES provided"],
            )

        payload = {
            "smiles": [precursors],
            "backend": api_backend,
            "model_name": model_name,
            "reagents": reagents_mix,
            "solvent": solvent_mix,
        }
        try:
            status_code, response_payload = await asyncio.to_thread(self._post, payload)
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload},
                warnings=[*warnings, f"{type(exc).__name__}: {exc}"],
            )

        api_status = int(response_payload.get("status_code") or status_code or 0)
        if status_code >= 400 or api_status >= 400:
            message = str(response_payload.get("message") or "ASKCOS API error")
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload, "http_status": status_code},
                warnings=[*warnings, message],
            )

        candidates = _parse_candidates(
            response_payload.get("result"),
            api_backend=api_backend,
            model_name=model_name,
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

    def _post(self, payload: JsonDict) -> tuple[int, JsonDict]:
        url = self.config.base_url.rstrip("/") + _FORWARD_ENDPOINT
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "spectune-askcos-reaction-forward-predict/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                body = response.read().decode("utf-8", "replace")
                return int(response.status), (json.loads(body) if body else {})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {"message": body[:500]}
            return int(exc.code), parsed


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


def _normalize_backend(value: str) -> str:
    backend = str(value or "wldn5").strip().lower().replace("-", "_")
    backend = _BACKEND_ALIASES.get(backend, backend)
    return backend if backend in _BACKENDS else "wldn5"


def _default_model_name(api_backend: str) -> str:
    return "pistachio" if api_backend == "wldn5" else "USPTO_STEREO"


def _iter_raw_candidates(value: Any) -> list[JsonDict]:
    out: list[JsonDict] = []
    if isinstance(value, list):
        for item in value:
            out.extend(_iter_raw_candidates(item))
        return out
    if not isinstance(value, dict):
        return out

    if "outcome" in value:
        outcome = value.get("outcome")
        smiles = outcome.get("smiles") if isinstance(outcome, dict) else outcome
        if isinstance(smiles, str) and smiles.strip():
            out.append({"smiles": smiles.strip(), "rank": value.get("rank"), "score": value.get("score")})
        return out

    products = value.get("products")
    if isinstance(products, list):
        scores = value.get("scores") if isinstance(value.get("scores"), list) else []
        for index, product in enumerate(products):
            if isinstance(product, str) and product.strip():
                out.append(
                    {
                        "smiles": product.strip(),
                        "rank": index + 1,
                        "score": scores[index] if index < len(scores) else None,
                    }
                )
        return out

    for child in value.values():
        out.extend(_iter_raw_candidates(child))
    return out


def _check_constraints(smiles: str, constraints: Sequence[str], target_formula: str) -> JsonDict:
    formula = molecular_formula(smiles)
    formula_match = formula == target_formula.strip() if target_formula.strip() else None
    # Structural constraint substrings are checked at the caller only when RDKit
    # SMARTS matching is warranted; ASKCOS candidates are validated by formula here.
    return {"formula": formula, "formula_match": formula_match, "checked_constraints": list(constraints)}


def _parse_candidates(
    raw_result: Any,
    *,
    api_backend: str,
    model_name: str,
    canonical_reactants: list[str],
    canonical_reagents: list[str],
    canonical_solvent: list[str],
    constraints: Sequence[str],
    target_formula: str,
    topk: int,
) -> list[JsonDict]:
    raw_records = _iter_raw_candidates(raw_result)
    raw_scores = [_float_or_none(item.get("score")) for item in raw_records]
    finite_scores = [score for score in raw_scores if score is not None]
    softmax_scores: list[float | None] = [None] * len(raw_records)
    if finite_scores:
        max_score = max(finite_scores)
        exp_scores = [math.exp((score if score is not None else max_score - 100.0) - max_score) for score in raw_scores]
        denom = sum(exp_scores) or 1.0
        softmax_scores = [value / denom for value in exp_scores]

    seen: dict[str, JsonDict] = {}
    for index, raw in enumerate(raw_records):
        canonical = strict_canonical_smiles(str(raw.get("smiles") or ""))
        if not canonical:
            continue
        score = softmax_scores[index]
        if score is None:
            rank_value = _float_or_none(raw.get("rank")) or float(index + 1)
            score = 1.0 / max(rank_value, 1.0)
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
            "smiles": str(raw.get("smiles") or ""),
            "canonical_smiles": canonical,
            "formula": constraint_evidence["formula"],
            "score": round(float(score), 6),
            "confidence": _confidence(score),
            "source": "askcos_public",
            "source_id": f"askcos_public:{api_backend}:{index + 1}",
            "reaction_evidence": {
                "api_backend": api_backend,
                "model_name": model_name,
                "canonical_reactants": canonical_reactants,
                "canonical_reagents": canonical_reagents,
                "canonical_solvent": canonical_solvent,
                "api_rank": raw.get("rank"),
                "api_score": raw.get("score"),
            },
            "constraint_evidence": constraint_evidence,
            "warnings": candidate_warnings,
        }
        previous = seen.get(canonical)
        if previous is None or candidate["score"] > previous["score"]:
            seen[canonical] = candidate

    ranked = sorted(seen.values(), key=lambda item: item["score"], reverse=True)[:topk]
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
