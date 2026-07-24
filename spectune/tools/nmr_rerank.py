"""Rerank and score candidate molecules against NMR spectral evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from spectune.config import NmrRerankConfig

from .base import JsonDict, Tool, ToolResult
from .utils import canonical_smiles, has_rdkit, molecular_formula, post_json, spectrum_payload


class NmrRerankTool(Tool):
    name = "nmr_rerank"
    description = (
        "Score and rerank candidate SMILES against NMR spectral evidence, optionally filtered "
        "by a target molecular formula, via an external scoring model. Runs remotely (may be "
        "slow); scores are a reliable ranking signal worth cross-checking."
    )

    def __init__(self, config: NmrRerankConfig | None = None) -> None:
        self.config = config or NmrRerankConfig()

    @property
    def parameters(self) -> JsonDict:
        return {
            "type": "object",
            "properties": {
                "smiles_list": {"type": "array", "items": {"type": "string"}, "description": "Candidates to score."},
                "topk": {"type": "integer", "minimum": 1, "default": self.config.default_topk},
                "max_to_score": {"type": "integer", "minimum": 1, "description": "Alias for topk."},
                "formula": {"type": "string", "description": "Optional hard formula filter applied before scoring."},
                "molecular_formula": {"type": "string", "description": "Alias for formula."},
                "h_nmr_peaks": {"type": "array", "items": {"type": "object"}},
                "c_nmr_peaks": {"type": "array", "items": {"type": "object"}},
                "h_shifts": {"type": "array", "items": {"type": "number"}},
                "c_shifts": {"type": "array", "items": {"type": "number"}},
                "solvent": {"type": "string"},
            },
            "required": ["smiles_list"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        if not self.config.api_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["NMR_RANK_API_URL is not configured"],
            )

        args = dict(arguments or {})
        topk = max(1, int(args.get("topk") or args.get("max_to_score") or self.config.default_topk))
        smiles_list = _unique_canonical(args.get("smiles_list"))
        if not smiles_list:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=["nmr_rerank requires a non-empty smiles_list"],
            )

        target_formula = str(args.get("formula") or args.get("molecular_formula") or "").strip()
        warnings: list[str] = []
        rdkit_available = has_rdkit()
        if target_formula and not rdkit_available:
            warnings.append("rdkit is not installed; formula_filter was skipped")
            target_formula = ""
        elif target_formula:
            before = len(smiles_list)
            smiles_list = [smi for smi in smiles_list if molecular_formula(smi) == target_formula]
            warnings.append(f"formula_filter_applied: {target_formula} kept {len(smiles_list)}/{before} candidates")
            if not smiles_list:
                return ToolResult(
                    completion="success",
                    status="no_candidates",
                    data={"candidates": []},
                    warnings=warnings,
                )

        payload = {"entries": [{"input": spectrum_payload(args), "smiles_list": smiles_list}]}
        try:
            raw = await asyncio.to_thread(post_json, self.config.api_url, payload, timeout=self.config.timeout_s)
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload},
                warnings=[*warnings, f"{type(exc).__name__}: {exc}"],
            )

        candidates = _parse_ranking(raw, target_formula=target_formula, topk=topk)
        return ToolResult(
            completion="success",
            status="ok" if candidates else "no_candidates",
            data={"request": payload, "candidates": candidates},
            warnings=warnings,
        )


def _unique_canonical(value: Any) -> list[str]:
    raw = value if isinstance(value, list) else ([value] if value else [])
    out: list[str] = []
    for item in raw:
        canonical = canonical_smiles(str(item or ""))
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def _parse_ranking(raw: JsonDict, *, target_formula: str, topk: int) -> list[JsonDict]:
    entries = raw.get("entries") if isinstance(raw.get("entries"), list) else []
    entry = entries[0] if entries and isinstance(entries[0], dict) else {}
    ranking = entry.get("ranking") if isinstance(entry.get("ranking"), list) else []

    rows: list[JsonDict] = []
    seen: set[str] = set()
    for item in ranking:
        if not isinstance(item, dict):
            continue
        canonical = canonical_smiles(str(item.get("smiles") or ""))
        if not canonical or canonical in seen:
            continue
        formula = molecular_formula(canonical)
        if target_formula and formula != target_formula:
            continue
        seen.add(canonical)
        row: JsonDict = {"rank": len(rows) + 1, "smiles": canonical, "canonical_smiles": canonical}
        for key in ("total_score", "h_cost", "c_cost", "score"):
            if item.get(key) is not None:
                row[key] = item.get(key)
        if formula:
            row["formula"] = formula
        rows.append(row)
        if len(rows) >= topk:
            break
    return rows
