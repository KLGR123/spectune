"""Repair or refine a seed molecule toward a target molecular formula."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from spectune.config import NmrRepairConfig

from .base import JsonDict, Tool, ToolResult
from .utils import canonical_smiles, molecular_formula, post_json


class NmrRepairTool(Tool):
    name = "nmr_repair"
    description = "Repair or refine a seed candidate molecule so it matches a target molecular formula."
    parameters: JsonDict = {
        "type": "object",
        "properties": {
            "input_smiles": {"type": "string", "description": "Seed molecule SMILES to repair."},
            "smiles": {"type": "string", "description": "Alias for input_smiles."},
            "candidate_smiles": {"type": "string", "description": "Alias for input_smiles."},
            "target_formula": {"type": "string", "description": "Molecular formula the output must match."},
            "formula": {"type": "string", "description": "Alias for target_formula."},
            "molecular_formula": {"type": "string", "description": "Alias for target_formula."},
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, config: NmrRepairConfig | None = None) -> None:
        self.config = config or NmrRepairConfig()

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        if not self.config.api_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["NMR_REPAIR_API_URL is not configured"],
            )

        args = dict(arguments or {})
        input_smiles = canonical_smiles(
            str(args.get("input_smiles") or args.get("smiles") or args.get("candidate_smiles") or "")
        )
        target_formula = str(
            args.get("target_formula") or args.get("formula") or args.get("molecular_formula") or ""
        ).strip()
        if not input_smiles or not target_formula:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=["nmr_repair requires input_smiles and target_formula"],
            )

        payload = {"input_smiles": input_smiles, "target_formula": target_formula}
        try:
            raw = await asyncio.to_thread(post_json, self.config.api_url, payload, timeout=self.config.timeout_s)
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload},
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

        repaired = canonical_smiles(str(raw.get("repaired_smiles") or ""))
        candidates: list[JsonDict] = []
        if repaired:
            row: JsonDict = {
                "rank": 1,
                "smiles": repaired,
                "canonical_smiles": repaired,
                "repair": {
                    "input_smiles": input_smiles,
                    "target_formula": target_formula,
                    "fallback_used": raw.get("fallback_used"),
                },
            }
            formula = molecular_formula(repaired)
            if formula:
                row["formula"] = formula
            candidates.append(row)

        success = bool(raw.get("success")) and bool(candidates)
        return ToolResult(
            completion="success",
            status="ok" if success else "no_candidates",
            data={"request": payload, "candidates": candidates},
        )
