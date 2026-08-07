"""Forward-predict atom-level NMR shifts for candidate molecules via HTTP JSON."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .base import JsonDict, Tool, ToolResult
from .config import NmrForwardPredictConfig
from .utils import post_json


class NmrForwardPredictTool(Tool):
    name = "nmr_forward_predict"
    description = (
        "Predict atom-level 1H/13C NMR shifts for a candidate SMILES via an external NMR "
        "prediction model. The model runs remotely (may be slow); predictions are approximate "
        "references worth cross-checking."
    )
    parameters: JsonDict = {
        "type": "object",
        "properties": {
            "smiles": {"type": "string", "description": "Candidate SMILES to predict."},
            "solvent": {"type": "string", "description": "NMR solvent (e.g. CDCl3, DMSO-d6)."},
        },
        "required": ["smiles"],
        "additionalProperties": False,
    }

    def __init__(self, config: NmrForwardPredictConfig | None = None) -> None:
        self.config = config or NmrForwardPredictConfig()

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        args = dict(arguments or {})
        smiles = str(args.get("smiles") or "").strip()
        if not smiles:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=["nmr_forward_predict requires a non-empty smiles"],
            )

        if not self.config.api_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["NMR_PREDICT_API_URL is not configured"],
            )

        payload: JsonDict = {"smiles": smiles}
        solvent = str(args.get("solvent") or "").strip()
        if solvent:
            payload["solvent"] = solvent

        try:
            result = await asyncio.to_thread(
                post_json, self.config.api_url, payload, timeout=self.config.timeout_s
            )
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

        return ToolResult(
            completion="success",
            status="ok",
            data=result,
        )
