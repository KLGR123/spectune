"""Generate candidate molecular structures from NMR spectral evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from spectune.config import NmrGenerateConfig

from .base import JsonDict, Tool, ToolResult
from .utils import canonical_smiles, infer_nmr_type, molecular_formula, post_json, spectrum_payload


class NmrGenerateTool(Tool):
    name = "nmr_generate"
    description = (
        "Generate candidate molecular structures from NMR spectral peaks (1H/13C shifts), "
        "optionally constrained by a target molecular formula."
    )

    def __init__(self, config: NmrGenerateConfig | None = None) -> None:
        self.config = config or NmrGenerateConfig()

    @property
    def parameters(self) -> JsonDict:
        return {
            "type": "object",
            "properties": {
                "topk": {"type": "integer", "minimum": 1, "default": self.config.default_topk},
                "beam_size": {"type": "integer", "minimum": 1, "description": "Generation beam size."},
                "batch_size": {"type": "integer", "minimum": 1},
                "nmr_type": {"type": "string", "description": "Optional override, e.g. 'CHF'."},
                "formula": {"type": "string", "description": "Target molecular formula."},
                "molecular_formula": {"type": "string", "description": "Alias for formula."},
                "h_nmr_peaks": {"type": "array", "items": {"type": "object"}},
                "c_nmr_peaks": {"type": "array", "items": {"type": "object"}},
                "h_shifts": {"type": "array", "items": {"type": "number"}, "description": "Raw 1H shifts (ppm)."},
                "c_shifts": {"type": "array", "items": {"type": "number"}, "description": "Raw 13C shifts (ppm)."},
                "h_split": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiplicities for h_shifts.",
                },
                "solvent": {"type": "string"},
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        if not self.config.api_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["NMR_GENERATE_API_URL is not configured"],
            )

        args = dict(arguments or {})
        topk = max(1, int(args.get("topk") or args.get("beam_size") or self.config.default_topk))
        spectrum = spectrum_payload(args)
        has_evidence = spectrum.get("h_nmr_peaks") or spectrum.get("c_nmr_peaks") or spectrum.get("molecular_formula")
        if not has_evidence:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=["nmr_generate requires NMR peaks (or shift arrays) and/or a target formula"],
            )

        payload = {
            "spec_list": [spectrum],
            "nmr_type": str(args.get("nmr_type") or infer_nmr_type(spectrum)),
            "rerank": False,
            "beam_size": int(args.get("beam_size") or topk),
            "batch_size": int(args.get("batch_size") or args.get("beam_size") or topk),
        }
        try:
            raw = await asyncio.to_thread(post_json, self.config.api_url, payload, timeout=self.config.timeout_s)
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload},
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

        candidates = _parse_candidates(raw, topk=topk)
        return ToolResult(
            completion="success",
            status="ok" if candidates else "no_candidates",
            data={"request": payload, "candidates": candidates},
        )


def _parse_candidates(raw: JsonDict, *, topk: int) -> list[JsonDict]:
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    sequences = data.get("sequences")
    if isinstance(sequences, list) and sequences and isinstance(sequences[0], list):
        smiles_list = sequences[0]
    elif isinstance(sequences, list):
        smiles_list = sequences
    else:
        smiles_list = raw.get("sequences") or raw.get("smiles") or []
    scores = data.get("scores") if isinstance(data, dict) else None

    rows: list[JsonDict] = []
    seen: set[str] = set()
    for index, smiles in enumerate(smiles_list or []):
        canonical = canonical_smiles(str(smiles or ""))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        row: JsonDict = {"rank": len(rows) + 1, "smiles": canonical, "canonical_smiles": canonical}
        if isinstance(scores, list) and index < len(scores) and scores[index] is not None:
            row["score"] = scores[index]
        formula = molecular_formula(canonical)
        if formula:
            row["formula"] = formula
        rows.append(row)
        if len(rows) >= topk:
            break
    return rows
