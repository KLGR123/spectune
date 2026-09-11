"""Generate candidate molecular structures from NMR spectral evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .base import JsonDict, Tool, ToolResult
from .config import NmrGenerateConfig
from .utils import canonical_smiles, infer_nmr_type, molecular_formula, post_json, spectrum_payload


class NmrGenerateTool(Tool):
    name = "nmr_generate"
    description = (
        "Generate candidate structures from NMR peaks (1H/13C shifts), optionally constrained "
        "by a target molecular formula, via an external generative model. Runs remotely; "
        "candidates are suggestions to verify, not final answers."
    )

    def __init__(self, config: NmrGenerateConfig | None = None) -> None:
        self.config = config or NmrGenerateConfig()

    @property
    def parameters(self) -> JsonDict:
        return {
            "type": "object",
            "properties": {
                "topk": {"type": "integer", "minimum": 1, "default": self.config.default_topk},
                "beam_size": {"type": "integer", "minimum": 1, "description": "Generation beam size; defaults to topk."},
                "batch_size": {"type": "integer", "minimum": 1, "description": "Inference batch size; defaults to 64."},
                # nmr_type is inferred automatically from which peak arrays are present — not a caller parameter
                "formula": {"type": "string", "description": "Target molecular formula."},
                "molecular_formula": {"type": "string", "description": "Alias for formula."},
                "h_nmr_peaks": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "List of 1H peak objects as produced by parse_nmr_text. Each object "
                        "contains: 'centroid' (float, ppm, midpoint of the peak range), "
                        "'delta' (float, same value), 'nH' (int, number of protons), "
                        "'category' (str, canonical multiplicity code, e.g. 's','d','t','q','m',"
                        "'dd','td','dq','brs'), 'j_values' (str, J coupling constants in Hz "
                        "joined by '_', e.g. '8.0_4.0'), and optionally 'rangeMax'/'rangeMin' "
                        "(floats, ppm) when the peak spans a range. "
                        "Do NOT split a range peak into two separate centroids."
                    ),
                },
                "c_nmr_peaks": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "List of 13C peak objects as produced by parse_nmr_text. Each object "
                        "contains 'delta (ppm)' (float, ppm — note the key name includes the "
                        "unit in parentheses, exactly as output by the parser). "
                        "'delta' or 'centroid' are also accepted as fallback key names."
                    ),
                },
                # h_shifts / c_shifts / h_split: legacy flat-array form; not used in the primary API path
                # "h_shifts": {"type": "array", "items": {"type": "number"}, "description": "Raw 1H shifts (ppm)."},
                # "c_shifts": {"type": "array", "items": {"type": "number"}, "description": "Raw 13C shifts (ppm)."},
                # "h_split": {"type": "array", "items": {"type": "string"}, "description": "Multiplicities for h_shifts."},
                # solvent: read from the spectrum internally by spectrum_payload(); not a caller-provided parameter
                # "solvent": {"type": "string"},
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
            "batch_size": int(args.get("batch_size") or 64),
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
