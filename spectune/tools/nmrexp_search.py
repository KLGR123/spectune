"""Search an experimental NMR structure database for candidate molecules."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .base import JsonDict, Tool, ToolResult
from .config import NmrExpSearchConfig
from .utils import as_list, canonical_smiles, molecular_formula, post_json, spectrum_shift_arrays


class NmrExpSearchTool(Tool):
    name = "nmrexp_search"
    description = (
        "Search for structures consistent with 1H/13C NMR evidence via an external "
        "experimental-NMR search backend (iterative mutate/filter/pool, not a generative model). "
        "The database search runs remotely and can be slow; candidates are references to verify, not answers."
    )

    def __init__(self, config: NmrExpSearchConfig | None = None) -> None:
        self.config = config or NmrExpSearchConfig()

    @property
    def parameters(self) -> JsonDict:
        config = self.config
        return {
            "type": "object",
            "properties": {
                "h_nmr_peaks": {"type": "array", "items": {"type": "object"}},
                "c_nmr_peaks": {"type": "array", "items": {"type": "object"}},
                "h_shifts": {"type": "array", "items": {"type": "number"}, "description": "Raw 1H shifts (ppm)."},
                "c_shifts": {"type": "array", "items": {"type": "number"}, "description": "Raw 13C shifts (ppm)."},
                "h_split": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiplicities aligned with h_shifts.",
                },
                "num_search": {"type": "integer", "minimum": 1, "default": config.default_num_search},
                "topk": {"type": "integer", "minimum": 1, "default": config.default_topk},
                "allowed_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": list(config.default_allowed_elements),
                },
                "name": {"type": "string", "description": "Optional identifier attached to the search request."},
                "sigma_h": {"type": "number", "default": config.default_sigma_h},
                "sigma_c": {"type": "number", "default": config.default_sigma_c},
                "use_h_split": {"type": "boolean", "default": config.default_use_h_split},
                "split_coef": {"type": "number", "default": config.default_split_coef},
                "max_iter": {"type": "integer", "minimum": 1, "default": config.default_max_iter},
                "num_pool": {"type": "integer", "minimum": 1, "default": config.default_num_pool},
                "num_filter_pair": {"type": "integer", "minimum": 1, "default": config.default_num_filter_pair},
                "num_filter_mol": {"type": "integer", "minimum": 1, "default": config.default_num_filter_mol},
                "num_mutate_mol": {"type": "integer", "minimum": 1, "default": config.default_num_mutate_mol},
                "use_stereo": {"type": "boolean", "default": config.default_use_stereo},
                "optional_halogens": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": list(config.default_optional_halogens),
                },
                "max_cycle_length": {"type": "integer", "minimum": 1, "default": config.default_max_cycle_length},
                "invalid_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": list(config.default_invalid_patterns),
                    "description": "SMARTS patterns excluded from generated structures.",
                },
                "include_active_hs": {"type": "string", "default": config.default_include_active_hs},
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        if not self.config.mcp_base_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["NMREXP_SEARCH_MCP_BASE_URL is not configured"],
            )

        args = dict(arguments or {})
        h_shifts, h_split, c_shifts = spectrum_shift_arrays(args)
        if not h_shifts and not c_shifts:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=["nmrexp_search requires 1H and/or 13C spectral evidence"],
            )

        config = self.config
        num_search = max(1, int(args.get("num_search") or config.default_num_search))
        topk = max(1, int(args.get("topk") or config.default_topk))
        allowed_elements = [str(item) for item in as_list(args.get("allowed_elements")) if str(item).strip()] or list(
            config.default_allowed_elements
        )

        payload = {
            "name": str(args.get("name") or ""),
            "input_data": {
                "search": {
                    "H_split": h_split,
                    "H_shifts": h_shifts,
                    "C_shifts": c_shifts,
                    "num_search": num_search,
                    "topk": topk,
                    "allowed_elements": allowed_elements,
                },
                "config": _search_config(args, config, num_search=num_search, topk=topk),
            },
        }

        url = config.mcp_base_url.rstrip("/") + config.endpoint_path
        try:
            raw = await asyncio.to_thread(post_json, url, payload, timeout=config.timeout_s)
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": payload},
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

        candidates = _parse_search_results(raw, topk=topk)
        return ToolResult(
            completion="success",
            status="ok" if candidates else "no_candidates",
            data={"request": payload, "candidates": candidates},
        )


def _search_config(
    args: JsonDict,
    config: NmrExpSearchConfig,
    *,
    num_search: int,
    topk: int,
) -> JsonDict:
    optional_halogens = [str(item) for item in as_list(args.get("optional_halogens"))] or list(
        config.default_optional_halogens
    )
    invalid_patterns = [str(item) for item in as_list(args.get("invalid_patterns"))] or list(
        config.default_invalid_patterns
    )
    return {
        "sigma_h": float(args.get("sigma_h") if args.get("sigma_h") is not None else config.default_sigma_h),
        "sigma_c": float(args.get("sigma_c") if args.get("sigma_c") is not None else config.default_sigma_c),
        "use_H_split": bool(args.get("use_h_split", config.default_use_h_split)),
        "split_coef": float(
            args.get("split_coef") if args.get("split_coef") is not None else config.default_split_coef
        ),
        "max_iter": int(args.get("max_iter") or config.default_max_iter),
        "num_search": num_search,
        "num_pool": int(args.get("num_pool") or config.default_num_pool),
        "num_filter_pair": int(args.get("num_filter_pair") or config.default_num_filter_pair),
        "num_filter_mol": int(args.get("num_filter_mol") or config.default_num_filter_mol),
        "num_mutate_mol": int(args.get("num_mutate_mol") or config.default_num_mutate_mol),
        "topk": topk,
        "use_stereo": bool(args.get("use_stereo", config.default_use_stereo)),
        "optional_halogens": optional_halogens,
        "max_cycle_length": int(args.get("max_cycle_length") or config.default_max_cycle_length),
        "invalid_patterns": invalid_patterns,
        "include_active_hs": str(args.get("include_active_hs") or config.default_include_active_hs),
    }


def _candidate_rows(raw: Any) -> list[Any]:
    """Best-effort extraction of a candidate list from a search-backend response.

    The exact response schema of the upstream search service is not part of the
    reference material available for this port, so this accepts several
    reasonably-shaped containers (``candidates``, ``results``, ``smiles``) at the
    top level or nested under ``data``.
    """
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []
    for key in ("candidates", "results", "smiles_list", "smiles"):
        value = raw.get(key)
        if isinstance(value, list):
            return value
    data = raw.get("data")
    if isinstance(data, dict):
        return _candidate_rows(data)
    return []


def _parse_search_results(raw: JsonDict, *, topk: int) -> list[JsonDict]:
    rows: list[JsonDict] = []
    seen: set[str] = set()
    for item in _candidate_rows(raw):
        if isinstance(item, dict):
            smiles = str(item.get("smiles") or item.get("canonical_smiles") or "")
        else:
            smiles = str(item or "")
        canonical = canonical_smiles(smiles)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        row: JsonDict = {"rank": len(rows) + 1, "smiles": canonical, "canonical_smiles": canonical}
        if isinstance(item, dict):
            for key in ("score", "total_score", "h_cost", "c_cost"):
                if item.get(key) is not None:
                    row[key] = item.get(key)
        formula = molecular_formula(canonical)
        if formula:
            row["formula"] = formula
        rows.append(row)
        if len(rows) >= topk:
            break
    return rows
