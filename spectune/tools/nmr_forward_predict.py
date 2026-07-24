"""Forward-predict atom-level NMR shifts for candidate molecules via MCP."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from spectune.config import NmrForwardPredictConfig

from .base import JsonDict, Tool, ToolResult
from .utils import canonical_smiles, experimental_shift_lists, float_list, molecular_formula


class NmrForwardPredictTool(Tool):
    name = "nmr_forward_predict"
    description = (
        "Predict atom-level 1H/13C NMR shifts for candidate SMILES via an external NMR "
        "prediction model, optionally comparing against experimental shifts. The model "
        "runs remotely (may be slow); predictions are approximate references worth cross-checking."
    )
    parameters: JsonDict = {
        "type": "object",
        "properties": {
            "smiles_list": {"type": "array", "items": {"type": "string"}, "description": "Candidates to predict."},
            "spectrum": {"type": "object", "description": "Optional experimental spectrum for comparison."},
            "h_shifts": {"type": "array", "items": {"type": "number"}, "description": "Experimental 1H shifts."},
            "c_shifts": {"type": "array", "items": {"type": "number"}, "description": "Experimental 13C shifts."},
        },
        "required": ["smiles_list"],
        "additionalProperties": False,
    }

    def __init__(self, config: NmrForwardPredictConfig | None = None) -> None:
        self.config = config or NmrForwardPredictConfig()

    async def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        args = dict(arguments or {})
        smiles_list = [str(s) for s in (args.get("smiles_list") or []) if str(s or "").strip()]
        if not smiles_list:
            return ToolResult(
                completion="failure",
                status="error",
                warnings=["nmr_forward_predict requires a non-empty smiles_list"],
            )

        if not self.config.mcp_url:
            return ToolResult(
                completion="failure",
                status="unavailable",
                warnings=["NMR_PREDICT_MCP_URL is not configured"],
            )

        h_shifts, c_shifts = experimental_shift_lists(args)
        request_data: JsonDict = {"smiles_list": smiles_list}
        if h_shifts:
            request_data["H_shifts"] = h_shifts
        if c_shifts:
            request_data["C_shifts"] = c_shifts

        try:
            raw = await asyncio.wait_for(self._call_mcp(request_data), timeout=self.config.timeout_s)
        except Exception as exc:
            return ToolResult(
                completion="failure",
                status="error",
                data={"request": request_data},
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

        candidates = parse_predictions(raw, smiles_list)
        return ToolResult(
            completion="success",
            status="ok" if candidates else "no_candidates",
            data={"request": request_data, "candidates": candidates},
        )

    async def _call_mcp(self, data: JsonDict) -> Any:
        try:
            from fastmcp import Client  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("fastmcp is not installed; install spectune[mcp] to enable nmr_forward_predict") from exc

        client = Client({"mcpServers": {"nmr": {"url": self.config.mcp_url}}})
        async with client:
            await client.ping()
            result = await client.call_tool(self.config.mcp_tool_name, {"data": data})
        content = getattr(result, "content", None) or []
        if content:
            text = getattr(content[0], "text", None)
            if text is not None:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw_text": str(text)}
        return {"raw": repr(result)}


def parse_predictions(raw: Any, input_smiles: list[str]) -> list[JsonDict]:
    """Normalize an MCP ``NMR_predict`` response into candidate rows.

    Exposed at module level (not private) so its chemistry-free parsing logic
    can be unit-tested without a real MCP round trip.
    """
    data = raw.get("data") if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and data is None:
        data = raw.get("result") or raw.get("predictions") or []
    if not isinstance(data, list):
        return []

    rows: list[JsonDict] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        smiles = str(item.get("smiles") or (input_smiles[index] if index < len(input_smiles) else ""))
        canonical = canonical_smiles(smiles)
        if not canonical:
            continue
        atoms_shift = float_list(item.get("atoms_shift"))
        atom_rows, h_shifts, c_shifts = split_atom_shifts(str(item.get("smiles_with_atom_order") or ""), atoms_shift)
        row: JsonDict = {
            "rank": len(rows) + 1,
            "smiles": canonical,
            "canonical_smiles": canonical,
            "predicted_nmr": {"atom_shifts": atom_rows, "h_shifts": h_shifts, "c_shifts": c_shifts},
        }
        formula = molecular_formula(canonical)
        if formula:
            row["formula"] = formula
        rows.append(row)
    return rows


def split_atom_shifts(
    smiles_with_atom_order: str,
    atoms_shift: list[float],
) -> tuple[list[JsonDict], list[float], list[float]]:
    """Split atom-map-indexed shifts into per-atom rows and H/C shift lists."""
    if not smiles_with_atom_order or not atoms_shift:
        return [], [], []
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError:
        return [], [], []

    molecule = Chem.MolFromSmiles(smiles_with_atom_order, sanitize=False)
    if molecule is None:
        return [], [], []

    atom_rows: list[JsonDict] = []
    h_shifts: list[float] = []
    c_shifts: list[float] = []
    for atom in molecule.GetAtoms():
        atom_map = int(atom.GetAtomMapNum() or 0)
        if atom_map <= 0 or atom_map > len(atoms_shift):
            continue
        shift = float(atoms_shift[atom_map - 1])
        element = atom.GetSymbol()
        atom_rows.append({"atom_map": atom_map, "element": element, "shift": shift})
        if element == "H":
            h_shifts.append(shift)
        elif element == "C":
            c_shifts.append(shift)
    return atom_rows, sorted(h_shifts), sorted(c_shifts)
