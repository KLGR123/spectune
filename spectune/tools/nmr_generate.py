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
        "通过外部生成式模型，根据 NMR 谱峰（1H/13C 化学位移）生成候选结构，"
        "可选以目标分子式作约束。"
    )

    def __init__(self, config: NmrGenerateConfig | None = None) -> None:
        self.config = config or NmrGenerateConfig()

    @property
    def parameters(self) -> JsonDict:
        return {
            "type": "object",
            "properties": {
                "topk": {"type": "integer", "minimum": 1, "default": self.config.default_topk},
                # "beam_size": {"type": "integer", "minimum": 1, "description": "生成 beam size；默认等于 topk"},
                # "batch_size": {"type": "integer", "minimum": 1, "description": "推理 batch size；默认 64"},
                "formula": {"type": "string", "description": "目标分子式"},
                # "molecular_formula": {"type": "string", "description": "formula 别名"},
                "h_nmr_peaks": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "1H 谱峰对象列表，每个对象包含 "
                        "'centroid'（float，ppm，峰范围的中点）、"
                        "'delta'（float，与 centroid 相同的值）、'nH'（int，质子数）、"
                        "'category'（str，标准化的多重性代码，如 's'、'd'、't'、'q'、'm'、"
                        "'dd'、'td'、'dq'、'brs'）、'j_values'（str，以 '_' 连接的 J 耦合常数，"
                        "单位 Hz，如 '8.0_4.0'），以及当峰跨越一个范围时可选的 'rangeMax'/'rangeMin'"
                        "（float，ppm）；不要把一个范围峰拆成两个独立的 centroid"
                    ),
                },
                "c_nmr_peaks": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "13C 谱峰对象列表，每个对象包含 "
                        "'delta (ppm)'（float，ppm 注意键名中括号内包含单位，"
                        "与解析器输出保持一致）；也接受 'delta' 或 'centroid' 作为备选键名"
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
