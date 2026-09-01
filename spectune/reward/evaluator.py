"""Composable rewards for molecular-structure rollouts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import Any

from spectune.format.v1 import TOOL_CALL_END, TOOL_CALL_START, TOOL_RESPONSE_END, TOOL_RESPONSE_START
from spectune.format.v1 import extract_smiles_candidates as extract_v1_smiles
from spectune.format.v1 import extract_tool_calls as extract_v1_tool_calls
from spectune.format.v1 import final_answer_region
from spectune.format.v1 import messages_from_decoded_hermes

from .base import JsonDict, RewardResult
from .config import COMPONENT_KEYS, RewardConfig

_TOOL_CALL_RE = re.compile(
    rf"{re.escape(TOOL_CALL_START)}\s*(.*?)\s*{re.escape(TOOL_CALL_END)}",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_RESPONSE_RE = re.compile(
    rf"{re.escape(TOOL_RESPONSE_START)}\s*(.*?)\s*{re.escape(TOOL_RESPONSE_END)}",
    re.DOTALL | re.IGNORECASE,
)
_NMR_GENERATE_TOOL_NAME = "nmr_generate"
_SMILES_TAG_RE = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.DOTALL | re.IGNORECASE)
_ANSWER_KEYS = ("candidates", "smiles_list", "answers", "answer", "smiles", "canonical_smiles")
_SMILES_TOKEN_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$\\/%.:*]+$")
_RANKED_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+\s*[.)])\s+(\S+)")
_LABELED_LINE_RE = re.compile(
    r"^\s*(?:rank\s*\d+\s*[:.)-]?|(?:canonical_)?smiles(?:\s*\d+)?\s*:)\s*(\S+)",
    re.IGNORECASE,
)


class RewardEvaluator:
    """Evaluate GT matching, tool-call format, SMILES validity and tool usage."""

    def __init__(
        self,
        config: RewardConfig | None = None,
        *,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.tool_schemas = list(tool_schemas or [])

    @classmethod
    def from_tool_manager(cls, tool_manager: Any, config: RewardConfig | None = None) -> RewardEvaluator:
        """Build an evaluator using a ``ToolManager``'s OpenAI schemas."""
        return cls(config, tool_schemas=tool_manager.schemas)

    def __call__(
        self,
        rollout: Sequence[Any] | str,
        ground_truth: Any,
        *,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    ) -> RewardResult:
        return self.evaluate(rollout, ground_truth, tool_schemas=tool_schemas)

    def evaluate(
        self,
        rollout: Sequence[Any] | str,
        ground_truth: Any,
        *,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    ) -> RewardResult:
        """Return the weighted reward and all independently inspectable parts."""
        messages = _normalize_rollout(rollout)
        final_answer = _last_assistant_content(messages)
        calls, malformed_calls = _collect_tool_calls(messages)
        schemas = list(tool_schemas) if tool_schemas is not None else self.tool_schemas
        schema_map = _schema_map(schemas)

        invalid_call_errors = list(malformed_calls)
        invalid_call_count = len(malformed_calls)
        for index, call in enumerate(calls):
            call_errors = _validate_tool_call(call, schema_map, index=index)
            invalid_call_errors.extend(call_errors)
            invalid_call_count += bool(call_errors)

        raw_candidates = _extract_answer_candidates(
            final_answer,
            strict=self.config.strict_answer_format,
        )
        gt_smiles = _ground_truth_smiles(ground_truth)
        canonical_gt, rdkit_available = _canonicalize_smiles(gt_smiles)
        valid_candidates: list[str] = []
        invalid_candidates: list[str] = []
        seen: set[str] = set()
        for raw_candidate in raw_candidates:
            canonical, _ = _canonicalize_smiles(raw_candidate)
            if canonical is None:
                invalid_candidates.append(raw_candidate)
            elif canonical not in seen:
                seen.add(canonical)
                valid_candidates.append(canonical)

        # Legacy free-form fallback: only when strict format is disabled.
        if not self.config.strict_answer_format and not raw_candidates and gt_smiles and gt_smiles in final_answer:
            valid_candidates.append(canonical_gt or gt_smiles.strip())

        gt_rank: int | None = None
        if canonical_gt:
            try:
                gt_rank = valid_candidates.index(canonical_gt) + 1
            except ValueError:
                pass

        config = self.config
        if not gt_smiles:
            # Empty gt means the correct answer is an empty candidate list.
            # Only reward when the agent explicitly output {"smiles": []} — not
            # when it produced unparseable output that happened to yield [].
            explicit_empty = _has_explicit_smiles_answer(final_answer, strict=config.strict_answer_format)
            gt_reward = config.gt_match_reward if (explicit_empty and not raw_candidates) else 0.0
        else:
            gt_reward = config.gt_match_reward * config.rank_discount ** (gt_rank - 1) if gt_rank is not None else 0.0
        tool_format_reward = config.invalid_tool_call_penalty * invalid_call_count
        smiles_validity_reward = config.invalid_smiles_penalty if invalid_candidates else 0.0
        tool_call_count = len(calls) + len(malformed_calls)
        excess_calls = max(0, tool_call_count - config.max_tool_calls) if config.max_tool_calls is not None else 0
        tool_count_reward = config.excess_tool_call_penalty * excess_calls

        components = {
            "gt_smiles": float(gt_reward),
            "tool_call_format": float(tool_format_reward),
            "smiles_validity": float(smiles_validity_reward),
            "tool_call_count": float(tool_count_reward),
        }
        weights = config.component_weights
        weighted_components = {key: float(weights[key] * components[key]) for key in COMPONENT_KEYS}

        called_nmr_generate, nmr_generate_first_results = _nmr_generate_calls_and_first_results(messages)
        first_answer_candidate = valid_candidates[0] if valid_candidates else None
        nmr_diversity_bonus_applied = bool(
            called_nmr_generate
            and nmr_generate_first_results
            and first_answer_candidate is not None
            and gt_rank == 1
            and first_answer_candidate not in nmr_generate_first_results
        )
        nmr_diversity_bonus = config.nmr_diversity_bonus if nmr_diversity_bonus_applied else 0.0

        warnings: list[str] = []
        if not rdkit_available:
            warnings.append("rdkit is unavailable; SMILES were compared as opaque strings and validity was not checked")
        if not gt_smiles:
            warnings.append("ground truth does not contain a non-empty SMILES")

        if os.getenv("VERL_DEBUG"):
            print("\n[DEBUG] === reward calculation ===")
            print(f"[DEBUG] ground_truth_smiles = {gt_smiles!r}, gt_rank = {gt_rank}")
            print(f"[DEBUG] answer_candidates = {raw_candidates}")
            print(f"[DEBUG] valid_candidates = {valid_candidates}")
            print(f"[DEBUG] invalid_tool_calls = {invalid_call_errors}")
            print(f"[DEBUG] components = {components}")
            print(f"[DEBUG] weighted_components = {weighted_components}")
            print(f"[DEBUG] nmr_diversity_bonus = {nmr_diversity_bonus}")
            print(f"[DEBUG] total_score = {sum(weighted_components.values()) + nmr_diversity_bonus:.4f}")
            breakpoint()

        return RewardResult(
            score=float(sum(weighted_components.values()) + nmr_diversity_bonus),
            components=components,
            details={
                "gt_rank": gt_rank,
                "ground_truth_smiles": gt_smiles or None,
                "answer_candidates": raw_candidates,
                "canonical_candidates": valid_candidates,
                "invalid_smiles": invalid_candidates,
                "tool_call_count": tool_call_count,
                "excess_tool_calls": excess_calls,
                "invalid_tool_call_count": invalid_call_count,
                "invalid_tool_calls": invalid_call_errors,
                "component_weights": dict(weights),
                "weighted_components": weighted_components,
                "called_nmr_generate": called_nmr_generate,
                "nmr_generate_first_results": nmr_generate_first_results,
                "nmr_diversity_bonus_applied": nmr_diversity_bonus_applied,
                "nmr_diversity_bonus": float(nmr_diversity_bonus),
                "strict_answer_format": config.strict_answer_format,
            },
            warnings=warnings,
        )


def _normalize_rollout(rollout: Sequence[Any] | str) -> list[JsonDict]:
    if isinstance(rollout, str):
        # Decoded verl responses are flat strings; rebuild hermes turns so
        # ``_last_assistant_content`` sees only the post-tool answer region.
        return [dict(message) for message in messages_from_decoded_hermes(rollout)]
    messages: list[JsonDict] = []
    for value in rollout:
        mapping = _as_mapping(value)
        if mapping is not None:
            messages.append(dict(mapping))
    return messages


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    if is_dataclass(value) and not isinstance(value, type):
        from dataclasses import asdict

        dumped = asdict(value)
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, bytes):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text", block.get("content"))
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _last_assistant_content(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _content_text(message.get("content"))
    return ""


def _collect_tool_calls(messages: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[str]]:
    calls: list[JsonDict] = []
    errors: list[str] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        structured = message.get("tool_calls")
        if isinstance(structured, Sequence) and not isinstance(structured, str | bytes):
            for value in structured:
                mapping = _as_mapping(value)
                if mapping is None:
                    errors.append(f"message {message_index}: tool call must be an object")
                else:
                    calls.append(dict(mapping))
            continue

        content = _content_text(message.get("content"))
        matches = list(_TOOL_CALL_RE.finditer(content))
        orphan_count = max(
            0,
            len(re.findall(r"<tool_call>", content, re.IGNORECASE)) - len(matches),
        )
        errors.extend(f"message {message_index}: unclosed <tool_call> tag" for _ in range(orphan_count))
        for match in matches:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                errors.append(f"message {message_index}: invalid tool-call JSON: {exc.msg}")
                continue
            if not isinstance(parsed, dict):
                errors.append(f"message {message_index}: tool call must decode to an object")
                continue
            calls.append(parsed)
    return calls, errors


def _assistant_tool_call_names(message: Mapping[str, Any]) -> list[str]:
    """Names of the tool calls issued by one assistant message, in call order."""
    structured = message.get("tool_calls")
    if isinstance(structured, Sequence) and not isinstance(structured, str | bytes):
        names: list[str] = []
        for value in structured:
            mapping = _as_mapping(value)
            if mapping is None:
                continue
            function = mapping.get("function") if isinstance(mapping.get("function"), Mapping) else mapping
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names
    # Hermes-tag content: reuse the shared v1 parser instead of redefining it.
    return [call["name"] for call in extract_v1_tool_calls(_content_text(message.get("content")))]


def _parse_tool_response_payload(content: str) -> JsonDict | None:
    """Parse a ``ToolResult``-shaped JSON blob out of a tool-response message.

    Tool responses may be wrapped in ``<tool_response>...</tool_response>``
    (live rollout, see ``rollout.py``) or appear as a bare JSON object (verl
    hermes decoding, ``role=tool``). Plain user turns never match this shape.
    """
    text = (content or "").strip()
    match = _TOOL_RESPONSE_RE.search(text)
    if match:
        text = match.group(1).strip()
    if not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or "completion" not in value or "status" not in value:
        return None
    return value


def _nmr_generate_calls_and_first_results(messages: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Return ``(called, first_results)`` for every ``nmr_generate`` call in the rollout.

    ``first_results`` holds the canonical top-candidate SMILES from each
    ``nmr_generate`` call whose response carried candidates (deduplicated,
    call order), matched to its response by call order (each tool-response
    message pairs with the earliest still-unmatched pending call name). When
    the trajectory issues multiple ``nmr_generate`` calls, the diversity bonus
    should only apply if the final answer diverges from *every one* of them,
    not just the most recent -- otherwise the model could echo an earlier
    call's top hit and still collect the bonus.
    """
    called = False
    first_results: list[str] = []
    seen: set[str] = set()
    pending_names: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            names = _assistant_tool_call_names(message)
            if _NMR_GENERATE_TOOL_NAME in names:
                called = True
            pending_names.extend(names)
            continue
        if role in ("tool", "user") and pending_names:
            payload = _parse_tool_response_payload(_content_text(message.get("content")))
            if payload is None:
                continue
            name = pending_names.pop(0)
            if name != _NMR_GENERATE_TOOL_NAME:
                continue
            result = _nmr_generate_first_candidate(payload)
            if result is not None and result not in seen:
                seen.add(result)
                first_results.append(result)
    return called, first_results


def _nmr_generate_first_candidate(payload: Mapping[str, Any]) -> str | None:
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, Mapping):
        return None
    value = first.get("canonical_smiles") or first.get("smiles")
    if isinstance(value, str) and value.strip():
        canonical, _ = _canonicalize_smiles(value.strip())
        return canonical or value.strip()
    return None


def _schema_map(schemas: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for schema in schemas:
        function = schema.get("function") if isinstance(schema.get("function"), Mapping) else schema
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str):
            out[name] = dict(function)
    return out


def _validate_tool_call(call: Mapping[str, Any], schemas: Mapping[str, JsonDict], *, index: int) -> list[str]:
    prefix = f"tool call {index}"
    function = call.get("function") if isinstance(call.get("function"), Mapping) else call
    name = function.get("name") if isinstance(function, Mapping) else None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    errors: list[str] = []
    if not isinstance(name, str) or not name.strip():
        errors.append(f"{prefix}: missing function name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            errors.append(f"{prefix}: arguments are invalid JSON: {exc.msg}")
            return errors
    if not isinstance(arguments, Mapping):
        errors.append(f"{prefix}: arguments must be a JSON object")
        return errors
    if schemas:
        schema = schemas.get(str(name))
        if schema is None:
            errors.append(f"{prefix}: unknown tool {name!r}")
        else:
            errors.extend(
                f"{prefix}: {error}"
                for error in _validate_json_schema(arguments, schema.get("parameters", {}), path="arguments")
            )
    return errors


def _validate_json_schema(value: Any, schema: Any, *, path: str) -> list[str]:
    """Validate the JSON-Schema subset used by Spectune tool parameters."""
    if not isinstance(schema, Mapping):
        return []
    errors: list[str] = []
    expected = schema.get("type")
    type_matches = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if isinstance(expected, str) and expected in type_matches and not type_matches[expected]:
        return [f"{path} must be {expected}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']!r}")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required if isinstance(required, list) else []:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        if isinstance(properties, Mapping):
            if schema.get("additionalProperties") is False:
                for key in value:
                    if key not in properties:
                        errors.append(f"{path}.{key} is not allowed")
            for key, child in value.items():
                if key in properties:
                    errors.extend(_validate_json_schema(child, properties[key], path=f"{path}.{key}"))
    elif isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            errors.extend(_validate_json_schema(child, schema["items"], path=f"{path}[{index}]"))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")
    return errors


def _extract_answer_candidates(text: str, *, strict: bool) -> list[str]:
    if strict:
        return extract_v1_smiles(text)
    return _extract_legacy_answer_candidates(text)


def _has_explicit_smiles_answer(text: str, *, strict: bool) -> bool:
    """Return True if ``text`` contains a real ``{"smiles": [...]}`` object.

    Distinguishes an explicitly empty answer (e.g. ``{"smiles": []}``) from a
    malformed reply that simply failed to parse into candidates.
    """
    region = final_answer_region(text) if text else ""
    if not region:
        return False
    # Check bare JSON in the region
    for blob in [region] + re.findall(r"```(?:json)?\s*(.*?)```", region, re.DOTALL | re.IGNORECASE):
        blob = blob.strip()
        if not blob:
            continue
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "smiles" in parsed:
            return True
    # Fallback: regex presence of the key (covers partial/non-strict blobs)
    return bool(re.search(r'"smiles"\s*:\s*\[', region))


def _extract_legacy_answer_candidates(text: str) -> list[str]:
    """Broader parsers retained for offline audits only."""
    candidates = list(extract_v1_smiles(text))
    parsed_values: list[Any] = []
    stripped = text.strip()
    if stripped:
        try:
            parsed_values.append(json.loads(stripped))
        except json.JSONDecodeError:
            for fenced in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
                try:
                    parsed_values.append(json.loads(fenced.strip()))
                except json.JSONDecodeError:
                    pass
    for value in parsed_values:
        candidates.extend(_candidates_from_json(value))
    for tagged in _SMILES_TAG_RE.findall(text):
        candidates.extend(part.strip() for part in re.split(r"[\n,]", tagged) if part.strip())
    for line in text.splitlines():
        match = _LABELED_LINE_RE.match(line) or _RANKED_LINE_RE.match(line)
        if match:
            token = _clean_candidate_token(match.group(1))
            if token and _SMILES_TOKEN_RE.fullmatch(token):
                candidates.append(token)
    if not candidates and "\n" not in stripped:
        token = _clean_candidate_token(stripped)
        if token and _SMILES_TOKEN_RE.fullmatch(token):
            candidates.append(token)
    return _unique_strings(candidates)


def _candidates_from_json(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        for key in _ANSWER_KEYS:
            if key in value:
                return _string_values(value[key])
    return []


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping):
                for key in ("smiles", "canonical_smiles", "answer"):
                    if isinstance(item.get(key), str):
                        out.append(item[key])
                        break
        return out
    return []


def _clean_candidate_token(value: str) -> str:
    return value.strip().strip("`'\"").rstrip(",;")


def _unique_strings(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _ground_truth_smiles(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            try:
                return _ground_truth_smiles(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return stripped
    if isinstance(value, Mapping):
        for key in ("gt_smiles", "ground_truth", "smiles", "answer"):
            if key in value:
                return _ground_truth_smiles(value[key])
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return _ground_truth_smiles(value[0]) if value else ""
    return ""


def _canonicalize_smiles(smiles: str) -> tuple[str | None, bool]:
    text = str(smiles or "").strip()
    if not text:
        return None, True
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError:
        return text, False
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None, True
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True), True


__all__ = ["RewardEvaluator"]
