"""Resolve compound names from PubChem's PUG REST API.

The lookup follows PubChem's identifier graph rather than searching arbitrary
web pages:

``canonical SMILES -> PubChem CID -> registered synonyms``

The synonym list is classified locally into Chinese names, English names, and
CAS registry numbers. A result is considered complete only when both a Chinese
and an English name are present. Partial results remain in the enrichment
payload for analysis, but query construction must not use either name.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .config import EnrichmentConfig

JsonDict = dict[str, Any]

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
_LATIN_RE = re.compile(r"[A-Za-z]")
_IDENTIFIER_RE = re.compile(
    r"^(?:CID|CHEBI|CHEMBL|DB|DTXSID|DTXCID|EC |EINECS|HMDB|InChI|InChIKey|NSC|SCHEMBL|UNII|ZINC)",
    re.IGNORECASE,
)

_PUBCHEM_NAME_SOURCE = "pubchem_pug_rest_v1"


def empty_names(status: str = "not_found") -> JsonDict:
    return {
        "name_zh": None,
        "name_en": None,
        "cas": None,
        "cid": None,
        "status": status,
        "complete": False,
        "source": _PUBCHEM_NAME_SOURCE,
        "name_zh_candidates": [],
        "name_en_candidates": [],
        "cas_candidates": [],
        "evidence": [],
    }


async def resolve_names(
    smiles: str,
    config: EnrichmentConfig,
) -> JsonDict:
    """Resolve one SMILES through PubChem and classify its synonym list."""
    from spectune.tools.utils import strict_canonical_smiles

    canonical = strict_canonical_smiles(smiles)
    if not canonical:
        return empty_names("invalid_smiles")

    encoded = urllib.parse.quote(canonical, safe="")
    base = config.pubchem_base_url.rstrip("/")
    cid_url = f"{base}/compound/smiles/{encoded}/cids/JSON"
    try:
        cid_payload = await _get_json_with_retries(cid_url, config)
        cids = (cid_payload.get("IdentifierList") or {}).get("CID") or []
        if not cids:
            return empty_names("not_found")
        cid = int(cids[0])

        synonyms_url = f"{base}/compound/cid/{cid}/synonyms/JSON"
        synonyms_payload = await _get_json_with_retries(synonyms_url, config)
    except urllib.error.HTTPError as exc:
        return empty_names("not_found" if exc.code == 404 else "request_failed")
    except Exception:
        return empty_names("request_failed")

    information = (synonyms_payload.get("InformationList") or {}).get("Information") or []
    synonyms = information[0].get("Synonym") if information and isinstance(information[0], dict) else []
    if not isinstance(synonyms, list):
        synonyms = []

    name_zh_candidates = _unique(
        synonym for synonym in synonyms if isinstance(synonym, str) and _is_chinese_name(synonym)
    )
    name_en_candidates = _unique(
        synonym for synonym in synonyms if isinstance(synonym, str) and _is_english_name(synonym)
    )
    cas_candidates = _unique(
        synonym for synonym in synonyms if isinstance(synonym, str) and _CAS_RE.fullmatch(synonym.strip())
    )
    limit = config.name_max_candidates
    name_zh = name_zh_candidates[0] if name_zh_candidates else None
    name_en = name_en_candidates[0] if name_en_candidates else None
    complete = bool(name_zh and name_en)
    status = "found" if complete else ("partial" if name_zh or name_en else "not_found")
    return {
        "name_zh": name_zh,
        "name_en": name_en,
        "cas": cas_candidates[0] if cas_candidates else None,
        "cid": cid,
        "status": status,
        "complete": complete,
        "source": _PUBCHEM_NAME_SOURCE,
        "name_zh_candidates": name_zh_candidates[:limit],
        "name_en_candidates": name_en_candidates[:limit],
        "cas_candidates": cas_candidates[:limit],
        "evidence": [synonyms_url],
    }


async def _get_json_with_retries(url: str, config: EnrichmentConfig) -> JsonDict:
    last_error: Exception | None = None
    for attempt in range(config.pubchem_max_retries + 1):
        try:
            return await asyncio.to_thread(_get_json, url, config.pubchem_timeout_s)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt >= config.pubchem_max_retries:
                raise
            last_error = exc
        except Exception as exc:
            if attempt >= config.pubchem_max_retries:
                raise
            last_error = exc
        await asyncio.sleep(1.0 * (attempt + 1))
    raise RuntimeError("PubChem request failed") from last_error


def _get_json(url: str, timeout_s: float) -> JsonDict:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "spectune/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body) if body else {}
    return parsed if isinstance(parsed, dict) else {}


def _is_chinese_name(value: str) -> bool:
    text = value.strip()
    return bool(1 < len(text) <= 160 and _CHINESE_RE.search(text))


def _is_english_name(value: str) -> bool:
    text = value.strip()
    if not 2 <= len(text) <= 200 or _CHINESE_RE.search(text) or not _LATIN_RE.search(text):
        return False
    if _CAS_RE.fullmatch(text) or _IDENTIFIER_RE.match(text):
        return False
    return not text.startswith(("CHEMBL", "DTXSID", "DTXCID", "SCHEMBL", "UNII-", "ZINC"))


def _unique(values: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


__all__ = ["empty_names", "resolve_names"]
