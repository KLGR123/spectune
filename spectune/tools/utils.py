"""Shared utilities used across Spectune tools.

Organised into three sections:
  - Generic helpers  (_as_list, _float_list, extract_last_json, post_json)
  - Chemistry        (has_rdkit, canonical_smiles, molecular_formula)
  - NMR spectrum     (normalize_h/c_peaks, build_*_from_shifts, spectrum_payload,
                      infer_nmr_type, experimental_shift_lists)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

JsonDict = dict[str, Any]


# Generic helpers


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def float_list(value: Any) -> list[float]:
    out: list[float] = []
    for item in as_list(value):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def extract_last_json(text: str) -> Any:
    """Return the last valid JSON object/array found in ``text``, or ``None``."""
    decoder = json.JSONDecoder()
    last: Any = None
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        last = value
    return last


def post_json(url: str, payload: JsonDict, *, timeout: float) -> JsonDict:
    """POST ``payload`` as JSON to ``url`` and return the decoded response dict."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {"data": parsed}


# Chemistry


def has_rdkit() -> bool:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return True


_rdkit_quieted = False


def _rdkit_chem() -> Any:
    """Import and return ``rdkit.Chem``, or ``None`` if rdkit is not installed.

    Also disables RDKit's own C++-side logger (``rdApp.*``) the first time this
    is called. That logger prints straight to stderr outside Python's
    ``logging`` module (e.g. "Conflicting single bond directions..." on
    ambiguous stereo), so it can't be silenced any other way, and doing it once
    per row when canonicalizing a large dataset is itself real overhead.
    """
    global _rdkit_quieted
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError:
        return None
    if not _rdkit_quieted:
        from rdkit import RDLogger  # type: ignore[import-not-found]

        RDLogger.DisableLog("rdApp.*")
        _rdkit_quieted = True
    return Chem


def canonical_smiles(smiles: str) -> str:
    """Canonicalize with RDKit when available, else pass the text through."""
    text = str(smiles or "").strip()
    if not text:
        return ""
    Chem = _rdkit_chem()
    if Chem is None:
        return text
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return text
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def strict_canonical_smiles(smiles: str) -> str | None:
    """Canonicalize with RDKit, returning ``None`` when the input is not valid.

    Unlike :func:`canonical_smiles` (which passes invalid text through so display
    code always has *something* to show), this is for callers that must reject
    bad input outright, e.g. filtering reactant SMILES before reactant-set-overlap
    scoring. Without RDKit installed the input cannot be validated at all, so it
    is returned unchanged (opaque pass-through, same as ``canonical_smiles``).
    """
    text = str(smiles or "").strip()
    if not text:
        return None
    Chem = _rdkit_chem()
    if Chem is None:
        return text
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def molecular_formula(smiles: str) -> str | None:
    """Best-effort molecular formula; ``None`` if RDKit is unavailable or parsing fails."""
    text = str(smiles or "").strip()
    if not text:
        return None
    Chem = _rdkit_chem()
    if Chem is None:
        return None
    from rdkit.Chem import rdMolDescriptors  # type: ignore[import-not-found]

    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    return str(rdMolDescriptors.CalcMolFormula(molecule))


def canonical_smiles_and_formula(smiles: str) -> tuple[str, str | None]:
    """Canonicalize ``smiles`` and compute its molecular formula from a single RDKit parse.

    Equivalent to calling :func:`canonical_smiles` then :func:`molecular_formula`,
    but parses the input once instead of twice. ``Chem.MolFromSmiles`` (parsing +
    sanitization) dominates per-call cost, so this roughly halves RDKit time when
    both values are needed for every row of a large dataset (e.g. NMRexp
    preprocessing). Falls back to ``(text, None)`` if RDKit is unavailable or the
    SMILES does not parse, matching ``canonical_smiles``/``molecular_formula``.
    """
    text = str(smiles or "").strip()
    if not text:
        return "", None
    Chem = _rdkit_chem()
    if Chem is None:
        return text, None
    from rdkit.Chem import rdMolDescriptors  # type: ignore[import-not-found]

    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return text, None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    formula = str(rdMolDescriptors.CalcMolFormula(molecule))
    return canonical, formula


# NMR spectrum helpers


def normalize_h_peaks(peaks: Any) -> list[JsonDict]:
    out: list[JsonDict] = []
    for item in as_list(peaks):
        if not isinstance(item, dict):
            continue
        peak = dict(item)
        shift = peak.get("centroid", peak.get("delta", peak.get("delta (ppm)", 0.0)))
        try:
            shift_f = float(shift)
        except (TypeError, ValueError):
            shift_f = 0.0
        category = str(peak.get("category") or peak.get("multiplicity") or peak.get("split") or "m").strip() or "m"
        peak.setdefault("category", category)
        peak.setdefault("centroid", shift_f)
        peak.setdefault("delta", shift_f)
        peak.setdefault("delta (ppm)", shift_f)
        peak.setdefault("multiplicity", category)
        out.append(peak)
    return out


def normalize_c_peaks(peaks: Any) -> list[JsonDict]:
    out: list[JsonDict] = []
    for item in as_list(peaks):
        if isinstance(item, dict):
            peak = dict(item)
            value = peak.get("delta (ppm)", peak.get("delta", peak.get("centroid")))
        else:
            peak = {}
            value = item
        try:
            shift_f = float(value)
        except (TypeError, ValueError):
            continue
        peak.setdefault("delta (ppm)", shift_f)
        peak.setdefault("delta", shift_f)
        out.append(peak)
    return out


def build_h_peaks_from_shifts(h_shifts: Any, h_split: Any = None) -> list[JsonDict]:
    splits = as_list(h_split)
    out: list[JsonDict] = []
    for index, shift in enumerate(as_list(h_shifts)):
        try:
            shift_f = float(shift)
        except (TypeError, ValueError):
            continue
        category = "m"
        if index < len(splits) and isinstance(splits[index], str) and splits[index].strip():
            category = splits[index].strip()
        out.append(
            {
                "category": category,
                "centroid": shift_f,
                "delta": shift_f,
                "delta (ppm)": shift_f,
                "multiplicity": category,
            }
        )
    return out


def build_c_peaks_from_shifts(c_shifts: Any) -> list[JsonDict]:
    out: list[JsonDict] = []
    for shift in as_list(c_shifts):
        try:
            out.append({"delta (ppm)": float(shift)})
        except (TypeError, ValueError):
            continue
    return out


def spectrum_payload(arguments: dict[str, Any]) -> JsonDict:
    """Build the ``{h_nmr_peaks, c_nmr_peaks, molecular_formula, nmr_solvent}``
    payload NMR backends expect, from either explicit peak lists or raw shift arrays.
    """
    h_peaks = normalize_h_peaks(arguments.get("h_nmr_peaks")) or build_h_peaks_from_shifts(
        arguments.get("h_shifts"), arguments.get("h_split")
    )
    c_peaks = normalize_c_peaks(arguments.get("c_nmr_peaks")) or build_c_peaks_from_shifts(arguments.get("c_shifts"))
    payload: JsonDict = {"h_nmr_peaks": h_peaks, "c_nmr_peaks": c_peaks}
    formula = str(arguments.get("formula") or arguments.get("molecular_formula") or "").strip()
    if formula:
        payload["molecular_formula"] = formula
    solvent = str(arguments.get("solvent") or arguments.get("nmr_solvent") or "").strip()
    if solvent:
        payload["nmr_solvent"] = solvent
    return payload


def infer_nmr_type(spectrum: JsonDict) -> str:
    parts = ""
    if spectrum.get("c_nmr_peaks"):
        parts += "C"
    if spectrum.get("h_nmr_peaks"):
        parts += "H"
    if spectrum.get("molecular_formula"):
        parts += "F"
    return parts or "CHF"


def spectrum_shift_arrays(arguments: dict[str, Any]) -> tuple[list[float], list[str], list[float]]:
    """Return ``(h_shifts, h_split, c_shifts)`` aligned arrays for search-style backends.

    Unlike :func:`experimental_shift_lists`, the H-shift/H-split pair here is kept in
    input order (not sorted) since multiplicities must stay aligned with their shift.
    """
    h_peaks = normalize_h_peaks(arguments.get("h_nmr_peaks")) or build_h_peaks_from_shifts(
        arguments.get("h_shifts"), arguments.get("h_split")
    )
    c_peaks = normalize_c_peaks(arguments.get("c_nmr_peaks")) or build_c_peaks_from_shifts(arguments.get("c_shifts"))
    h_shifts = [float(peak["centroid"]) for peak in h_peaks]
    h_split = [str(peak.get("category") or "m") for peak in h_peaks]
    c_shifts = [float(peak["delta (ppm)"]) for peak in c_peaks]
    return h_shifts, h_split, c_shifts


def experimental_shift_lists(arguments: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Sorted experimental H/C shift lists from explicit args or a nested ``spectrum``."""
    spectrum = arguments.get("spectrum") if isinstance(arguments.get("spectrum"), dict) else {}
    h_values = float_list(arguments.get("h_shifts") or spectrum.get("H_shifts") or spectrum.get("h_shifts"))
    c_values = float_list(arguments.get("c_shifts") or spectrum.get("C_shifts") or spectrum.get("c_shifts"))
    if not h_values:
        h_peaks = normalize_h_peaks(spectrum.get("h_nmr_peaks"))
        h_values = float_list([peak.get("centroid", peak.get("delta")) for peak in h_peaks])
    if not c_values:
        c_peaks = normalize_c_peaks(spectrum.get("c_nmr_peaks"))
        c_values = float_list([peak.get("delta (ppm)", peak.get("delta")) for peak in c_peaks])
    return sorted(h_values), sorted(c_values)
