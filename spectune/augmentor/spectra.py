"""Spectrum text rendering and deliberate corruption.

The clean rendering reassembles the NMRexp fields into the one-line form papers
and users actually write (``1H NMR (400 MHz, CDCl3): 8.52-8.49 (m, 1H), ...``).

The corruption modes exist for robustness experiments, and each mirrors a way
real pasted spectra are degraded: peak lists stripped down to bare numbers,
punctuation lost to a bad copy/paste, values mistyped, or peaks simply missing.
Corruption never touches the ground-truth structure, so a noised row keeps its
label and the run records exactly which mode was applied.

Reaction-information noise is reserved for a later experiment; see
:func:`apply_reaction_noise`.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

JsonDict = dict[str, Any]

# A leading minus only counts as a sign when it does not follow another number:
# "8.52-8.49" is a shift range, while " -63.4" is a genuine negative shift.
_NUMBER_RE = re.compile(r"(?<![\d.])-?\d+\.\d+|(?<![\d.])-?\d+")
_FLOAT_RE = re.compile(r"\d+\.\d+")


def build_spectrum_text(nmr: JsonDict | None) -> str:
    """Render one NMRexp ``nmr`` block as a conventional single-line spectrum."""
    nmr = nmr or {}
    shift_text = str(nmr.get("shift_text") or "").strip()
    if not shift_text:
        return ""
    nmr_type = str(nmr.get("type") or "").strip()
    conditions = ", ".join(
        part for part in (str(nmr.get("frequency") or "").strip(), str(nmr.get("solvent") or "").strip()) if part
    )
    head = nmr_type
    if conditions:
        head = f"{head} ({conditions})" if head else f"({conditions})"
    return f"{head}: {shift_text}" if head else shift_text


def split_peaks(shift_text: str) -> list[str]:
    """Split a shift list on the commas that separate peaks, ignoring nested ones."""
    peaks: list[str] = []
    depth = 0
    current: list[str] = []
    for character in shift_text:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth = max(depth - 1, 0)
        if character == "," and depth == 0:
            peaks.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    tail = "".join(current).strip()
    if tail:
        peaks.append(tail)
    return [peak for peak in peaks if peak]


def _digits_only(text: str, _rng: random.Random, _strength: float) -> tuple[str, JsonDict]:
    numbers = _NUMBER_RE.findall(text)
    return " ".join(numbers), {"kept_numbers": len(numbers)}


def _strip_punctuation(text: str, rng: random.Random, strength: float) -> tuple[str, JsonDict]:
    removed = 0
    characters: list[str] = []
    for character in text:
        if character in "(),[]" and rng.random() < strength:
            removed += 1
            continue
        characters.append(character)
    return "".join(characters), {"removed_characters": removed}


def _perturb_values(text: str, rng: random.Random, strength: float) -> tuple[str, JsonDict]:
    perturbed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal perturbed
        if rng.random() >= strength:
            return match.group(0)
        value = float(match.group(0))
        # Coupling constants live on a Hz scale, chemical shifts on a ppm scale;
        # a plausible typo differs by a scale-appropriate amount, not a fixed one.
        following = text[match.end() : match.end() + 4]
        magnitude = 0.9 if "Hz" in following or value > 50 else 0.08
        perturbed += 1
        return f"{value + rng.uniform(-magnitude, magnitude):.2f}"

    return _FLOAT_RE.sub(replace, text), {"perturbed_values": perturbed}


def _drop_peaks(text: str, rng: random.Random, strength: float) -> tuple[str, JsonDict]:
    peaks = split_peaks(text)
    if len(peaks) < 2:
        return text, {"dropped_peaks": 0}
    kept = [peak for peak in peaks if rng.random() >= strength]
    if not kept:
        kept = [rng.choice(peaks)]
    return ", ".join(kept), {"dropped_peaks": len(peaks) - len(kept)}


_NOISE_FUNCTIONS = {
    "digits_only": _digits_only,
    "strip_punctuation": _strip_punctuation,
    "perturb_values": _perturb_values,
    "drop_peaks": _drop_peaks,
}


def apply_nmr_noise(
    nmr: JsonDict | None,
    mode: str,
    rng: random.Random,
    *,
    strength: float = 0.3,
) -> tuple[str, JsonDict]:
    """Return ``(noised_text, detail)`` for one spectrum under one noise ``mode``.

    ``digits_only`` drops the ``1H NMR (400 MHz, CDCl3)`` header as well, since
    a bare number sequence is precisely the case where the user pasted values
    with no context; the other modes corrupt the peak list but keep the header.
    """
    if mode not in _NOISE_FUNCTIONS:
        raise ValueError(f"unknown NMR noise mode {mode!r}; expected one of {sorted(_NOISE_FUNCTIONS)}")
    nmr = nmr or {}
    shift_text = str(nmr.get("shift_text") or "").strip()
    if not shift_text:
        return "", {"mode": mode, "applied": False}

    noised, detail = _NOISE_FUNCTIONS[mode](shift_text, rng, strength)
    if mode == "digits_only":
        text = noised
    else:
        text = build_spectrum_text({**nmr, "shift_text": noised})
    return text, {"mode": mode, "applied": True, "strength": strength, **detail}


# Maps NMR type keywords to Chinese labels used in the "labelled phrasing" rendering mode.
# Matched case-insensitively against the nmr["type"] field.
_NMR_TYPE_ZH: tuple[tuple[str, str], ...] = (
    ("1H",  "氢谱"),
    ("13C", "碳谱"),
    ("19F", "氟谱"),
    ("31P", "磷谱"),
    ("11B", "硼谱"),
    ("29Si","硅谱"),
    ("15N", "氮谱"),
    ("DEPT","DEPT谱"),
    ("COSY","COSY谱"),
    ("HSQC","HSQC谱"),
    ("HMBC","HMBC谱"),
    ("NOESY","NOESY谱"),
)

# Phrasing patterns for the Chinese-label rendering path.
# {label} → the Chinese label string, {data} → the spectrum text line.
_MULTIMODAL_LABEL_TEMPLATES: tuple[str, ...] = (
    "{label}是{data}",
    "{label}数据：{data}",
    "{label}如下：{data}",
    "{label}为{data}",
)
_MULTIMODAL_CONNECTORS: tuple[str, ...] = (
    "、",
    "；",
    "\n",
)


def _nmr_type_zh(nmr_type: str) -> str:
    """Return a Chinese label for ``nmr_type``, or the original string if unrecognised."""
    upper = nmr_type.upper()
    for keyword, label in _NMR_TYPE_ZH:
        if keyword.upper() in upper:
            return label
    return nmr_type


def build_multimodal_spectrum_text(
    nmr_list: Sequence[JsonDict | None],
    rng: random.Random | None = None,
) -> str:
    """Render multiple NMR blocks into one text block.

    When ``rng`` is supplied, randomly picks between two presentation styles:

    - **plain** (≈50 %): one spectrum per line, same as before.
    - **labelled** (≈50 %): Chinese-label phrasing such as
      "氢谱是1H NMR (400 MHz, CDCl3): …、碳谱是13C NMR (101 MHz, CDCl3): …"
      using a randomly chosen connector (、/；/newline) and sentence template.

    When ``rng`` is None the function always uses the plain style, keeping the
    behaviour of callers that do not participate in query construction (tests,
    ``nmr_text_clean`` etc.).

    Empty blocks (no shift_text) are silently skipped.
    """
    lines = [build_spectrum_text(nmr) for nmr in nmr_list]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    if len(lines) == 1 or rng is None or rng.random() < 0.5:
        return "\n".join(lines)
    # Labelled phrasing: pair each line with its Chinese NMR-type label.
    template = rng.choice(_MULTIMODAL_LABEL_TEMPLATES)
    connector = rng.choice(_MULTIMODAL_CONNECTORS)
    parts: list[str] = []
    for nmr, line in zip(nmr_list, lines):
        nmr_type = str((nmr or {}).get("type") or "").strip()
        label = _nmr_type_zh(nmr_type) if nmr_type else "谱图"
        parts.append(template.format(label=label, data=line))
    return connector.join(parts)


def apply_reaction_noise(reaction: JsonDict, _rng: random.Random, _strength: float = 0.0) -> tuple[JsonDict, JsonDict]:
    """Reserved: corrupt reaction context the way :func:`apply_nmr_noise` corrupts spectra.

    Deliberately a no-op for now. The intended modes (dropping a reactant,
    swapping conditions, degrading a reaction class to a vaguer one) need the
    reaction-context schema to settle first, and
    :class:`~spectune.augmentor.config.AugmentorConfig` rejects a non-zero
    ``reaction_noise_ratio`` until then.
    """
    return reaction, {"mode": None, "applied": False}


__all__ = [
    "apply_nmr_noise",
    "apply_reaction_noise",
    "build_multimodal_spectrum_text",
    "build_spectrum_text",
    "split_peaks",
]
