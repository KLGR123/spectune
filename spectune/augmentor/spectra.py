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


def apply_reaction_noise(reaction: JsonDict, _rng: random.Random, _strength: float = 0.0) -> tuple[JsonDict, JsonDict]:
    """Reserved: corrupt reaction context the way :func:`apply_nmr_noise` corrupts spectra.

    Deliberately a no-op for now. The intended modes (dropping a reactant,
    swapping conditions, degrading a reaction class to a vaguer one) need the
    reaction-context schema to settle first, and
    :class:`~spectune.augmentor.config.AugmentorConfig` rejects a non-zero
    ``reaction_noise_ratio`` until then.
    """
    return reaction, {"mode": None, "applied": False}


__all__ = ["apply_nmr_noise", "apply_reaction_noise", "build_spectrum_text", "split_peaks"]
