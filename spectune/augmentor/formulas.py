"""Small, traceable perturbations for molecular-formula robustness tests."""

from __future__ import annotations

import random
import re
from typing import Any

JsonDict = dict[str, Any]

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def perturb_formula(formula: str, rng: random.Random) -> tuple[str, JsonDict]:
    """Return a nearby formula and an exact description of the atom edits.

    Candidates use only small composition changes: ``H±2``, ``O±1``,
    ``N±1``, or one homologous-unit change (``C±1, H±2``). This keeps the
    condition chemically close without pretending to preserve valence or
    identify a real molecule.
    """
    counts = _parse_formula(formula)
    if not counts:
        return formula, {
            "applied": False,
            "changed": False,
            "original_formula": formula,
            "perturbed_formula": formula,
            "reason": "unsupported_formula",
        }

    candidates: list[tuple[str, dict[str, int]]] = []
    _add_candidate(candidates, counts, "hydrogen_plus_2", {"H": 2})
    _add_candidate(candidates, counts, "hydrogen_minus_2", {"H": -2})
    _add_candidate(candidates, counts, "oxygen_plus_1", {"O": 1})
    _add_candidate(candidates, counts, "oxygen_minus_1", {"O": -1})
    _add_candidate(candidates, counts, "nitrogen_plus_1", {"N": 1})
    _add_candidate(candidates, counts, "nitrogen_minus_1", {"N": -1})
    _add_candidate(candidates, counts, "homolog_plus_CH2", {"C": 1, "H": 2})
    _add_candidate(candidates, counts, "homolog_minus_CH2", {"C": -1, "H": -2})

    if not candidates:
        return formula, {
            "applied": False,
            "changed": False,
            "original_formula": formula,
            "perturbed_formula": formula,
            "reason": "no_valid_perturbation",
        }

    operation, deltas = rng.choice(candidates)
    perturbed_counts = dict(counts)
    for element, delta in deltas.items():
        perturbed_counts[element] = perturbed_counts.get(element, 0) + delta
        if perturbed_counts[element] == 0:
            del perturbed_counts[element]
    perturbed = _format_formula(perturbed_counts)
    return perturbed, {
        "applied": True,
        "changed": perturbed != formula,
        "original_formula": formula,
        "perturbed_formula": perturbed,
        "operation": operation,
        "element_deltas": deltas,
    }


def _parse_formula(formula: str) -> dict[str, int] | None:
    matches = list(_TOKEN_RE.finditer(formula))
    if not matches or "".join(match.group(0) for match in matches) != formula:
        return None
    counts: dict[str, int] = {}
    for match in matches:
        element = match.group(1)
        count = int(match.group(2) or 1)
        if count < 1:
            return None
        counts[element] = counts.get(element, 0) + count
    return counts


def _add_candidate(
    candidates: list[tuple[str, dict[str, int]]],
    counts: dict[str, int],
    operation: str,
    deltas: dict[str, int],
) -> None:
    valid = all(counts.get(element, 0) + delta >= 0 for element, delta in deltas.items())
    if deltas.get("C", 0) < 0 and counts.get("C", 0) + deltas["C"] < 1:
        valid = False
    if valid:
        candidates.append((operation, deltas))


def _format_formula(counts: dict[str, int]) -> str:
    order = [element for element in ("C", "H") if element in counts]
    order.extend(sorted(element for element in counts if element not in {"C", "H"}))
    return "".join(element + (str(counts[element]) if counts[element] != 1 else "") for element in order)


__all__ = ["perturb_formula"]
