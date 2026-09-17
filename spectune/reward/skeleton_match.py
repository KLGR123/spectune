"""Lenient SMILES comparison used as a fallback GT-matching pass.

Migrated from ``rank_smiles.py``. Where :mod:`spectune.reward.evaluator`'s
exact match requires identical RDKit canonical *isomeric* SMILES, this module
normalizes away distinctions that don't change the underlying skeleton for
ranking purposes:

- atom-map numbers and explicit Hs are stripped;
- unspecified E/Z alkene stereo is forced to trans (except alkenes next to a
  terminal nitrogen, where geometry is often ambiguous/unreliable);
- a lone (single) stereocenter has its chirality dropped entirely;
- a molecule with 2+ stereocenters is folded together with its full-inversion
  enantiomer (same key for both), so mirror-image predictions still count as
  a hit while true diastereomers (partial inversion) remain distinct.

Only used when the stricter exact match misses, and only if RDKit is
installed -- :func:`get_skeleton_key` returns ``None`` (not ``""``) when
RDKit is unavailable so callers can distinguish "no key" from "unavailable".
"""

from __future__ import annotations

from typing import Any


def _canonical_no_map_no_h(mol: Any) -> str:
    from rdkit import Chem

    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    m = Chem.RemoveHs(m, implicitOnly=False)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def _find_chiral_center_indices(mol: Any) -> list[int]:
    from rdkit import Chem

    m = Chem.Mol(mol)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    return sorted({int(idx) for idx, _ in centers})


def _clear_single_chiral_center(mol: Any, center_idx: int) -> Any:
    from rdkit import Chem

    m = Chem.Mol(mol)
    atom = m.GetAtomWithIdx(int(center_idx))
    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    if atom.HasProp("_CIPCode"):
        atom.ClearProp("_CIPCode")
    return m


def _invert_all_tetrahedral_centers(mol: Any, center_indices: list[int]) -> Any:
    from rdkit import Chem

    m = Chem.Mol(mol)
    for idx in center_indices:
        atom = m.GetAtomWithIdx(int(idx))
        tag = atom.GetChiralTag()
        if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
    return m


def _alkene_has_terminal_nitrogen(bond: Any) -> bool:
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    if begin_atom.GetAtomicNum() == 7 or end_atom.GetAtomicNum() == 7:
        return True
    for atom, other in ((begin_atom, end_atom), (end_atom, begin_atom)):
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() == other.GetIdx():
                continue
            if nbr.GetAtomicNum() == 7:
                return True
    return False


def _clear_neighbor_single_bond_dirs(double_bond: Any) -> None:
    from rdkit import Chem

    for atom in (double_bond.GetBeginAtom(), double_bond.GetEndAtom()):
        for neighbor_bond in atom.GetBonds():
            if neighbor_bond.GetIdx() == double_bond.GetIdx():
                continue
            if neighbor_bond.GetBondType() == Chem.BondType.SINGLE:
                neighbor_bond.SetBondDir(Chem.BondDir.NONE)


def _force_unspecified_alkene_to_trans(mol: Any) -> Any:
    """Treat unspecified E/Z double bonds as trans (E); explicit labels are kept."""
    from rdkit import Chem

    m = Chem.Mol(mol)
    Chem.FindPotentialStereoBonds(m)
    stereo_changed = False
    for bond in m.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        if _alkene_has_terminal_nitrogen(bond):
            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                _clear_neighbor_single_bond_dirs(bond)
                stereo_changed = True
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            bond.SetStereo(Chem.BondStereo.STEREOE)
            stereo_changed = True
    if stereo_changed:
        Chem.SetDoubleBondNeighborDirections(m)
    return m


def get_skeleton_key(smiles: str) -> str | None:
    """Normalize ``smiles`` into a lenient comparison key.

    Returns ``""`` for empty input, the raw (stripped) string if RDKit can't
    parse it or normalization otherwise fails, and ``None`` if RDKit itself
    isn't installed (callers should treat that as "lenient matching
    unavailable", distinct from "no match").
    """
    if not smiles:
        return ""
    raw = str(smiles).strip()
    try:
        from rdkit import Chem
    except ImportError:
        return None
    try:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return raw

        mol = _force_unspecified_alkene_to_trans(mol)
        centers = _find_chiral_center_indices(mol)

        if not centers:
            return _canonical_no_map_no_h(mol)
        if len(centers) == 1:
            return _canonical_no_map_no_h(_clear_single_chiral_center(mol, centers[0]))

        can_self = _canonical_no_map_no_h(mol)
        can_enantiomer = _canonical_no_map_no_h(_invert_all_tetrahedral_centers(mol, centers))
        return "||".join(sorted([can_self, can_enantiomer]))
    except Exception:
        return raw


def is_same_molecule_for_rank(smiles_a: str, smiles_b: str) -> bool:
    """Return True if two SMILES represent the same molecule under the lenient rank rules."""
    if not smiles_a or not smiles_b:
        return False
    key_a = get_skeleton_key(smiles_a)
    key_b = get_skeleton_key(smiles_b)
    if not key_a or key_b is None:
        return False
    return key_a == key_b


__all__ = ["get_skeleton_key", "is_same_molecule_for_rank"]
