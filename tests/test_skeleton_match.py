import pytest

from spectune.reward.skeleton_match import get_skeleton_key, is_same_molecule_for_rank


def test_empty_input_returns_empty_string():
    assert get_skeleton_key("") == ""
    assert get_skeleton_key(None) == ""


def test_unparseable_smiles_falls_back_to_raw_text():
    assert get_skeleton_key("not-a-smiles") == "not-a-smiles"


def test_achiral_molecules_use_plain_canonical_form():
    assert get_skeleton_key("CCO") == get_skeleton_key("OCC")
    assert is_same_molecule_for_rank("CCO", "OCC")
    assert not is_same_molecule_for_rank("CCO", "CCN")


def test_lone_stereocenter_chirality_is_ignored():
    assert get_skeleton_key("C[C@H](N)C(=O)O") == get_skeleton_key("C[C@@H](N)C(=O)O")
    assert is_same_molecule_for_rank("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O")


def test_full_inversion_enantiomer_matches_for_multi_center_molecule():
    full_inversion = "C[C@@H](Cl)[C@@H](Cl)C"
    original = "C[C@H](Cl)[C@H](Cl)C"
    assert is_same_molecule_for_rank(original, full_inversion)


def test_partial_inversion_diastereomer_does_not_match():
    diastereomer = "C[C@H](Cl)[C@@H](Cl)C"
    original = "C[C@H](Cl)[C@H](Cl)C"
    assert not is_same_molecule_for_rank(original, diastereomer)


def test_unspecified_alkene_stereo_is_treated_as_trans():
    assert get_skeleton_key("CC=CC") == get_skeleton_key("C/C=C/C")


@pytest.mark.parametrize("smiles_a,smiles_b", [("", "CCO"), ("CCO", ""), (None, "CCO")])
def test_is_same_molecule_for_rank_requires_both_inputs(smiles_a, smiles_b):
    assert not is_same_molecule_for_rank(smiles_a, smiles_b)
