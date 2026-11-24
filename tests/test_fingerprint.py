"""Tests for rdworks.fingerprint and MolLibr.pick_diverse()."""

import pytest
from rdkit import Chem, DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from rdworks import Mol, MolLibr
from rdworks.fingerprint import calculate_fingerprints, distance_function


FP_TYPES = ("morgan", "rdkit", "maccs")


SMILES = [
    "CCO",
    "CCCO",
    "CCCCO",
    "c1ccccc1",
    "c1ccccc1C",
    "c1ccncc1",
    "CC(=O)O",
    "CC(=O)OC",
    "C1CCCCC1",
    "OC1CCCCC1",
]
NAMES = [f"m{i}" for i in range(len(SMILES))]


@pytest.fixture
def libr() -> MolLibr:
    return MolLibr(SMILES, NAMES)


##################################################
### calculate_fingerprints / distance_function
##################################################


@pytest.mark.parametrize(
    "fp_type, num_bits", [("morgan", 2048), ("rdkit", 2048), ("maccs", 167)]
)
def test_fingerprint_types(fp_type, num_bits):
    fps = calculate_fingerprints([Chem.MolFromSmiles(s) for s in SMILES], fp_type)
    assert len(fps) == len(SMILES)
    assert all(fp.GetNumBits() == num_bits for fp in fps)


@pytest.mark.parametrize("fp_type", FP_TYPES)
def test_fingerprints_accept_rdworks_mol(fp_type):
    rdmols = [Chem.MolFromSmiles(s) for s in SMILES]
    from_rdkit = calculate_fingerprints(rdmols, fp_type)
    from_rdworks = calculate_fingerprints([Mol(s) for s in SMILES], fp_type)
    assert from_rdworks == from_rdkit


@pytest.mark.parametrize("radius, n_bits", [(2, 2048), (3, 1024)])
def test_morgan_matches_rdworks_mol_fingerprint(radius, n_bits):
    # same bits as RDKit's Morgan generator used for Mol.get_similarity()
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rdmols = [Chem.MolFromSmiles(s) for s in SMILES]
    fps = calculate_fingerprints(rdmols, "morgan", radius=radius, n_bits=n_bits)
    assert fps == [generator.GetFingerprint(m) for m in rdmols]


def test_fingerprints_none_entries():
    fps = calculate_fingerprints([Chem.MolFromSmiles("CCO"), None])
    assert fps[0] is not None
    assert fps[1] is None


def test_fingerprints_unknown_type():
    with pytest.raises(ValueError, match="Unknown fingerprint type"):
        calculate_fingerprints([Chem.MolFromSmiles("CCO")], "ecfp")


def test_distance_function():
    fps = calculate_fingerprints([Chem.MolFromSmiles(s) for s in ("CCO", "CCO", "c1ccccc1")])
    assert distance_function(0, 1, fps) == 0.0
    assert distance_function(0, 2, fps) == pytest.approx(
        1.0 - DataStructs.TanimotoSimilarity(fps[0], fps[2])
    )
    assert 0.0 < distance_function(0, 2, fps) <= 1.0


##################################################
### MolLibr.pick_diverse
##################################################


def _reference_picks(fp_type: str, n: int, seed: int) -> list[str]:
    """Independent MaxMin selection using an explicit Tanimoto distance function."""
    fps = calculate_fingerprints([Chem.MolFromSmiles(s) for s in SMILES], fp_type)
    picks = MaxMinPicker().LazyPick(lambda i, j: distance_function(i, j, fps), len(fps), n, seed=seed)
    return [NAMES[i] for i in sorted(picks)]


@pytest.mark.parametrize("fp_type", FP_TYPES)
@pytest.mark.parametrize("n, seed", [(4, 1), (3, 0)])
def test_pick_diverse_matches_maxmin(libr, fp_type, n, seed):
    picked = libr.pick_diverse(n, seed=seed, fp_type=fp_type)
    assert picked.count() == n
    assert [m.name for m in picked] == _reference_picks(fp_type, n, seed)


def test_pick_diverse_keeps_picked_not_first_molecules(libr):
    # the old mask kept the first n molecules regardless of the picks
    picked = [m.name for m in libr.pick_diverse(4, seed=1)]
    assert picked != NAMES[:4]
    assert len(set(picked)) == 4


def test_pick_diverse_keeps_library_order(libr):
    picked = [m.name for m in libr.pick_diverse(5, seed=2)]
    assert picked == sorted(picked, key=NAMES.index)


def test_pick_diverse_all(libr):
    assert [m.name for m in libr.pick_diverse(len(SMILES))] == NAMES


def test_pick_diverse_too_many(libr):
    with pytest.raises(ValueError, match="Requested 20 molecules"):
        libr.pick_diverse(20)
