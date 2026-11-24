"""Regression tests for bug fixes in rgroup.py, matchedseries.py, and scaffold.py.

Covers the fixes listed under `Unreleased` in CHANGELOG.md.
"""

import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdworks import MatchedSeries
from rdworks.matchedseries import (
    calculate_bond_length,
    get_attachment_orientation,
    get_average_bond_length,
    standardize_rgroup_scales,
)
from rdworks.rgroup import connect_core_and_chains, expand_rgroup
from rdworks.scaffold import (
    BRICS_fragment_indices,
    breakup,
    connected_components,
    remove_exocyclic,
    rigid_fragment_indices,
)


CORE = "[*:1]-c1ccc2ccn(-[*:2])c2c1"
X = [
    "[*]C#N",
    "[*]C(O)=O",
    "[*]CO",
    "[*]COC",
    "[*]C(NC)=O",
    "[*]CNC(C)=O",
    "[*]CC=C",
    "[*][H]",
]
Y = ["[*][H]", "[*]O", "[*]OC", "[*]CC(F)(F)F", "[*]OCCOC"]


@pytest.fixture
def series():
    libr = expand_rgroup(core=CORE, r={1: X, 2: Y}, prefix="RGX", progress=False)
    return MatchedSeries(libr, sort_props=["QED", "HAC"])


##################################################
### rgroup
##################################################


def test_connect_core_and_chains_invalid_smiles_returns_none():
    # used to raise AttributeError: 'NoneType' object has no attribute 'GetAtoms'
    assert connect_core_and_chains("this is not a smiles") is None


def test_expand_rgroup_skips_failed_combinations():
    # the pentavalent carbon cannot be connected and used to crash
    # MolLibr.rename with AttributeError: 'NoneType' object has no attribute 'rename'
    libr = expand_rgroup(
        core="[*:1]c1ccncc1",
        r={1: ["[*]C#N", "[*]C(C)(C)(C)(C)C", "[*]CO"], 2: ["[*][H]"]},
        prefix="RGX",
        progress=False,
    )
    assert libr.count() == 2  # 2 of 3 combinations can be connected
    assert all(mol is not None for mol in libr.libr)
    assert all(mol.name.startswith("RGX") for mol in libr)


##################################################
### matchedseries
##################################################


def test_matchedseries_iterable_not_iterator(series):
    # __next__ was broken (always raised TypeError: 'list' object is not an
    # iterator) and has been removed, following Mol and MolLibr
    assert not hasattr(series, "__next__")
    assert len(list(series)) == series.count()
    with pytest.raises(TypeError):
        next(series)


def test_matchedseries_unknown_sort_prop_raises():
    # an unknown sort prop used to set None and crash with TypeError
    # during sorting (None is not comparable)
    libr = expand_rgroup(core=CORE, r={1: X, 2: Y}, prefix="RGX", progress=False)
    with pytest.raises(ValueError, match="unknown sort prop"):
        MatchedSeries(libr, sort_props=["BOGUS"])


def test_matchedseries_getitem_out_of_range(series):
    with pytest.raises(ValueError, match="index should be"):
        series[999]


def test_matchedseries_template(series):
    # the attribute used to be set as `template_coord2D` (typo) but read as
    # `template_coords2D`
    assert series.template_coords2D is None
    template = "[*:1]c1ccc2ccn(-[*:2])c2c1"
    rdmol = Chem.MolFromSmiles(template)
    series.template(template, rdmol)
    assert series.template_coords2D is not None
    assert len(series.template_coords2D) == 11


def test_matchedseries_helpers_do_not_modify_input():
    # 2D coordinates used to be computed in-place on the input molecules
    mol = Chem.MolFromSmiles("[*:1]CCO")
    get_attachment_orientation(mol)
    assert mol.GetNumConformers() == 0

    mol = Chem.MolFromSmiles("CCO")
    calculate_bond_length(mol, 0, 1)
    assert mol.GetNumConformers() == 0

    mol = Chem.MolFromSmiles("CCO")
    get_average_bond_length(mol)
    assert mol.GetNumConformers() == 0

    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1O")]
    standardized = standardize_rgroup_scales(mols)
    assert all(m.GetNumConformers() == 0 for m in mols)
    assert len(standardized) == 2


def test_matchedseries_template_does_not_modify_input(series):
    template = "[*:1]c1ccc2ccn(-[*:2])c2c1"
    rdmol = Chem.MolFromSmiles(template)
    series.template(template, rdmol)
    assert rdmol.GetNumConformers() == 0


##################################################
### scaffold
##################################################


def test_breakup_processes_all_parents():
    # a parent with only one ring used to terminate the whole batch
    # (early `return parents`), skipping later parents that still had children
    benzene = Chem.MolFromSmiles("c1ccccc1")
    naphthalene = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    parents = [
        (benzene, "c1ccccc1", 1, benzene.GetRingInfo().AtomRings(), ()),
        (naphthalene, "c1ccc2ccccc2c1", 2, naphthalene.GetRingInfo().AtomRings(), ()),
    ]
    scaffolds = [smiles for _, smiles, *_ in breakup(parents)]
    assert len(scaffolds) == 3  # benzene, naphthalene, and its benzene child


def test_breakup_fused_three_membered_ring_uses_own_fusion_bond():
    # with two fused 3-membered rings, the old code removed the bond between
    # the first two shared atoms of the whole parent (the other ring's fusion
    # bond) instead of the fusion bond of the ring being removed
    rdmol = Chem.MolFromSmiles("c1ccc2c(c1)C1NC13OC23")  # aziridine + epoxide
    scaffolds = {smiles for _, smiles, *_ in breakup(Chem.Mol(rdmol))}
    # removing the aziridine turns its own fusion bond into a double bond
    assert "c1ccc(C2CN2)cc1" in scaffolds
    # wrong children from converting the epoxide's fusion bond instead
    assert "C1=C(c2ccccc2)N1" not in scaffolds
    assert "C1=CCC=C1" not in scaffolds
    assert "c1ccc2c(c1)CC1=C2N1" not in scaffolds


def test_connected_components():
    rdmol = Chem.MolFromSmiles("CCC")
    assert connected_components(rdmol, [(0, 1)]) == [[0], [1, 2]]
    assert connected_components(rdmol, []) == [[0, 1, 2]]


@pytest.mark.parametrize(
    "smiles",
    [
        "CCC",
        "CC(=O)Nc1ccccc1",
        "CC(=O)CCC",
        "c1ccccc1CCc1ccccc1",
        "CCOc1ccccc1CCN",
        "CN1C(=O)N(C)c2ncn(C)c2C1=O",
    ],
)
@pytest.mark.parametrize("frag_func", [rigid_fragment_indices, BRICS_fragment_indices])
def test_fragment_indices_partition(smiles, frag_func):
    # every heavy atom must appear in exactly one fragment; the old DFS
    # dropped the atoms at one end of each cut bond and returned empty groups
    rdmol = Chem.MolFromSmiles(smiles)
    heavy = sorted(atom.GetIdx() for atom in rdmol.GetAtoms() if atom.GetAtomicNum() != 1)
    indices = frag_func(rdmol)
    flat = [i for group in indices for i in group]
    assert sorted(flat) == heavy
    assert len(flat) == len(set(flat))
    assert all(len(group) > 0 for group in indices)


def test_rigid_fragment_indices_covers_cut_bond_atoms():
    # the amide N is at both ends of cut bonds and used to be missing from
    # every fragment
    rdmol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
    assert rigid_fragment_indices(rdmol) == [[3], [4, 5, 6, 7, 8, 9], [0, 1, 2]]


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",  # acyclic
        "c1ccccc1",  # single ring system
        "CC(=O)Nc1ccccc1",
        "c1ccccc1CCc1ccccc1",  # ring systems connected through a linker
        "COc1ccc2c(c1)[nH]c(CC(C)C)c2C(=O)O",
    ],
)
def test_remove_exocyclic_matches_murcko(smiles):
    # used to raise ValueError: empty bond indices for acyclic molecules and
    # for ring systems connected only through linkers
    rdmol = Chem.MolFromSmiles(smiles)
    expected = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(rdmol))
    assert Chem.MolToSmiles(remove_exocyclic(rdmol)) == expected
