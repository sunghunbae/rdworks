"""Regression tests for bug fixes listed under `Unreleased` in CHANGELOG.md."""

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from rdworks import Mol
from rdworks.stereoisomers import can_have_stereo, complete_stereoisomers
from rdworks.tautomers import complete_tautomers


ACETAMINOPHEN = "CC(=O)Nc1ccc(O)cc1"


@pytest.fixture
def apap():
    """Acetaminophen with 6 reproducible conformers."""
    return Mol(ACETAMINOPHEN, "apap").make_confs(n=6, seed=1)


def _double_bond(smiles: str) -> Chem.Bond:
    rdmol = Chem.MolFromSmiles(smiles)
    return next(b for b in rdmol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)


##################################################
### stereoisomers / tautomers
##################################################


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("CC=CC", True),  # 2-butene: one heavy neighbor + H on each atom
        ("CCCCN=C(O)C", True),  # imine with a lone pair on N
        ("C1CC=CC1", False),  # small-ring double bond
        ("CC(C)=CC", False),  # identical substituents
    ],
)
def test_can_have_stereo(smiles, expected):
    assert can_have_stereo(_double_bond(smiles)) is expected


def test_complete_stereoisomers_does_not_modify_input():
    m = Mol("C[C@H](N)C(=O)O", "ala")
    before = (m.name, m.smiles)
    libr = complete_stereoisomers(m, name="renamed", override=True)
    assert (m.name, m.smiles) == before
    assert [x.name for x in libr] == ["renamed.1", "renamed.2"]


def test_complete_tautomers_single_result_keeps_name():
    libr = complete_tautomers(Mol("c1ccccc1", "benzene"))
    assert libr.count() == 1
    assert libr[0].name == "benzene"


def test_complete_backward_compatible_aliases():
    import rdworks.complete as c
    import rdworks.stereoisomers as s
    import rdworks.tautomers as t

    assert c.complete_stereoisomers is s.complete_stereoisomers
    assert c.can_have_stereo is s.can_have_stereo
    assert c.complete_tautomers is t.complete_tautomers


##################################################
### Mol.make_confs
##################################################


def test_make_confs_verbose_and_calculator():
    assert Mol(ACETAMINOPHEN).make_confs(n=3, verbose=True).num_confs > 0
    assert Mol(ACETAMINOPHEN).make_confs(n=3, calculator="UFF").num_confs > 0


@pytest.mark.parametrize("kwargs", [{"method": "CONFORGE"}, {"method": "ETDKG"}, {"n": 0}])
def test_make_confs_invalid_input(kwargs):
    with pytest.raises(ValueError):
        Mol(ACETAMINOPHEN).make_confs(**kwargs)


@pytest.mark.parametrize("smiles", ["OB(O)c1ccccc1", "C[Se]C"])
def test_make_confs_falls_back_to_uff(smiles):
    m = Mol(smiles).make_confs(n=3, seed=1)
    assert m.num_confs > 0
    assert all(c.props["E_tot(kcal/mol)"] is not None for c in m.confs)


def test_make_confs_seed_is_reproducible():
    def coords(seed):
        m = Mol(ACETAMINOPHEN).make_confs(n=3, seed=seed)
        return [c.positions.round(4).tolist() for c in m.confs]

    assert coords(42) == coords(42)
    assert coords(42) != coords(7)


def test_make_confs_prune_rms():
    unpruned = Mol("CCCCCCCC").make_confs(n=30, seed=1).num_confs
    pruned = Mol("CCCCCCCC").make_confs(n=30, seed=1, prune_rms=1.0).num_confs
    assert pruned < unpruned


def test_make_confs_keeps_conformers_on_failure(apap):
    before = [c.name for c in apap.confs]
    with pytest.raises(ValueError):
        apap.make_confs(n=2, calculator="bogus")
    assert [c.name for c in apap.confs] == before


def test_make_confs_sorted_and_named(apap):
    e_rel = [c.props["E_rel(kcal/mol)"] for c in apap.confs]
    assert e_rel == sorted(e_rel)
    assert e_rel[0] == 0.0
    assert apap.confs[0].name == "apap/1"


def test_etkdg_params_are_independent():
    a = Mol.etkdg_params(seed=1)
    b = Mol.etkdg_params()
    assert a.randomSeed == 1
    assert b.randomSeed == -1


##################################################
### Conf.singlepoint
##################################################


def test_conf_singlepoint_missing_mmff_params():
    m = Mol("OB(O)c1ccccc1", "boronic").make_confs(n=1, seed=1)
    with pytest.raises(ValueError, match="missing parameters"):
        m.confs[0].singlepoint("MMFF94")


##################################################
### Mol dunder methods and __init__
##################################################


def test_hash_consistent_with_eq():
    # InChIKey treats these two tautomers as the same molecule
    a, b = Mol("Oc1ccccn1", "a"), Mol("O=c1cccc[nH]1", "b")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_eq_with_non_mol():
    assert (Mol("CCO") == "CCO") is False


def test_iteration_without_next(apap):
    assert not hasattr(Mol, "__next__")
    assert len([c for c in apap]) == apap.num_confs


def test_name_none():
    assert Mol("CCO", None).name == ""


##################################################
### Mol conformer pipeline
##################################################


def test_sort_confs_verbose_with_calculator(apap):
    assert apap.sort_confs("MMFF94", verbose=True).num_confs == 6


def test_align_confs_invalid_method(apap):
    with pytest.raises(ValueError):
        apap.align_confs(method="bogus")


@pytest.mark.parametrize("method", ["rigid_fragment", "CrippenO3A", "MMFFO3A", "best_rms"])
def test_align_confs_methods(apap, method):
    assert apap.align_confs(method=method).is_confs_aligned


def test_drop_confs_cluster_keeps_centroids(apap):
    apap = apap.cluster_confs(threshold=0.5)
    num_centroids = sum(c.props["cluster_centroid"] for c in apap.confs)
    assert num_centroids < apap.num_confs  # precondition: something to drop
    assert apap.drop_confs(cluster=True).num_confs == num_centroids


def test_cluster_confs_dqt_wraps_torsion_difference():
    # two biphenyl conformers at +179 and -179 degrees differ by only 2 degrees
    m = Mol("c1ccccc1-c1ccccc1", "biphenyl").make_confs(n=1, seed=1)
    torsion = m.get_torsion_angle_atoms()[0]
    other = m.confs[0].copy()
    rdMolTransforms.SetDihedralDeg(m.confs[0].rdmol.GetConformer(), *torsion, 179.0)
    rdMolTransforms.SetDihedralDeg(other.rdmol.GetConformer(), *torsion, -179.0)
    m.confs.append(other)
    for c in m.confs:
        c.props["E_rel(kcal/mol)"] = 0.0
    m.is_confs_aligned = True
    m.cluster_confs(method="DQT", threshold=0.1)  # 0.1 rad ~ 5.7 deg
    assert m.confs[0].props["cluster"] == m.confs[1].props["cluster"]


##################################################
### Mol endpoint functions
##################################################


def test_draw_invalid_axis():
    with pytest.raises(ValueError):
        Mol("CCO").draw(rotate=True, axis="w", degree=30)
    Mol("CCO").draw(rotate=True, axis="z", degree=30)  # valid axis still works


def test_has_substr_invalid_smarts():
    with pytest.raises(ValueError):
        Mol("CCO").has_substr("C((")
    assert Mol(ACETAMINOPHEN).has_substr("OC(=O)C") is False
    assert Mol("CC(=O)O").has_substr("OC(=O)C") is True


def test_is_matching_zero_bound(tmp_path):
    # ethanol has HBD = 1, so it must fail a max of 0 (0.0 was previously ignored)
    xml = tmp_path / "no_hbd.xml"
    xml.write_text(
        '<filter combine="AND">'
        '<descriptor name="HBD"><max>0</max></descriptor>'
        "</filter>"
    )
    assert Mol("CCO").is_matching(xml) is False
    assert Mol("CCOC").is_matching(xml) is True
    # a leading '~' inverts the result
    assert Mol("CCO").is_matching("~" + str(xml)) is True

# std.py regression tests are in tests/test_std.py
