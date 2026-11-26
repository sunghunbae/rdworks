"""Tests for GAFF/GAFF2/OpenFF energies and optimizations (rdworks.torsion.openforcefield)."""

import shutil

import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("openff.toolkit")
pytest.importorskip("openff.nagl")

from rdworks import Mol  # noqa: E402
from rdworks.torsion import openforcefield as off  # noqa: E402
from rdworks.torsion import torsion_drive, torsion_scan  # noqa: E402

OPENFF = ["Parsley", "Sage", "Rosemary"]


def _has_gaff() -> bool:
    try:
        import openmmforcefields  # noqa: F401
    except ImportError:
        return False
    return shutil.which("antechamber") is not None


FORCEFIELDS = OPENFF + [
    pytest.param(name, marks=pytest.mark.skipif(not _has_gaff(), reason="needs openmmforcefields + AmberTools"))
    for name in ("GAFF", "GAFF2")
]


@pytest.fixture(scope="module")
def biphenyl():
    conf = Mol(molecule="c1ccccc1-c1ccccc1", name="biphenyl").make_confs(n=1).confs[0]
    return conf, conf.get_torsion_angle_atoms()[0]


def test_resolve_forcefield():
    assert off.resolve_forcefield("GAFF") == "gaff-1.81"
    assert off.resolve_forcefield("gaff2") == "gaff-2.11"
    assert off.resolve_forcefield("Parsley") == "openff_unconstrained-1.3.1.offxml"
    assert off.resolve_forcefield("SAGE") == "openff_unconstrained-2.3.0.offxml"
    assert off.resolve_forcefield("Rosemary") == "openff_unconstrained-3.0.0-rc1.offxml"
    assert off.resolve_forcefield("openff_unconstrained-2.2.1.offxml") == "openff_unconstrained-2.2.1.offxml"
    with pytest.raises(ValueError, match="Unsupported force field"):
        off.resolve_forcefield("MMFF94")
    assert off.is_supported("sage") and off.is_supported("gaff-2.11")
    assert not off.is_supported("MMFF94") and not off.is_supported(None)


@pytest.mark.parametrize("forcefield", FORCEFIELDS)
def test_singlepoint_and_optimize(biphenyl, forcefield):
    conf, _ = biphenyl
    sp = off.singlepoint(conf.copy(), forcefield)
    E_sp = sp.props["E_tot(kcal/mol)"]
    assert isinstance(E_sp, float)

    work = conf.copy()
    out = off.optimize(work, forcefield)
    assert out is work  # optimized in place, like Conf.optimize()
    assert out.props["Converged"]
    assert out.props["E_tot_init(kcal/mol)"] == pytest.approx(E_sp)
    assert out.props["E_tot(kcal/mol)"] < out.props["E_tot_init(kcal/mol)"]
    assert not np.allclose(out.positions, conf.positions)
    # the reported energy belongs to the optimized coordinates
    E_check = off.singlepoint(out.copy(), forcefield).props["E_tot(kcal/mol)"]
    assert E_check == pytest.approx(out.props["E_tot(kcal/mol)"], abs=1e-6)


@pytest.mark.parametrize("forcefield", FORCEFIELDS)
def test_constrained_optimize(biphenyl, forcefield):
    conf, ijkl = biphenyl
    before = conf.positions.copy()
    for angle in (-150.0, 0.0, 45.0, 90.0):
        opt = off.constrained_optimize(conf, forcefield, ijkl, angle)
        assert opt is not conf
        assert opt.props["Converged"]
        assert opt.get_torsion_angle(*ijkl) == pytest.approx(angle, abs=0.5)
        # reported energy excludes the torsion restraint
        E_check = off.singlepoint(opt.copy(), forcefield).props["E_tot(kcal/mol)"]
        assert E_check == pytest.approx(opt.props["E_tot(kcal/mol)"], abs=1e-6)
    np.testing.assert_allclose(conf.positions, before)  # input is not modified


def test_constrained_scan_is_symmetric(biphenyl):
    # biphenyl: E(t) = E(-t) = E(180 - t)
    conf, ijkl = biphenyl
    E = {
        a: off.constrained_optimize(conf, "Sage", ijkl, a).props["E_tot(kcal/mol)"]
        for a in (30.0, -30.0, 150.0, 0.0, 90.0)
    }
    assert E[30.0] == pytest.approx(E[-30.0], abs=0.05)
    assert E[30.0] == pytest.approx(E[150.0], abs=0.05)
    assert E[30.0] < E[0.0]
    assert E[30.0] < E[90.0]


def test_systems_are_cached(biphenyl):
    conf, ijkl = biphenyl
    off._create_system.cache_clear()
    off.singlepoint(conf.copy(), "Sage")
    off.optimize(conf.copy(), "Sage")
    off.constrained_optimize(conf, "Sage", ijkl, 60.0)
    info = off._create_system.cache_info()
    assert info.misses == 1
    assert info.hits == 2


def _symmetric(profile):
    E = dict(zip(profile["angle"], profile["E_rel(kcal/mol)"]))
    for a in E:
        for b in (-a, 180.0 - a):
            b = (b + 180.0) % 360.0 - 180.0  # wrap to [-180, 180)
            assert E[a] == pytest.approx(E[b], abs=0.05)
    return E


@pytest.mark.parametrize("forcefield", ["Sage", FORCEFIELDS[-1]])  # Sage, GAFF2
def test_torsion_drive_with_forcefield(biphenyl, forcefield):
    conf, _ = biphenyl
    work = conf.copy()
    torsion_drive(work, calculator=forcefield, simplify=False, interval=30.0)
    (profile,) = work.props["torsion"]
    assert profile["angle"] == [-180.0 + 30.0 * g for g in range(12)]
    E = _symmetric(profile)
    assert min(E.values()) == 0.0
    # twisted minima; planar (0) and perpendicular (90) barriers, whose order depends on the force field
    assert min(E[30.0], E[60.0]) == 0.0
    assert E[0.0] > 0.5 and E[90.0] > 0.5


def test_torsion_scan_with_forcefield(biphenyl):
    conf, _ = biphenyl
    work = conf.copy()
    torsion_scan(work, calculator="Sage", simplify=False, interval=30.0)
    (profile,) = work.props["torsion"]
    assert profile["torsion_angle_idx"] == 0
    assert len(profile["angle"]) == len(profile["E_rel(kcal/mol)"]) == 12
    assert min(profile["E_rel(kcal/mol)"]) == 0.0
    with pytest.raises(ValueError, match="gas phase"):
        torsion_scan(conf.copy(), calculator="Sage", water="gbsa")


def test_torsion_drive_with_forcefield_and_fragment():
    conf = Mol(molecule="CC(=O)Nc1ccc(O)cc1", name="acetaminophen").make_confs(n=1).confs[0]
    torsion_drive(conf, calculator="Sage", torsion_angle_idx=0, interval=30.0)  # simplify=True
    (profile,) = conf.props["torsion"]
    assert {"frag", "frag_indices"} <= set(profile)
    assert len(profile["angle"]) == 12
    assert min(profile["E_rel(kcal/mol)"]) == 0.0


def test_constrained_bond_forcefield_is_rejected(biphenyl):
    conf, _ = biphenyl
    with pytest.raises(ValueError, match="unconstrained"):
        off.singlepoint(conf.copy(), "openff-2.2.1.offxml")
