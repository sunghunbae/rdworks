"""Tests for TorsionDrive (rdworks.torsion.wavefront) with the MMFF94 calculator."""

import numpy as np
import pytest

from rdworks import Mol
from rdworks.torsion import (
    constrained_optimize,
    get_torsion_plot_data,
    torsion_drive,
    torsion_drive_profile,
)


def _biphenyl_conf():
    conf = Mol(molecule="c1ccccc1-c1ccccc1", name="biphenyl").make_confs(n=1).confs[0]
    ta = conf.get_torsion_angle_atoms()
    assert len(ta) == 1
    return conf, ta[0]


def _drive(conf, ijkl, start, interval=30.0):
    seed = conf.copy().set_torsion_angle(*ijkl, start)
    return torsion_drive_profile(
        seed,
        calculator="MMFF94",
        indices=ijkl,
        simplify=False,
        interval=interval,
        fmax=0.05,
        max_iter=1000,
        energy_decrease_thresh=1.0e-3,
        use_converged_only=True,
    )


def test_constrained_optimize_holds_torsion_angle():
    conf, ijkl = _biphenyl_conf()
    before = conf.get_torsion_angle(*ijkl)
    for angle in (-150.0, 0.0, 45.0, 90.0):
        opt = constrained_optimize(conf, "MMFF94", ijkl, angle)
        assert opt.props["Converged"]
        assert opt.get_torsion_angle(*ijkl) == pytest.approx(angle, abs=0.5)
        assert isinstance(opt.props["E_tot(kcal/mol)"], float)
    # input is not modified
    assert conf.get_torsion_angle(*ijkl) == pytest.approx(before)


def test_torsion_drive_biphenyl_profile():
    conf, ijkl = _biphenyl_conf()
    profile = _drive(conf, ijkl, start=0.0)
    angles = profile["angle"]
    E = dict(zip(angles, profile["E_rel(kcal/mol)"]))
    # every grid point is reached by wavefront propagation
    assert angles == [-180.0 + 30.0 * g for g in range(12)]
    assert min(E.values()) == 0.0
    # planar biphenyl is the global maximum; twisted forms are lower
    assert E[0.0] == max(E.values())
    assert E[0.0] > 2.0
    assert E[60.0] < E[90.0] < E[0.0]
    # symmetry of biphenyl: E(t) = E(-t) = E(180 - t)
    for a in angles:
        for b in (-a, 180.0 - a):
            b = (b + 180.0) % 360.0 - 180.0  # wrap to [-180, 180)
            assert E[a] == pytest.approx(E[b], abs=0.05)


def test_torsion_drive_is_independent_of_starting_angle():
    # wavefront propagation removes the hysteresis of a one-directional scan
    conf, ijkl = _biphenyl_conf()
    profiles = [_drive(conf, ijkl, start)["E_rel(kcal/mol)"] for start in (0.0, 45.0, 90.0, -140.0)]
    spread = np.max(profiles, axis=0) - np.min(profiles, axis=0)
    assert spread.max() <= 0.05


def test_torsion_drive_conf_props():
    mol = Mol(molecule="CC(=O)Nc1ccc(O)cc1", name="acetaminophen.3").make_confs(n=1)
    conf = mol.confs[0]
    ta = conf.get_torsion_angle_atoms()
    assert len(ta) == 1
    out = torsion_drive(conf, calculator="MMFF94", torsion_angle_idx=0, interval=30.0)
    assert out is conf
    profiles = conf.props["torsion"]
    assert len(profiles) == 1
    p = profiles[0]
    assert p["torsion_angle_idx"] == 0
    assert p["indices"] == ta[0]
    assert {"frag", "frag_indices", "angle", "E_rel(kcal/mol)"} <= set(p)
    assert len(p["angle"]) == len(p["E_rel(kcal/mol)"]) == 12
    assert min(p["E_rel(kcal/mol)"]) == 0.0
    data = get_torsion_plot_data(profiles, idx=0)
    assert data["angle"] == p["angle"]


def test_torsion_drive_invalid_inputs():
    conf, ijkl = _biphenyl_conf()
    with pytest.raises(ValueError, match="multiple of interval"):
        _drive(conf, ijkl, start=0.0, interval=25.0)
    with pytest.raises(ValueError, match="Unsupported calculator"):
        constrained_optimize(conf, "xTB", ijkl, 0.0)
