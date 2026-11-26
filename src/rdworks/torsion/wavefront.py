"""TorsionDrive: torsion scans driven by wavefront propagation.

Qiu, Y., Smith, D. G. A., Stern, C. D., Feng, M., Jang, H. & Wang, L.-P.
Driving torsion scans with wavefront propagation.
J. Chem. Phys. 152, 244116 (2020). https://doi.org/10.1063/5.0009232

Unlike `rdworks.torsion.sp_relaxation`, every grid point is a constrained optimization
(the torsion angle is held fixed while all other degrees of freedom relax). Each optimized
grid point seeds constrained optimizations at its neighboring grid points (the wavefront);
whenever a grid point reaches a lower energy, it joins the next wavefront. The scan ends
when no grid point improves, which removes the hysteresis of a one-directional scan.

The results are stored in `conf.props['torsion']` in the same format as `torsion_scan()`.
"""

import logging
import numpy as np
import ase

from collections.abc import Callable
from typing import TYPE_CHECKING
from ase.constraints import FixInternals
from ase.optimize import FIRE
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from rdworks.std import clean_2d
from rdworks.torsion.fragment import create_torsion_fragment
from rdworks.torsion import openforcefield
import rdworks.units

if TYPE_CHECKING:
    # only for type hints; a runtime import would be circular (rdworks.conf imports rdworks.torsion)
    from rdworks.conf import Conf

logger = logging.getLogger(__name__)


def constrained_optimize(
    conf: "Conf",
    calculator: str | Callable,
    indices: tuple[int, ...],
    angle: float,
    fmax: float = 0.05,
    max_iter: int = 1000,
    force_constant: float = 1.0e3,
    ) -> "Conf":
    """Optimize a copy of `conf` with the torsion angle `indices` held at `angle`.

    Args:
        conf (Conf): starting conformer (not modified).
        calculator (str | Callable): 'MMFF94' (= 'MMFF'), 'MMFF94s', 'UFF', or ASE calculator;
            or 'GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary', which are delegated to
            `rdworks.torsion.openforcefield.constrained_optimize()` with its own default
            restraint force constant (kcal/mol/rad^2).
        indices (tuple[int, ...]): atom indices (i,j,k,l) of the torsion angle.
        angle (float): target torsion angle in degrees.
        fmax (float, optional): force convergence limit of the ASE optimizer. Defaults to 0.05.
        max_iter (int, optional): max iterations of the optimizer. Defaults to 1000.
        force_constant (float, optional): force constant of the RDKit torsion constraint.
            Defaults to 1.0e3. Much stiffer constraints (e.g. 1.0e4) can make the RDKit
            minimizer stop early in higher-energy states.

    Raises:
        ValueError: for an unsupported calculator (including 'xTB', which has no
            constrained optimization; use an ASE xTB calculator instead) or when
            the force field cannot be set up.

    Returns:
        Conf: optimized copy with `E_tot(kcal/mol)` (unconstrained energy) and `Converged`
            in its props.
    """
    if openforcefield.is_supported(calculator):
        return openforcefield.constrained_optimize(
            conf, calculator, indices, angle, fmax=fmax, max_iter=max_iter
        )

    (i, j, k, l) = indices
    work = conf.copy()
    work.set_torsion_angle(i, j, k, l, angle)  # atoms bonded to `l` move.

    if isinstance(calculator, str):
        name = calculator.lower()
        if name in ("mmff94", "mmff", "mmff94s"):
            variant = "MMFF94s" if name == "mmff94s" else "MMFF94"
            mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(work.rdmol, mmffVariant=variant)
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(work.rdmol, mp)
            if ff is not None:
                ff.MMFFAddTorsionConstraint(i, j, k, l, False, angle, angle, force_constant)
        elif name == "uff":
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(work.rdmol)
            if ff is not None:
                ff.UFFAddTorsionConstraint(i, j, k, l, False, angle, angle, force_constant)
        else:
            raise ValueError(
                f"Unsupported calculator for constrained optimization: {calculator!r} "
                "(choose 'MMFF94', 'MMFF94s', 'UFF', or an ASE calculator)"
            )
        if ff is None:
            raise ValueError(
                f"{calculator} force field could not be set up for {work.name!r} "
                "(missing parameters); try calculator='UFF'"
            )
        # the force field holds the conformer coordinates, which Minimize() updates in place
        converged = ff.Minimize(maxIts=max_iter) == 0
        work.singlepoint(calculator)  # energy without the torsion constraint

    else:
        # assuming ASE calculator
        ase_atoms = ase.Atoms(symbols=work.symbols, positions=work.positions)
        ase_atoms.info["charge"] = work.charge
        ase_atoms.info["spin"] = work.spin
        ase_atoms.set_constraint(
            FixInternals(dihedrals_deg=[[angle % 360.0, [i, j, k, l]]])
        )
        ase_atoms.calc = calculator
        FIRE(ase_atoms, logfile=None).run(fmax=fmax, steps=max_iter)
        # get_forces() returns forces projected onto the constraint
        converged = bool(np.linalg.norm(ase_atoms.get_forces(), axis=1).max() < fmax)
        PE = float(np.ravel(ase_atoms.get_potential_energy())[0])
        work.sync(ase_atoms.get_positions())
        work.props["E_tot(kcal/mol)"] = rdworks.units.ev2kcalpermol * PE

    work.props["Converged"] = converged

    return work


def torsion_drive_profile(
    conf: "Conf",
    calculator: str | Callable,
    indices: tuple[int, ...],
    simplify: bool,
    interval: float,
    fmax: float,
    max_iter: int,
    energy_decrease_thresh: float,
    use_converged_only: bool,
    ) -> dict[str, list]:
    """Calculate potential energy profile for one torsion angle by wavefront propagation.

    Args:
        conf (Conf): reference conformer (not modified); its geometry seeds the scan.
        calculator (str | Callable): 'MMFF94' (= 'MMFF'), 'MMFF94s', 'UFF', or ASE calculator;
            or 'GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary'.
        indices (tuple[int, ...]): atom indices (i,j,k,l) for a torsion angle.
        simplify (bool): whether to use fragmentation.
        interval (float): grid spacing in degrees; 360 must be a multiple of it.
        fmax (float): convergence limit of the ASE optimizer.
        max_iter (int): max iterations of each constrained optimization.
        energy_decrease_thresh (float): a new result replaces a grid point only if it
            lowers the energy by more than this (kcal/mol).
        use_converged_only (bool): whether to discard non-converged optimizations.

    Raises:
        ValueError: if 360 is not a multiple of `interval`.
        RuntimeError: if no grid point has a (converged) energy.

    Returns:
        dict[str, list]: A dictionary containing the torsion energy profile with `indices`, `angle`,
            and `E_rel(kcal/mol)` (plus `frag` and `frag_indices` if `simplify`).
            E_rel = E - min(E), where E is the constrained-optimized energy at each grid point.
    """
    n_grid = 360.0 / interval
    if interval <= 0 or not np.isclose(n_grid, round(n_grid)):
        raise ValueError(f"360 must be a multiple of interval (got {interval})")
    n_grid = int(round(n_grid))
    grid = [-180.0 + g * interval for g in range(n_grid)]

    profile_dict = {"indices": indices}

    if simplify:
        ref_conf = conf.copy()
        (frag, frag_ijkl, frag_created, wbo_filtered) = create_torsion_fragment(
            ref_conf.rdmol, indices
        )
        frag_conf = type(conf)(frag)
        frag_cleaned, _ = clean_2d(frag, reset_isotope=True, remove_H=True)
        # to serialize the molecule
        profile_dict["frag"] = Chem.MolToMolBlock(frag_cleaned)
        profile_dict["frag_indices"] = frag_ijkl
        work_conf = frag_conf
        work_indices = frag_ijkl
    else:
        work_conf = conf
        work_indices = indices

    def _optimize(start: "Conf", g: int) -> "Conf | None":
        result = constrained_optimize(
            start, calculator, work_indices, grid[g], fmax=fmax, max_iter=max_iter
        )
        if use_converged_only and not result.props["Converged"]:
            return None
        return result

    # seed: constrained optimization at the grid point nearest to the current torsion angle
    current = work_conf.get_torsion_angle(*work_indices)
    g0 = int(round((current + 180.0) / interval)) % n_grid
    best: dict[int, "Conf"] = {}  # grid index -> lowest-energy optimized conformer
    seed = _optimize(work_conf, g0)
    if seed is not None:
        best[g0] = seed
    wavefront = set(best)

    n_wave = 0
    while wavefront:
        n_wave += 1
        # constrained optimizations from each wavefront grid point to its neighbors
        candidates: dict[int, "Conf"] = {}
        for g in sorted(wavefront):
            for nb in ((g - 1) % n_grid, (g + 1) % n_grid):
                result = _optimize(best[g], nb)
                if result is None:
                    continue
                E = result.props["E_tot(kcal/mol)"]
                if nb not in candidates or E < candidates[nb].props["E_tot(kcal/mol)"]:
                    candidates[nb] = result
        # grid points whose energy decreased form the next wavefront
        wavefront = set()
        for nb, result in candidates.items():
            E = result.props["E_tot(kcal/mol)"]
            if nb not in best or E < best[nb].props["E_tot(kcal/mol)"] - energy_decrease_thresh:
                best[nb] = result
                wavefront.add(nb)
        logger.debug(
            "torsion drive %s: wavefront %d updated %d grid points", indices, n_wave, len(wavefront)
        )

    if len(best) == 0:
        raise RuntimeError(f"torsion drive of {indices} has no (converged) energy data")

    grid_indices = sorted(best)
    E = np.array([best[g].props["E_tot(kcal/mol)"] for g in grid_indices])
    E_rel = E - np.min(E)

    profile_dict["angle"] = np.round(np.array([grid[g] for g in grid_indices]), 1).tolist()
    profile_dict["E_rel(kcal/mol)"] = np.round(E_rel, 2).tolist()

    return profile_dict


def torsion_drive(
    conf: "Conf",
    calculator: str | Callable,
    torsion_angle_idx: int | None = None,
    simplify: bool = True,
    interval: float = 15.0,
    fmax: float = 0.05,
    max_iter: int = 1000,
    energy_decrease_thresh: float = 1.0e-3,
    use_converged_only: bool = True,
    ) -> "Conf":
    """Calculates potential energy profiles for each torsion angle with TorsionDrive.

    TorsionDrive (https://doi.org/10.1063/5.0009232) scans a torsion angle on a periodic grid
    by wavefront propagation of constrained optimizations, so each grid point ends up with
    the lowest energy reachable from its neighbors.

    Args:
        conf (Conf): reference conformer.
        calculator (str | Callable): 'MMFF94' (= 'MMFF'), 'MMFF94s', 'UFF', or ASE calculator;
            or 'GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary' (see `rdworks.torsion.openforcefield`).
        torsion_angle_idx (int | None): key to the torsion indices to calculate. Defaults to None (all).
        simplify (bool, optional): whether to use fragment surrogate. Defaults to True.
        interval (float, optional): grid spacing of torsion angles in degree; 360 must be
            a multiple of it. Defaults to 15.0.
        fmax (float, optional): fmax of ASE optimizer. Defaults to 0.05.
        max_iter (int, optional): max iterations of each constrained optimization. Defaults to 1000.
        energy_decrease_thresh (float, optional): minimum energy decrease (kcal/mol) for a
            grid point to be updated and propagated. Defaults to 1.0e-3.
        use_converged_only (bool, optional): whether to use only converged data. Defaults to True.

    Returns:
        Conf: `conf` with `conf.props['torsion']` set to a list of profile dicts, one per torsion,
            each with `indices`, `angle`, `E_rel(kcal/mol)`, and `torsion_angle_idx`
            (plus `frag` and `frag_indices` if `simplify`).
    """

    torsion_angle_atom_indices_list = conf.get_torsion_angle_atoms()
    # [(5, 4, 3, 1), ...]

    if isinstance(torsion_angle_idx, int) and \
        torsion_angle_idx < len(torsion_angle_atom_indices_list):
        indices = {
            torsion_angle_idx: torsion_angle_atom_indices_list[torsion_angle_idx]
        }
        # atom indices for a single torsion angle
        # ex. {0: (5,4,3,1)}
    else:
        indices = {k: v for k, v in enumerate(torsion_angle_atom_indices_list)}
        # atom indices for all torsion angles
        # ex. {0: (5,4,3,1), 1: ...}

    torsion_props = []
    for torsion_angle_idx, ijkl in indices.items():
        profile_dict = torsion_drive_profile(
            conf=conf,
            calculator=calculator,
            indices=ijkl,
            simplify=simplify,
            interval=interval,
            fmax=fmax,
            max_iter=max_iter,
            energy_decrease_thresh=energy_decrease_thresh,
            use_converged_only=use_converged_only,
        )
        profile_dict["torsion_angle_idx"] = torsion_angle_idx
        torsion_props.append(profile_dict)

    conf.props["torsion"] = torsion_props

    return conf
