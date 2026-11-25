"""Torsion energy scans (potential energy profiles).

Torsion scan functions take an `rdworks.Conf` and store potential energy profiles
in `conf.props['torsion']`. `Conf` methods of the same names delegate here.
"""

import itertools
import numpy as np

from collections.abc import Callable
from typing import TYPE_CHECKING
from rdkit import Chem
from rdworks.std import clean_2d
from rdworks.torsion.fragment import create_torsion_fragment

if TYPE_CHECKING:
    # only for type hints; a runtime import would be circular (rdworks.conf imports rdworks.torsion)
    from rdworks.conf import Conf


def torsion_energy_profile(
    conf: "Conf",
    calculator: str | Callable,
    indices: tuple[int, ...],
    simplify: bool,
    interval: float,
    fmax: float,
    water: str | None,
    use_converged_only: bool,
    ) -> dict[str, list]:
    """Calculate potential energy profile for one torsion angle.

    Args:
        conf (Conf): reference conformer.
        calculator (str | Callable): 'MMFF', 'UFF', 'xTB' or ASE calculator. It will be passed to `conf.optimize()`.
        indices (tuple[int, ...]): atom indices (i,j,k,l) for a torsion angle
        simplify (bool, optional): whether to use fragmentation.
        interval (float, optional): angle intervals.
        fmax (float, optional): convergence limit for optimize. It will be passed to `conf.optimize()`.
        water (str, optional): water solvation model (choose 'gbsa' or 'alpb'). It will be passed to `conf.optimize()`.
        use_converged_only (bool, optional): whether to use only converged data.

    Args for xTB calculator:
        water (str, optional): water solvation model (choose 'gbsa' or 'alpb')
            alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
            gbsa: generalized Born (GB) model with Surface Area contributions.
            Defaults to None.

    Returns:
        dict[str, list]: A dictionary containing the torsion energy profile with `indices`, `angle`,
            and `E_rel(kcal/mol)` (plus `frag` and `frag_indices` if `simplify`).
            E_rel = E_init - median(E_last), shifted so that its minimum is 0.
    """

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

    data = {"angle": [], "init": [], "last": [], "Converged": []}

    for angle in np.arange(-180.0, 180.0, interval):
        # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
        scan_conf = work_conf.copy()
        # set torsion angle
        scan_conf.set_torsion_angle(*work_indices, angle)  # atoms bonded to `l` move.
        data["angle"].append(angle)
        # relax the structure starting from the one with the set torsion angle
        scan_conf = scan_conf.optimize(calculator, fmax=fmax, water=water)
        # conf.optimize() updates coordinates and conf.props:
        #   `E_tot_init(kcal/mol)`, `E_tot(kcal/mol)`, `Converged`.
        data["init"].append(scan_conf.props["E_tot_init(kcal/mol)"])
        data["last"].append(scan_conf.props["E_tot(kcal/mol)"])
        data["Converged"].append(scan_conf.props["Converged"])
        
    # Post-processing
    if use_converged_only:
        data["angle"] = list(itertools.compress(data["angle"], data["Converged"]))
        data["init"] = list(itertools.compress(data["init"], data["Converged"]))
        data["last"] = list(itertools.compress(data["last"], data["Converged"]))

    if len(data["init"]) == 0:
        raise RuntimeError(f"torsion scan of {indices} has no (converged) energy data")

    # calculate the relaxation energy (energy difference before and after optimization)
    relax = np.array(data["init"]) - np.median(data["last"])
    E_rel = relax - np.min(relax)

    profile_dict['angle'] = np.round(np.array(data["angle"]), 1).tolist() 
    profile_dict['E_rel(kcal/mol)'] = np.round(E_rel, 2).tolist()

    return profile_dict



def torsion_scan(
    conf: "Conf",
    calculator: str | Callable,
    torsion_angle_idx: int | None = None,
    simplify: bool = True,
    interval: float = 20.0,
    fmax: float = 0.05,
    water: str | None = None,
    use_converged_only: bool = True,
    ) -> "Conf":
    """Calculates potential energy profiles for each torsion angle using ASE or BatchOptimizer.

    Args:
        conf (Conf): reference conformer.
        calculator (str | Callable): 'MMFF', 'UFF', 'xTB', ASE calculator. It will be passed to `conf.optimize()`.
        torsion_angle_idx (int | None): key to the torsion indices to calculate. Defaults to None (all).
        simplify (bool, optional): whether to use fragment surrogate. Defaults to True.
        interval (float, optional): interval of torsion angles in degree. Defaults to 20.0.
        fmax (float, optional): fmax of ASE optimizer. It will be passed to `conf.optimize()`. Defaults to 0.05.
        water (str, optional): water solvation model (choose 'gbsa' or 'alpb'). It will be passed to `conf.optimize()`. Defaults to None.
        use_converged_only (bool, optional): whether to use only converged data. Defaults to True.

    Args for xTB calculator:
        water (str, optional): water solvation model (choose 'gbsa' or 'alpb')
            alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
            gbsa: generalized Born (GB) model with Surface Area contributions.
            Defaults to None.

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
        profile_dict = torsion_energy_profile(
            conf=conf,
            calculator=calculator,
            indices=ijkl,
            simplify=simplify,
            fmax=fmax,
            interval=interval,
            use_converged_only=use_converged_only,
            water=water,
        )
        profile_dict["torsion_angle_idx"] = torsion_angle_idx
        torsion_props.append(profile_dict)
    
    conf.props["torsion"] = torsion_props

    return conf