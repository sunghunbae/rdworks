"""Torsion energy scans (potential energy profiles).

Torsion scan functions take an `rdworks.Conf` and store potential energy profiles
in `conf.props['torsion']`. `Conf` methods of the same names delegate here.
"""

import itertools
import numpy as np

from collections.abc import Callable
from typing import TYPE_CHECKING
from rdkit import Chem
from rdworks.batch import prepare_batches
from rdworks.std import clean_2d
from rdworks.torsion.fragment import create_torsion_fragment

if TYPE_CHECKING:
    # only for type hints; a runtime import would be circular (rdworks.conf imports rdworks.torsion)
    from rdworks.conf import Conf


##################################################
### Torsion scan (potential energy profiles)
##################################################
#
# These functions take an `rdworks.Conf` object. `rdworks.Conf` is not imported at module
# level because `rdworks.conf` imports this module (circular import). New Conf objects are
# created with `type(conf)(...)`.


def calculate_torsion_energies_one(
    conf: "Conf",
    calculator: str | Callable,
    indices: tuple[int, ...],
    simplify: bool = True,
    fmax: float = 0.05,
    interval: float = 20.0,
    use_converged_only: bool = True,
    water: str | None = None,
) -> "Conf":
    """Calculate potential energy profile for one torsion angle.

    Args:
        conf (Conf): reference conformer.
        calculator (str | Callable): 'MMFF', 'UFF', 'xTB' or ASE calculator.
        indices (tuple[int, ...]): atom indices (i,j,k,l) for a torsion angle
        simplify (bool, optional): whether to use fragmentation. Defaults to True.
        fmax (float, optional): convergence limit for optimize. Defaults to 0.05.
        interval (float, optional): angle intervals. Defaults to 20.0.
        use_converged_only (bool, optional): whether to use only converged data. Defaults to True.

    Args for xTB calculator:
        water (str, optional): water solvation model (choose 'gbsa' or 'alpb')
            alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
            gbsa: generalized Born (GB) model with Surface Area contributions.
            Defaults to None.

    Returns:
        Conf: `conf` with `conf.props['torsion']` set to a profile dict with `indices`, `angle`,
            and `E_rel(kcal/mol)` (plus `frag` and `frag_indices` if `simplify`).
            E_rel = E_init - median(E_last), shifted so that its minimum is 0.
    """
    ref_conf = conf.copy()

    data = {
        "indices": indices,
        "angle": [],
        "init": [],
        "last": [],
        "Converged": [],
    }

    if simplify:
        (frag, frag_ijkl, frag_created, wbo_filtered) = create_torsion_fragment(
            ref_conf.rdmol, indices
        )
        frag_conf = type(conf)(frag)
        frag_cleaned, _ = clean_2d(frag, reset_isotope=True, remove_H=True)
        # to serialize the molecule
        data["frag"] = Chem.MolToMolBlock(frag_cleaned)
        data["frag_indices"] = frag_ijkl
        for angle in np.arange(-180.0, 180.0, interval):
            # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
            scan_conf = frag_conf.copy()
            scan_conf.set_torsion_angle(*frag_ijkl, angle)  # atoms bonded to `l` move.
            scan_conf = scan_conf.optimize(calculator, fmax=fmax, water=water)
            data["angle"].append(angle)
            # conf.optimize() updates coordinates and conf.props:
            #   `E_tot_init(kcal/mol)`, `E_tot(kcal/mol)`, `Converged`.
            data["init"].append(scan_conf.props["E_tot_init(kcal/mol)"])
            data["last"].append(scan_conf.props["E_tot(kcal/mol)"])
            data["Converged"].append(scan_conf.props["Converged"])
    else:
        for angle in np.arange(-180.0, 180.0, interval):
            # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
            scan_conf = ref_conf.copy()
            scan_conf.set_torsion_angle(*indices, angle)  # atoms bonded to `l` move.
            scan_conf = scan_conf.optimize(calculator, fmax=fmax, water=water)
            data["angle"].append(angle)
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

    relax = np.array(data["init"]) - np.median(data["last"])
    E_rel = relax - np.min(relax)

    torsion_energy_profile = {
        "indices": data["indices"],
        "angle": np.round(
            np.array(data["angle"]), 1
        ).tolist(),  # np.ndarray -> list for serialization
        "E_rel(kcal/mol)": np.round(
            E_rel, 2
        ).tolist(),  # np.ndarray -> list for serialization
    }

    if simplify:
        torsion_energy_profile.update(
            {
                "frag": data.get("frag", None),
                "frag_indices": data.get("frag_indices", None),
            }
        )

    conf.props["torsion"] = torsion_energy_profile

    return conf


def calculate_torsion_energies_batch(
    conf: "Conf",
    batchoptimizer: Callable,
    indices: tuple,
    simplify: bool = True,
    fmax: float = 0.05,
    interval: float = 20.0,
    use_converged_only: bool = True,
    batchsize_atoms: int = 16 * 1024,
) -> "Conf":
    """Calculate potential energy profile for torsion angles in a batch.

    Args:
        conf (Conf): reference conformer.
        batchoptimizer (Callable): BatchOptimizer.
        indices (tuple): atom indices (i,j,k,l) for a torsion angle
        simplify (bool, optional): whether to use fragmentation. Defaults to True.
        fmax (float, optional): unused; set convergence on the batch optimizer. Defaults to 0.05.
        interval (float, optional): angle intervals. Defaults to 20.0.
        use_converged_only (bool, optional): whether to use only converged data. Defaults to True.
        batchsize_atoms (int, optional): maximum number of atoms in a single batch.
            Defaults to 16384 (16 * 1024).

    Returns:
        Conf: `conf` with `conf.props['torsion']` set to a profile dict with `indices`, `angle`,
            and `E_rel(kcal/mol)` (plus `frag` and `frag_indices` if `simplify`).
            E_rel = E_init - median(E_last), shifted so that its minimum is 0.
    """
    ref_conf = conf.copy()

    data = {
        "indices": indices,
        "angle": [],
        "init": [],
        "last": [],
        "Converged": [],
    }

    confs = []
    if simplify:
        (frag, frag_ijkl, frag_created, wbo_filtered) = create_torsion_fragment(
            ref_conf.rdmol, indices
        )
        frag_conf = type(conf)(frag)
        frag_cleaned, _ = clean_2d(frag, reset_isotope=True, remove_H=True)
        # to serialize the molecule
        data["frag"] = Chem.MolToMolBlock(frag_cleaned)
        data["frag_indices"] = frag_ijkl
        for angle in np.arange(-180.0, 180.0, interval):
            # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
            scan_conf = frag_conf.copy()
            scan_conf.set_torsion_angle(*frag_ijkl, angle)  # atoms bonded to `l` move.
            data["angle"].append(angle)
            confs.append(scan_conf)

    else:
        for angle in np.arange(-180.0, 180.0, interval):
            # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
            scan_conf = ref_conf.copy()
            scan_conf.set_torsion_angle(*indices, angle)  # atoms bonded to `l` move.
            data["angle"].append(angle)
            confs.append(scan_conf)

    batches = prepare_batches(confs, batchsize_atoms)

    for batch in batches:
        optimized = batchoptimizer(batch.rdmols).run()
        for rdmol in optimized.mols:
            # conf.optimize() updates coordinates and conf.props:
            #   `E_tot_init(kcal/mol)`, `E_tot(kcal/mol)`, `Converged`.
            data["init"].append(float(rdmol.GetProp("E_tot_init(kcal/mol)")))
            data["last"].append(float(rdmol.GetProp("E_tot(kcal/mol)")))
            data["Converged"].append(
                True if rdmol.GetProp("Converged") == "True" else False
            )

    # Post-processing
    if use_converged_only:
        data["angle"] = list(itertools.compress(data["angle"], data["Converged"]))
        data["init"] = list(itertools.compress(data["init"], data["Converged"]))
        data["last"] = list(itertools.compress(data["last"], data["Converged"]))

    if len(data["init"]) == 0:
        raise RuntimeError(f"torsion scan of {indices} has no (converged) energy data")

    relax = np.array(data["init"]) - np.median(data["last"])
    E_rel = relax - np.min(relax)

    torsion_energy_profile = {
        "indices": data["indices"],
        "angle": np.round(
            np.array(data["angle"]), 1
        ).tolist(),  # np.ndarray -> list for serialization
        "E_rel(kcal/mol)": np.round(
            E_rel, 2
        ).tolist(),  # np.ndarray -> list for serialization
    }

    if simplify:
        torsion_energy_profile.update(
            {
                "frag": data.get("frag", None),
                "frag_indices": data.get("frag_indices", None),
            }
        )

    conf.props["torsion"] = torsion_energy_profile

    return conf


def calculate_torsion_energies(
    conf: "Conf",
    calculator: str | Callable,
    torsion_angle_idx: int | None = None,
    simplify: bool = True,
    fmax: float = 0.05,
    interval: float = 20.0,
    use_converged_only: bool = True,
    batchsize_atoms: int = 0,
    water: str | None = None,
) -> "Conf":
    """Calculates potential energy profiles for each torsion angle using ASE or BatchOptimizer.

    Args:
        conf (Conf): reference conformer.
        calculator (str | Callable): 'MMFF', 'UFF', 'xTB', ASE calculator, or BatchOptimizer.
        torsion_angle_idx (int | None): key to the torsion indices to calculate. Defaults to None (all).
        simplify (bool, optional): whether to use fragment surrogate. Defaults to True.
        fmax (float, optional): fmax of ASE optimizer. Defaults to 0.05.
        interval (float, optional): interval of torsion angles in degree. Defaults to 20.0.
        use_converged_only (bool, optional): whether to use only converged data. Defaults to True.
        batchsize_atoms (int, optional): maximum number of atoms in a single batch.
            Setting any number smaller than conf.natoms to disable batch optimization.
            Defaults to 0.

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

    if isinstance(torsion_angle_idx, int) and torsion_angle_idx < len(
        torsion_angle_atom_indices_list
    ):
        indices = {
            torsion_angle_idx: torsion_angle_atom_indices_list[torsion_angle_idx]
        }
        # atom indices for a single torsion angle
        # ex. {0: (5,4,3,1)}
    else:
        indices = {k: v for k, v in enumerate(torsion_angle_atom_indices_list)}
        # atom indices for all torsion angles
        # ex. {0: (5,4,3,1), 1: ...}

    work_conf = conf.copy()
    torsion_energies = []
    if batchsize_atoms >= work_conf.natoms:
        for torsion_angle_idx, ijkl in indices.items():
            work_conf = calculate_torsion_energies_batch(
                work_conf,
                calculator,
                ijkl,
                simplify,
                fmax,
                interval,
                use_converged_only,
                batchsize_atoms=batchsize_atoms,
            )
            work_conf.props["torsion"].update({"torsion_angle_idx": torsion_angle_idx})
            torsion_energies.append(work_conf.props["torsion"])
    else:
        for torsion_angle_idx, ijkl in indices.items():
            work_conf = calculate_torsion_energies_one(
                work_conf,
                calculator,
                ijkl,
                simplify,
                fmax,
                interval,
                use_converged_only,
                water=water,
            )
            work_conf.props["torsion"].update({"torsion_angle_idx": torsion_angle_idx})
            torsion_energies.append(work_conf.props["torsion"])

    conf.props["torsion"] = torsion_energies

    return conf


def sp_torsion_energies(
    conf: "Conf",
    calculator: str | Callable,
    indices: tuple[int, ...],
    simplify: bool = True,
    interval: float = 20.0,
    water: str | None = None,
) -> "Conf":
    """Single-Point based potential energy profile for one torsion angle.

    Args:
        conf (Conf): reference conformer.
        calculator (str | Callable): 'MMFF', 'UFF', 'xTB' or ASE calculator.
        indices (tuple[int, ...]): atom indices (i,j,k,l) for a torsion angle
        simplify (bool, optional): whether to use fragmentation. Defaults to True.
        interval (float, optional): angle intervals. Defaults to 20.0.

    Args for xTB calculator:
        water (str, optional): water solvation model (choose 'gbsa', 'alpb', 'cpcmx')
            alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
            gbsa: generalized Born (GB) model with Surface Area contributions.
            cpcmx: Extended Conductor-like Polarizable Continuum Solvation Model (CPCM-X).
            Defaults to None.

    Returns:
        Conf: `conf` with `conf.props['torsion']` set to a profile dict with `indices`, `angle`,
            and `E_rel(kcal/mol)` (plus `frag` and `frag_indices` if `simplify`).
            E_rel = E_sp - E_ref (single-point energy relative to the reference conformer),
            shifted so that its minimum is 0.
    """
    ref_conf = conf.copy()  # optimized reference conformer
    ref_conf = ref_conf.singlepoint(calculator, water=water)

    data = {
        "indices": indices,
        "angle": [],
        "init": [],
        "last": [],
        "Converged": [],
    }

    if simplify:
        (frag, frag_ijkl, frag_created, wbo_filtered) = create_torsion_fragment(
            ref_conf.rdmol, indices
        )
        frag_conf = type(conf)(frag)
        frag_cleaned, _ = clean_2d(frag, reset_isotope=True, remove_H=True)
        # to serialize the molecule
        data["frag"] = Chem.MolToMolBlock(frag_cleaned)
        data["frag_indices"] = frag_ijkl
        for angle in np.arange(-180.0, 180.0, interval):
            # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
            scan_conf = frag_conf.copy()
            scan_conf.set_torsion_angle(*frag_ijkl, angle)  # atoms bonded to `l` move.
            scan_conf = scan_conf.singlepoint(calculator, water=water)
            data["angle"].append(angle)
            # conf.singlepoint() sets `E_tot(kcal/mol)`
            # conf.optimize() updates coordinates and conf.props:
            #   `E_tot_init(kcal/mol)`, `E_tot(kcal/mol)`, `Converged`.
            data["init"].append(scan_conf.props["E_tot(kcal/mol)"])
            data["last"].append(ref_conf.props["E_tot(kcal/mol)"])
    else:
        for angle in np.arange(-180.0, 180.0, interval):
            # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
            scan_conf = ref_conf.copy()
            scan_conf.set_torsion_angle(*indices, angle)  # atoms bonded to `l` move.
            scan_conf = scan_conf.singlepoint(calculator, water=water)
            data["angle"].append(angle)
            # conf.optimize() updates coordinates and conf.props:
            #   `E_tot_init(kcal/mol)`, `E_tot(kcal/mol)`, `Converged`.
            data["init"].append(scan_conf.props["E_tot(kcal/mol)"])
            data["last"].append(ref_conf.props["E_tot(kcal/mol)"])

    if len(data["init"]) == 0:
        raise RuntimeError(f"torsion scan of {indices} has no (converged) energy data")

    relax = np.array(data["init"]) - np.median(data["last"])
    E_rel = relax - np.min(relax)

    torsion_energy_profile = {
        "indices": data["indices"],
        "angle": np.round(
            np.array(data["angle"]), 1
        ).tolist(),  # np.ndarray -> list for serialization
        "E_rel(kcal/mol)": np.round(
            E_rel, 2
        ).tolist(),  # np.ndarray -> list for serialization
    }

    if simplify:
        torsion_energy_profile.update(
            {
                "frag": data.get("frag", None),
                "frag_indices": data.get("frag_indices", None),
            }
        )

    conf.props["torsion"] = torsion_energy_profile

    return conf


def sp_torsion_energies_batch(
    conf: "Conf",
    batch_calculator: Callable,
    indices: tuple,
    simplify: bool = True,
    interval: float = 20.0,
    water: str | None = None,
    batchsize_atoms: int = 16 * 1024,
) -> "Conf":
    """Calculate Single-Point potential energy profile for torsion angles in a batch.

    Args:
        conf (Conf): reference conformer.
        batch_calculator (Callable): BatchSinglePoint calculator.
        indices (tuple): atom indices (i,j,k,l) for a torsion angle
        simplify (bool, optional): whether to use fragmentation. Defaults to True.
        interval (float, optional): angle intervals. Defaults to 20.0.
        batchsize_atoms (int, optional): maximum number of atoms in a single batch.
            Defaults to 16384 (16 * 1024).

    Args for xTB calculator:
        water (str, optional): water solvation model (choose 'gbsa', 'alpb', 'cpcmx')
            alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
            gbsa: generalized Born (GB) model with Surface Area contributions.
            cpcmx: Extended Conductor-like Polarizable Continuum Solvation Model (CPCM-X).
            Defaults to None.

    Returns:
        Conf: `conf` with `conf.props['torsion']` set to a profile dict with `indices`, `angle`,
            and `E_rel(kcal/mol)` (plus `frag` and `frag_indices` if `simplify`).
            E_rel = E_sp - E_ref (single-point energy relative to the reference conformer),
            shifted so that its minimum is 0.
    """
    ref_conf = conf.copy()  # optimized reference conformer
    evaluated_batch = batch_calculator([ref_conf.rdmol]).run()
    for rdmol in evaluated_batch.mols:
        ref_conf.props["E_tot(kcal/mol)"] = float(rdmol.GetProp("E_tot(kcal/mol)"))

    data = {
        "indices": indices,
        "angle": [],
        "init": [],
        "last": [],
        "Converged": [],
    }

    confs = []
    if simplify:
        (frag, frag_ijkl, frag_created, wbo_filtered) = create_torsion_fragment(
            ref_conf.rdmol, indices
        )
        frag_conf = type(conf)(frag)
        frag_cleaned, _ = clean_2d(frag, reset_isotope=True, remove_H=True)
        # to serialize the molecule
        data["frag"] = Chem.MolToMolBlock(frag_cleaned)
        data["frag_indices"] = frag_ijkl
        for angle in np.arange(-180.0, 180.0, interval):
            # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
            scan_conf = frag_conf.copy()
            scan_conf.set_torsion_angle(*frag_ijkl, angle)  # atoms bonded to `l` move.
            data["angle"].append(angle)
            confs.append(scan_conf)

    else:
        for angle in np.arange(-180.0, 180.0, interval):
            # Iterated numpy.ndarray does not contain the last 180: -180., ..., (180).
            scan_conf = ref_conf.copy()
            scan_conf.set_torsion_angle(*indices, angle)  # atoms bonded to `l` move.
            data["angle"].append(angle)
            confs.append(scan_conf)

    batches = prepare_batches(confs, batchsize_atoms)

    for batch in batches:
        evaluated_batch = batch_calculator(batch.rdmols).run()
        for rdmol in evaluated_batch.mols:
            # batch single point calculator sets 'E_tot(kcal/mol)'
            # batch optimizer updates coordinates and conf.props:
            #   `E_tot_init(kcal/mol)`, `E_tot(kcal/mol)`, `Converged`.
            data["init"].append(float(rdmol.GetProp("E_tot(kcal/mol)")))
            data["last"].append(ref_conf.props["E_tot(kcal/mol)"])

    if len(data["init"]) == 0:
        raise RuntimeError(f"torsion scan of {indices} has no (converged) energy data")

    relax = np.array(data["init"]) - np.median(data["last"])
    E_rel = relax - np.min(relax)

    torsion_energy_profile = {
        "indices": data["indices"],
        "angle": np.round(
            np.array(data["angle"]), 1
        ).tolist(),  # np.ndarray -> list for serialization
        "E_rel(kcal/mol)": np.round(
            E_rel, 2
        ).tolist(),  # np.ndarray -> list for serialization
    }

    if simplify:
        torsion_energy_profile.update(
            {
                "frag": data.get("frag", None),
                "frag_indices": data.get("frag_indices", None),
            }
        )

    conf.props["torsion"] = torsion_energy_profile

    return conf


def calculate_sp_torsion_energies(
    conf: "Conf",
    calculator: str | Callable,
    torsion_angle_idx: int | None = None,
    simplify: bool = True,
    interval: float = 20.0,
    water: str | None = None,
    batchsize_atoms: int = 0,
) -> "Conf":
    """Calculates Single-Point potential energy profiles for each torsion angle using ASE or BatchSinglePoint.

    Args:
        conf (Conf): reference conformer.
        calculator (str | Callable): 'MMFF', 'UFF', 'xTB', ASE calculator, or BatchSinglePoint.
        torsion_angle_idx (int | None): key to the torsion indices to calculate. Defaults to None (all).
        simplify (bool, optional): whether to use fragment surrogate. Defaults to True.
        interval (float, optional): interval of torsion angles in degree. Defaults to 20.0.
        batchsize_atoms (int, optional): maximum number of atoms in a single batch.
            Setting any number smaller than conf.natoms to disable batch optimization.
            Defaults to 0.

    Args for xTB calculator:
        water (str, optional): water solvation model (choose 'gbsa', 'alpb', or 'cpcmx')
            alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
            gbsa: generalized Born (GB) model with Surface Area contributions.
            cpcmx: Extended Conductor-like Polarizable Continuum Solvation Model (CPCM-X).
            Defaults to None.

    Returns:
        Conf: `conf` with `conf.props['torsion']` set to a list of profile dicts, one per torsion,
            each with `indices`, `angle`, `E_rel(kcal/mol)`, and `torsion_angle_idx`
            (plus `frag` and `frag_indices` if `simplify`).
    """

    torsion_angle_atom_indices_list = conf.get_torsion_angle_atoms()
    # [(5, 4, 3, 1), ...]

    if isinstance(torsion_angle_idx, int) and torsion_angle_idx < len(
        torsion_angle_atom_indices_list
    ):
        indices = {
            torsion_angle_idx: torsion_angle_atom_indices_list[torsion_angle_idx]
        }
        # atom indices for a single torsion angle
        # ex. {0: (5,4,3,1)}
    else:
        indices = {k: v for k, v in enumerate(torsion_angle_atom_indices_list)}
        # atom indices for all torsion angles
        # ex. {0: (5,4,3,1), 1: ...}

    work_conf = conf.copy()
    torsion_energies = []
    if batchsize_atoms >= work_conf.natoms:
        for torsion_angle_idx, ijkl in indices.items():
            work_conf = sp_torsion_energies_batch(
                work_conf,
                calculator,
                ijkl,
                simplify,
                interval,
                water=water,
                batchsize_atoms=batchsize_atoms,
            )
            work_conf.props["torsion"].update({"torsion_angle_idx": torsion_angle_idx})
            torsion_energies.append(work_conf.props["torsion"])
    else:
        for torsion_angle_idx, ijkl in indices.items():
            work_conf = sp_torsion_energies(
                work_conf, calculator, ijkl, simplify, interval, water
            )
            work_conf.props["torsion"].update({"torsion_angle_idx": torsion_angle_idx})
            torsion_energies.append(work_conf.props["torsion"])

    conf.props["torsion"] = torsion_energies

    return conf