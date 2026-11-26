"""Energies and geometry optimizations with Amber GAFF/GAFF2 and OpenFF force fields (OpenMM).

These functions follow `Conf.singlepoint()`, `Conf.optimize()`, and
`rdworks.torsion.wavefront.constrained_optimize()` (same props, same return values), so they
can be used for torsion scans and torsion drives.

Force fields (`forcefield`, case-insensitive), or any explicit name of the same kind:
    'GAFF'     -> 'gaff-1.81'                              (openmmforcefields + AmberTools)
    'GAFF2'    -> 'gaff-2.11'                              (openmmforcefields + AmberTools)
    'Parsley'  -> 'openff_unconstrained-1.3.1.offxml'      (openff-toolkit)
    'Sage'     -> 'openff_unconstrained-2.3.0.offxml'      (openff-toolkit)
    'Rosemary' -> 'openff_unconstrained-3.0.0-rc1.offxml'  (openff-toolkit, release candidate)

Partial charges (`charge_method`):
    'nagl'     -> OpenFF NAGL graph neural network AM1-BCC charges (openff-nagl; default, fast)
    'am1bcc'   -> AM1-BCC charges (needs AmberTools `sqm` or OpenEye)
    any other `openff.toolkit.Molecule.assign_partial_charges()` method, e.g. 'gasteiger'

All calculations are in the gas phase (no cutoff). OpenMM systems are cached per molecule
(topology, atom order, and stereochemistry), force field, and charge method, so repeated
calls on conformers of one molecule (e.g. a torsion scan) assign charges only once.

Install the optional dependencies with `pip install rdworks[openff]` (OpenFF force fields);
GAFF/GAFF2 also need `openmmforcefields` and `ambertools` from conda-forge.
"""

import copy
import functools
import logging
import math
import numpy as np

from typing import TYPE_CHECKING
from rdkit import Chem

if TYPE_CHECKING:
    # only for type hints; a runtime import would be circular (rdworks.conf imports rdworks.torsion)
    from rdworks.conf import Conf

logger = logging.getLogger(__name__)

FORCEFIELDS = {
    "gaff": "gaff-1.81",
    "gaff2": "gaff-2.11",
    "parsley": "openff_unconstrained-1.3.1.offxml",
    "sage": "openff_unconstrained-2.3.0.offxml",
    "rosemary": "openff_unconstrained-3.0.0-rc1.offxml",
}

NAGL_MODEL = "openff-gnn-am1bcc-0.1.0-rc.1.pt"

# 1 eV/Angstrom in kJ/mol/nm (ASE `fmax` units -> OpenMM force units)
EV_PER_A_TO_KJ_PER_MOL_NM = 964.8533212
KCAL_TO_KJ = 4.184

# force group of the torsion restraint; it is excluded from reported energies
_RESTRAINT_GROUP = 31
_ENERGY_GROUPS = set(range(_RESTRAINT_GROUP))


def is_supported(calculator) -> bool:
    """Whether `calculator` names a GAFF or OpenFF force field handled by this module.

    Args:
        calculator: calculator passed to torsion scans (str or ASE calculator).

    Returns:
        bool: True for 'GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary', 'gaff-<version>',
            or an OpenFF '.offxml' file name (case-insensitive short names).
    """
    if not isinstance(calculator, str):
        return False
    name = FORCEFIELDS.get(calculator.lower(), calculator)
    return name.startswith("gaff-") or name.endswith(".offxml")


def resolve_forcefield(forcefield: str) -> str:
    """Returns the full force field name for a short name ('GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary').

    Args:
        forcefield (str): short name, or a full name such as 'gaff-2.11' or
            'openff_unconstrained-2.2.1.offxml'.

    Raises:
        ValueError: if `forcefield` is not a GAFF or OpenFF force field name.

    Returns:
        str: full force field name.
    """
    if is_supported(forcefield):
        return FORCEFIELDS.get(forcefield.lower(), forcefield)
    raise ValueError(
        f"Unsupported force field: {forcefield!r} (choose {', '.join(['GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary'])}, "
        "'gaff-<version>', or an OpenFF '.offxml' file name)"
    )


def _mapped_smiles(rdmol: Chem.Mol) -> str:
    """Mapped SMILES (map number = atom index + 1) with stereochemistry from 3D coordinates."""
    mol = Chem.Mol(rdmol)
    if any(atom.GetNumImplicitHs() > 0 for atom in mol.GetAtoms()):
        raise ValueError("force field calculations need explicit hydrogen atoms (use Chem.AddHs)")
    Chem.AssignStereochemistryFrom3D(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(mol)


@functools.lru_cache(maxsize=64)
def _create_system(mapped_smiles: str, forcefield: str, charge_method: str):
    """Creates (and caches) an OpenMM System; callers must not modify it."""
    try:
        from openff.toolkit import Molecule
    except ImportError as e:
        raise ImportError(
            "GAFF/OpenFF calculations need openmm and openff-toolkit: pip install rdworks[openff]"
        ) from e

    molecule = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
    if charge_method.lower() == "nagl":
        molecule.assign_partial_charges(NAGL_MODEL)
    else:
        molecule.assign_partial_charges(charge_method)

    if forcefield.startswith("gaff-"):
        try:
            from openmm import app
            from openmmforcefields.generators import GAFFTemplateGenerator
        except ImportError as e:
            raise ImportError(
                "GAFF/GAFF2 need openmmforcefields and ambertools (conda install -c conda-forge "
                "openmmforcefields ambertools)"
            ) from e
        # the generator uses the charges assigned above instead of computing AM1-BCC charges
        generator = GAFFTemplateGenerator(molecules=[molecule], forcefield=forcefield)
        omm_forcefield = app.ForceField()
        omm_forcefield.registerTemplateGenerator(generator.generator)
        system = omm_forcefield.createSystem(
            molecule.to_topology().to_openmm(),
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
        )
    else:
        from openff.toolkit import ForceField

        system = ForceField(forcefield).create_openmm_system(
            molecule.to_topology(), charge_from_molecules=[molecule]
        )

    if system.getNumParticles() != molecule.n_atoms:
        raise ValueError(f"{forcefield} adds virtual sites, which are not supported")
    if system.getNumConstraints() > 0:
        raise ValueError(
            f"{forcefield} constrains bond lengths; use an unconstrained force field "
            "(e.g. 'openff_unconstrained-2.2.1.offxml') for geometry optimization"
        )

    return system


def _get_system(conf: "Conf", forcefield: str, charge_method: str):
    return _create_system(_mapped_smiles(conf.rdmol), resolve_forcefield(forcefield), charge_method)


def _create_context(system, positions: np.ndarray):
    """OpenMM Context (CPU platform, or Reference if CPU is unavailable) at `positions` (Angstrom)."""
    import openmm
    import openmm.unit

    try:
        platform = openmm.Platform.getPlatformByName("CPU")
    except openmm.OpenMMException:
        platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions(openmm.unit.Quantity(np.asarray(positions) / 10.0, openmm.unit.nanometer))
    return context


def _energy(context) -> float:
    """Potential energy in kcal/mol without the torsion restraint."""
    import openmm.unit

    state = context.getState(getEnergy=True, groups=_ENERGY_GROUPS)
    return state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)


def _minimize(context, fmax: float, max_iter: int) -> bool:
    """Minimizes the energy in place; returns whether the max atomic force is below `fmax` (eV/Angstrom)."""
    import openmm
    import openmm.unit

    fmax_omm = fmax * EV_PER_A_TO_KJ_PER_MOL_NM
    # the minimizer stops on the RMS of force components; a tighter tolerance makes the
    # max atomic force (the ASE `fmax` criterion) fall below `fmax` as well
    openmm.LocalEnergyMinimizer.minimize(context, 0.1 * fmax_omm, max_iter)
    forces = context.getState(getForces=True).getForces(asNumpy=True)
    forces = forces.value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
    return bool(np.linalg.norm(forces, axis=1).max() < fmax_omm)


def _positions(context) -> np.ndarray:
    """Positions in Angstrom."""
    import openmm.unit

    state = context.getState(getPositions=True)
    return state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)


def singlepoint(
    conf: "Conf",
    forcefield: str = "Sage",
    charge_method: str = "nagl",
    ) -> "Conf":
    """Get potential energy and set `E_tot(kcal/mol)` in the conf.props.

    Args:
        conf (Conf): conformer (with explicit hydrogens).
        forcefield (str, optional): 'GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary', or a full
            force field name (see `resolve_forcefield()`). Defaults to 'Sage'.
        charge_method (str, optional): partial charge method. Defaults to 'nagl'.

    Returns:
        Conf: `conf` with `E_tot(kcal/mol)` in its props.
    """
    context = _create_context(_get_system(conf, forcefield, charge_method), conf.positions)
    conf.props.update({"E_tot(kcal/mol)": _energy(context)})

    return conf


def optimize(
    conf: "Conf",
    forcefield: str = "Sage",
    fmax: float = 0.05,
    max_iter: int = 1000,
    charge_method: str = "nagl",
    ) -> "Conf":
    """Optimize 3D geometry.

    Args:
        conf (Conf): conformer (with explicit hydrogens); its coordinates are updated.
        forcefield (str, optional): 'GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary', or a full
            force field name (see `resolve_forcefield()`). Defaults to 'Sage'.
        fmax (float, optional): convergence limit of the max atomic force in eV/Angstrom
            (same units as ASE). Defaults to 0.05.
        max_iter (int, optional): max iterations of the minimizer. Defaults to 1000.
        charge_method (str, optional): partial charge method. Defaults to 'nagl'.

    Returns:
        Conf: `conf` with `E_tot_init(kcal/mol)`, `E_tot(kcal/mol)`, and `Converged` in its props.
    """
    context = _create_context(_get_system(conf, forcefield, charge_method), conf.positions)
    PE_start = _energy(context)
    converged = _minimize(context, fmax, max_iter)
    conf.sync(_positions(context))
    conf.props.update(
        {
            "E_tot_init(kcal/mol)": PE_start,  # energy before optimization
            "E_tot(kcal/mol)": _energy(context),  # energy after optimization
            "Converged": converged,  # True or False
        }
    )

    return conf


def constrained_optimize(
    conf: "Conf",
    forcefield: str,
    indices: tuple[int, ...],
    angle: float,
    fmax: float = 0.05,
    max_iter: int = 1000,
    force_constant: float = 1.0e3,
    charge_method: str = "nagl",
    ) -> "Conf":
    """Optimize a copy of `conf` with the torsion angle `indices` held at `angle`.

    Args:
        conf (Conf): starting conformer (not modified).
        forcefield (str): 'GAFF', 'GAFF2', 'Parsley', 'Sage', 'Rosemary', or a full
            force field name (see `resolve_forcefield()`).
        indices (tuple[int, ...]): atom indices (i,j,k,l) of the torsion angle.
        angle (float): target torsion angle in degrees.
        fmax (float, optional): convergence limit of the max atomic force in eV/Angstrom.
            Defaults to 0.05.
        max_iter (int, optional): max iterations of the minimizer. Defaults to 1000.
        force_constant (float, optional): force constant of the harmonic torsion restraint
            in kcal/mol/rad^2. Defaults to 1.0e3.
        charge_method (str, optional): partial charge method. Defaults to 'nagl'.

    Returns:
        Conf: optimized copy with `E_tot(kcal/mol)` (energy without the restraint) and
            `Converged` in its props.
    """
    import openmm

    (i, j, k, l) = indices
    work = conf.copy()
    work.set_torsion_angle(i, j, k, l, angle)  # atoms bonded to `l` move.

    system = copy.deepcopy(_get_system(work, forcefield, charge_method))  # cached; do not modify
    restraint = openmm.CustomTorsionForce(
        "0.5*k*dtheta^2; dtheta = min(dt, 2*pi - dt); dt = abs(theta - theta0); pi = 3.141592653589793"
    )
    restraint.addGlobalParameter("k", force_constant * KCAL_TO_KJ)  # kJ/mol/rad^2
    restraint.addGlobalParameter("theta0", math.radians(angle))
    restraint.addTorsion(i, j, k, l, [])
    restraint.setForceGroup(_RESTRAINT_GROUP)
    system.addForce(restraint)

    context = _create_context(system, work.positions)
    converged = _minimize(context, fmax, max_iter)
    work.sync(_positions(context))
    work.props.update(
        {
            "E_tot(kcal/mol)": _energy(context),  # energy without the torsion restraint
            "Converged": converged,
        }
    )

    return work
