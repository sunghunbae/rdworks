import copy
import itertools
import json
import logging
import tempfile
import os

from pathlib import Path
from collections import defaultdict
from collections.abc import Callable
from typing import Iterator, Self, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np

# import pandas as pd
# import matplotlib.ticker as ticker
# import matplotlib.pyplot as plt
# import seaborn as sns

from spyrmsd.molecule import Molecule as spyrmsd_Molecule
from spyrmsd.rmsd import rmsdwrapper as spyrmsd_wrapper
from spyrmsd.rmsd import symmrmsd as spyrmsd_symmrmsd


from rdkit import Chem, DataStructs
from rdkit.Chem import (
    rdMolDescriptors,
    AllChem,
    Descriptors,
    QED,
    rdFingerprintGenerator,
    Draw,
    rdDepictor,
    inchi,
    rdDistGeom,
    rdMolAlign,
    rdMolTransforms,
    rdmolops,
)
from rdkit.ML.Cluster import Butina


from PIL import Image

from rdworks.conf import Conf, batch_optimize, batch_singlepoint
from rdworks.std import generate_inchi_key, desalt_smiles, standardize, clean_2d
from rdworks.torsion import get_torsion_angle_atom_indices, get_torsion_plot_data
from rdworks.xml import list_predefined_xml, get_predefined_xml, parse_xml
from rdworks.scaffold import rigid_fragment_indices
from rdworks.descriptor import rd_descriptor, rd_descriptor_f
from rdworks.utils import (
    convert_tril_to_symm,
    QT,
    recursive_round,
    compress_string,
    decompress_string,
    serialize,
    deserialize,
)
from rdworks.units import ev2kcalpermol
from rdworks.cluster.autograph import NMRCLUST, DynamicTreeCut, RCKmeans, AutoGraph
from rdworks.cluster.bitqt import BitQT
import rdworks.writers as rwio  # shared writers (to_xyz, to_sdf, to_svg, ...)
from rdworks.stereoisomers import (
    enumerate_stereoisomers,
    enumerate_ring_bond_stereoisomers,
)


if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class Mol:
    """Container for molecular structure, conformers, and other information."""

    _rdworks_kind = "mol"  # dispatch key for rdworks.writers

    MFP2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    @staticmethod
    def etkdg_params(
        seed: int | None = None, prune_rms: float = -1.0, num_threads: int = 0
    ) -> rdDistGeom.EmbedParameters:
        """Returns a fresh ETKDGv3 parameter object used by `make_confs()`.

        A new object is built on every call (RDKit EmbedParameters cannot be copied),
        so per-call settings such as the random seed never leak between calls.

        Args:
            seed (int | None, optional): random seed for reproducible embedding.
                None means non-deterministic. Defaults to None.
            prune_rms (float, optional): RMSD threshold (Angstrom) for pruning
                near-duplicate conformers during embedding; <= 0 disables. Defaults to -1.0.
            num_threads (int, optional): number of threads; 0 uses all available. Defaults to 0.

        Returns:
            rdDistGeom.EmbedParameters: ETKDGv3 parameters.
        """
        params = rdDistGeom.ETKDGv3()
        params.useSmallRingTorsions = True
        params.maxIterations = 2000
        params.randomSeed = -1 if seed is None else int(seed)
        params.pruneRmsThresh = prune_rms
        params.numThreads = num_threads
        return params

    def __init__(
        self,
        molecule: str | Chem.Mol | Conf | None = None,
        name: str = "",
        std: bool = False,
        reset_isotope: bool = True,
        remove_H: bool = True,
        max_workers: int = 1,
        chunksize: int = 4,
        progress: bool = False,
    ) -> None:
        """Initialize.

        Examples:
            >>> from rdworks import Mol
            >>> m = Mol('c1ccccc1', name='benzene')

        Args:
            molecule (str | Chem.Mol | None): SMILES or rdkit.Chem.Mol or None
            name (str): name of the molecule. Defaults to ''.
            std (bool): whether to standardize the molecule. Defaults to False.
            max_workers (int): number of maximum workers for parallelization. Defaults to 1.
            chunksize (int): batch size for parallelization. Defaults to 4.
            progress (bool): whether to show progress bar. Defaults to False.
        """
        assert isinstance(molecule, str | Chem.Mol | Conf) or molecule is None

        self.name = ""
        self.rdmol = None  # 2D, one and only one Conformer
        self.smiles = ""  # isomeric SMILES
        self.confs = []  # container for 3D conformers
        self.InChIKey = ""  # 27 characters (SHA-256 hash of InChI)
        self.props = {}

        self.max_workers = max_workers
        self.chunksize = chunksize
        self.progress = progress
        self.is_confs_aligned = False
        self.fp = None

        if molecule is None:
            return

        if isinstance(molecule, str):  # 1-D SMILES
            try:
                if "." in molecule:  # mandatory desalting
                    (self.smiles, self.rdmol) = desalt_smiles(molecule)
                else:
                    self.rdmol = Chem.MolFromSmiles(molecule)
                    self.smiles = Chem.MolToSmiles(self.rdmol)
            except (TypeError, ValueError, RuntimeError) as e:
                # invalid SMILES -> MolFromSmiles returns None -> TypeError downstream
                raise ValueError(f"Mol() Error: invalid SMILES {molecule}") from e

        elif isinstance(molecule, Chem.Mol):  # 2-D or 3-D Chem.Mol
            try:
                self.rdmol, _ = clean_2d(molecule, reset_isotope, remove_H)
                self.smiles = Chem.MolToSmiles(self.rdmol)
                self.confs = [Conf(x) for x in _]
            except (TypeError, ValueError, RuntimeError) as e:
                raise ValueError("Mol() Error: invalid Chem.Mol object") from e

        elif isinstance(molecule, Conf):  # 3-D input
            try:
                self.rdmol, _ = clean_2d(molecule.rdmol, reset_isotope, remove_H)
                self.smiles = Chem.MolToSmiles(self.rdmol)
                self.confs = [molecule]
            except (TypeError, ValueError, RuntimeError) as e:
                raise ValueError("Mol() Error: invalid Conf object") from e

        try:
            if std:
                self.rdmol = standardize(self.rdmol)
                self.smiles = Chem.MolToSmiles(self.rdmol)
        except (TypeError, ValueError, RuntimeError) as e:
            raise RuntimeError("Mol() Error: standardization") from e

        assert self.smiles and self.rdmol, "Mol() Error: invalid molecule"

        rdDepictor.Compute2DCoords(self.rdmol)

        self.name = "" if name is None else str(name)

        self.rdmol.SetProp("_Name", self.name)  # _Name can't be None
        self.InChIKey = generate_inchi_key(self.rdmol)
        self.props.update(
            {
                "aka": [],  # <-- to be set by MolLibr.unique()
                "atoms": self.rdmol.GetNumAtoms(),  # hydrogens not excluded?
                "charge": rdmolops.GetFormalCharge(self.rdmol),
                "nrb": Descriptors.NumRotatableBonds(self.rdmol),
            }
        )

    def __str__(self) -> str:
        """String representation of the molecule.

        Examples:
            >>> m = Mol('CCO', name='ethanol')
            >>> print(m)

        Returns:
            str: string representation.
        """
        return f"<Mol({self.smiles} name={self.name} conformers={self.num_confs})>"

    def __hash__(self) -> int:
        """Hashed InChIKey of the molecule.

        Must be consistent with `__eq__` (which compares InChIKeys): objects that
        compare equal must have the same hash, otherwise sets and dict keys break.

        Examples:
            >>> m1 == m2

        Returns:
            int: hash of the InChIKey (canonical SMILES if no InChIKey).
        """
        return hash(self._identity_key())

    def _identity_key(self) -> str:
        """InChIKey, or canonical SMILES when InChI is unavailable (e.g. dummy atoms)."""
        return self.InChIKey or self.smiles

    def __eq__(self, other: Self) -> bool:
        """True if `other` Mol is identical with this Mol.

        It compares InChIKeys.

        Examples:
            >>> m1 == m2

        Args:
            other (object): other Mol object.

        Returns:
            bool: True if identical.
        """
        if not isinstance(other, Mol):
            return NotImplemented
        return self._identity_key() == other._identity_key()

    def __iter__(self) -> Iterator:
        """Yields an iterator of conformers of the molecule.

        Examples:
            >>> for conformer in mol:
            >>>     print(conformer.name)

        Yields:
            Iterator: conformers of the molecule.
        """
        return iter(self.confs)

    def __getitem__(self, index: int | slice) -> Conf | Self:
        """Conformer object of conformers of the molecule with given index or slice of indexes.

        Examples:
            >>> first_conformer = mol[0]

        Args:
            index (int | slice): index for conformers.

        Returns:
            Conf or Mol(copy) with conformers specified by index.
        """
        assert self.num_confs > 0, "no conformers"

        if isinstance(index, slice):
            new_object = self.copy()
            new_object.confs = new_object.confs[index]
            return new_object

        else:
            return self.confs[index]

    ##################################################
    ### Properties
    ##################################################

    @property
    def num_confs(self) -> int:
        """Returns the total number of conformers.

        Returns:
            int: total count of conformers.
        """
        return len(self.confs)

    @property
    def charge(self) -> int:
        """Returns molecular formal charge

        Returns:
            int: molecular formal charge
        """
        return rdmolops.GetFormalCharge(self.rdmol)

    @property
    def symbols(self) -> list[str]:
        """Returns the element symbols.

        Returns:
            list: list of element symbols.
        """
        return [atom.GetSymbol() for atom in self.rdmol.GetAtoms()]

    @property
    def numbers(self) -> list[int]:
        """Returns the atomic numbers.

        Returns:
            list: list of atomic numbers.
        """
        return [atom.GetAtomicNum() for atom in self.rdmol.GetAtoms()]

    @property
    def molblock(self) -> str:
        """Returns MolBlock"""
        return Chem.MolToMolBlock(self.rdmol)

    @property
    def is_stereo_specified(self) -> bool:
        """Check if the molecule is stereo-specified at tetrahedral atom and double bond.

        This function uses `Chem.FindPotentialStereo()` function which returns a list of `elements`.
        Explanation of the elements:
            element.type:
                whether the element is a stereocenter ('stereoAtom') or a stereobond ('stereoBond')
                - Atom_Octahedral
                - Atom_SquarePlanar
                - *Atom_Tetrahedral*
                - Atom_TrigonalBipyramidal
                - Bond_Atropisomer
                - Bond_Cumulene_Even
                - *Bond_Double*m.
                - Unspecified

            element.centeredOn:
                The atom or bond index where the stereochemistry is centered.

            element.specified:
                A boolean indicating whether the stereochemistry at that location
                is explicitly specified in the molecule.
                values = {
                    0: rdkit.Chem.rdchem.StereoSpecified.Unspecified,
                    1: rdkit.Chem.rdchem.StereoSpecified.Specified,
                    2: rdkit.Chem.rdchem.StereoSpecified.Unknown,
                    }

            element.descriptor:
                A descriptor that can be used to identify the type of stereochemistry (e.g., 'R', 'S', 'E', 'Z').
                - Bond_Cis = rdkit.Chem.StereoDescriptor.Bond_Cis
                - Bond_Trans = rdkit.Chem.StereoDescriptor.Bond_Trans
                - NoValue = rdkit.Chem.StereoDescriptor.NoValue
                - Tet_CCW = rdkit.Chem.StereoDescriptor.Tet_CCW
                - Tet_CW = rdkit.Chem.StereoDescriptor.Tet_CW

        Returns:
            bool: True if stereo-specified.
        """
        stereos = []
        for element in Chem.FindPotentialStereo(self.rdmol):
            if element.type == Chem.StereoType.Atom_Tetrahedral:
                stereos.append(element.specified == Chem.StereoSpecified.Specified)
            elif element.type == Chem.StereoType.Bond_Double:
                bond = self.rdmol.GetBondWithIdx(element.centeredOn)
                if (
                    bond.GetBeginAtom().GetSymbol() == "N"
                    or bond.GetEndAtom().GetSymbol() == "N"
                ):
                    continue
                else:
                    stereos.append(element.specified == Chem.StereoSpecified.Specified)

        # note all([]) returns True
        return all(stereos)

    @property
    def ring_bond_stereo_info(self) -> list[tuple]:
        """Returns double bond and cis/trans stereochemistry information.

        Returns:
            list[tuple]: [(element.centeredOn, element.descriptor), ...]
        """
        stereo_info = Chem.FindPotentialStereo(self.rdmol)
        info = []
        for element in stereo_info:
            if element.type == Chem.StereoType.Bond_Double:
                if self.rdmol.GetBondWithIdx(element.centeredOn).IsInRing():
                    info.append((element.centeredOn, element.descriptor))

        return info

    @property
    def num_stereoisomers(self) -> int:
        """Counts number of all possible stereoisomers ignoring the current stereochemistry.

        Returns:
            int: number of stereoisomers.
        """

        ring_bond_stereo_info = self.ring_bond_stereo_info
        mol = self.copy()
        # remove stereochemistry
        mol = mol.remove_stereo()
        rdmols = enumerate_stereoisomers(mol.rdmol)
        # ring bond stereo is not properly enumerated
        # cis/trans information is lost if stereochemistry is removed,
        # which cannot be enumerated by EnumerateStereoisomers() function
        # so enumerate_ring_bond_stereoisomers() is introduced
        if len(ring_bond_stereo_info) > 0:
            ring_cis_trans = []
            for rdmol in rdmols:
                ring_cis_trans += enumerate_ring_bond_stereoisomers(
                    rdmol, ring_bond_stereo_info, override=True
                )
            if len(ring_cis_trans) > 0:
                rdmols = ring_cis_trans

        unique_rdmols = set([Chem.MolToSmiles(rdmol) for rdmol in rdmols])

        return len(unique_rdmols)

    ##################################################
    ### Pipeline Functions (returns Self)
    ##################################################

    def from_molblock(self, molblock: str, compressed: bool = False) -> Self:
        """Initialize a new Mol object from MolBlock.

        Args:
            molblock (str): MolBlock string

        Raises:
            ValueError: invalid MolBlock

        Returns:
            Self: self.
        """

        if compressed:
            molblock = decompress_string(molblock)

        molecule = Chem.MolFromMolBlock(molblock)

        try:
            self.rdmol, _ = clean_2d(molecule, reset_isotope=True, remove_H=True)
            self.smiles = Chem.MolToSmiles(self.rdmol)
            self.confs = [Conf(x) for x in _]
        except (TypeError, ValueError, RuntimeError) as e:
            raise ValueError("Mol() Error: invalid MolBlock string") from e

        assert self.smiles and self.rdmol, "Mol() Error: invalid molecule"

        name = self.rdmol.GetProp("_Name")

        rdDepictor.Compute2DCoords(self.rdmol)

        self.name = str(name)

        self.rdmol.SetProp("_Name", self.name)  # _Name can't be None
        self.InChIKey = generate_inchi_key(self.rdmol)
        self.props.update(
            {
                "aka": [],  # <-- to be set by MolLibr.unique()
                "atoms": self.rdmol.GetNumAtoms(),  # hydrogens not excluded?
                "charge": rdmolops.GetFormalCharge(self.rdmol),
                "nrb": Descriptors.NumRotatableBonds(self.rdmol),
            }
        )

        return self

    def deserialize(self, serialized: str) -> Self:
        """De-serialize the information and build a new Mol object.

        Example:
            serialized = mol1.serialize()
            mol2 = Mol().deserialize(serialized)

        Args:
            serialized (str): serialized string.

        Returns:
            Self: modified self.
        """
        data = deserialize(serialized)

        self.name = data["name"]
        self.smiles = data["smiles"]  # isomeric SMILES, no H

        try:
            self.rdmol = Chem.MolFromMolBlock(data["molblock"])
        except KeyError:
            # backward-compatible
            self.rdmol = Chem.MolFromSmiles(data["smiles"])  # for 2D depiction
            self.rdmol.SetProp("_Name", self.name)

        self.InChIKey = data["InChIKey"]
        self.props = data["props"]
        self.confs = [
            Conf().deserialize(_) for _ in data["confs"]
        ]  # for 3D conformers (iterable)

        return self

    def count(self) -> int:
        """Returns the number of conformers"""
        return len(self.confs)

    def copy(self) -> Self:
        """Returns a copy of self.

        Returns:
            a copy of self.
        """
        return copy.deepcopy(self)

    def rename(self, prefix: str = "", sep: str = "/", start: int = 1) -> Self:
        """Updates name and conformer names.

        The first conformer name is {prefix}{sep}{start}

        Args:
            prefix (str, optional): prefix of the name. Defaults to ''.
            sep (str, optional): separtor betwween prefix and serial number. Defaults to '/'.
            start (int, optional): first serial number. Defaults to 1.

        Returns:
            Self: modified self.
        """
        if prefix:
            self.name = prefix
            self.rdmol.SetProp("_Name", prefix)

        # update conformer names
        num_digits = len(str(self.num_confs))  # ex. '100' -> 3
        for serial, conf in enumerate(self.confs, start=start):
            serial_str = str(serial)
            while len(serial_str) < num_digits:
                serial_str = "0" + serial_str
            conf.rename(f"{self.name}{sep}{serial_str}")

        return self

    def qed(
        self, properties: list[str] = ["QED", "MolWt", "LogP", "TPSA", "HBD"]
    ) -> Self:
        """Updates quantitative estimate of drug-likeness (QED) and other descriptors.

        Args:
            properties (list[str], optional): Defaults to ['QED', 'MolWt', 'LogP', 'TPSA', 'HBD'].

        Raises:
            KeyError: if property key is unknown.

        Returns:
            Self: modified self.
        """
        props_dict = {}
        for k in properties:
            try:
                descriptor_fn = rd_descriptor_f[k]
            except KeyError:
                raise KeyError(f"qed() Error: unknown property {k}") from None
            props_dict[k] = descriptor_fn(self.rdmol)
        self.props.update(props_dict)

        return self

    def remove_stereo(self) -> Self:
        """Removes stereochemistry.

        Examples:
            >>> m = Mol("C/C=C/C=C\\C", "double_bond")
            >>> m.remove_stereo().smiles == "CC=CC=CC"

        Returns:
            Self: modified self.
        """
        # keep the original stereo info. for ring double bond
        Chem.RemoveStereochemistry(self.rdmol)
        Chem.AssignStereochemistry(
            self.rdmol, cleanIt=False, force=False, flagPossibleStereoCenters=False
        )
        self.smiles = Chem.MolToSmiles(self.rdmol)

        return self

    def make_confs(
        self,
        n: int = 50,
        method: str = "ETKDG",
        calculator: str | Callable = "MMFF94",
        water: str | None = None,
        batchsize_atoms: int = 0,
        seed: int | None = None,
        prune_rms: float = -1.0,
        verbose: bool = False,
    ) -> Self:
        """Generates 3D conformers and ranks them by single-point energy.

        Sets `E_tot(kcal/mol)` and `E_rel(kcal/mol)` for each conformer, sorted by
        ascending energy. Existing conformers are replaced only on success.

        Args:
            n (int, optional): number of conformers to generate. Defaults to 50.
            method (str, optional): conformer generation method. Only `ETKDG`
                (RDKit ETKDGv3) is supported. Defaults to 'ETKDG'.
            calculator (str | Callable, optional): energy model used for ranking,
                passed to `singlepoint_confs()`. If `MMFF94`/`MMFF94s` lacks parameters
                for the molecule (e.g. B, Se, metals), `UFF` is used instead with a warning.
                Defaults to 'MMFF94'.
            water (str | None, optional): xTB solvation model ('gbsa' or 'alpb'). Defaults to None.
            batchsize_atoms (int, optional): batch size for Callable calculators,
                passed to `singlepoint_confs()`. Defaults to 0.
            seed (int | None, optional): random seed for reproducible conformers.
                Defaults to None (non-deterministic).
            prune_rms (float, optional): RMSD threshold (Angstrom) for pruning
                near-duplicate conformers during embedding; <= 0 disables. Defaults to -1.0.
            verbose (bool, optional): log a summary. Defaults to False.

        Raises:
            ValueError: if `n` < 1 or `method` is not supported.
            RuntimeError: if no conformer could be embedded.

        Returns:
            Self: modified self.
        """
        if n < 1:
            raise ValueError(f"make_confs() expects n >= 1, got {n}")

        if method.upper() != "ETKDG":
            raise ValueError(
                f"make_confs() unsupported method {method!r}; supported: 'ETKDG'"
            )

        rdmol_H = Chem.AddHs(
            self.rdmol, addCoords=True
        )  # returns a copy with hydrogens added
        rdDistGeom.EmbedMultipleConfs(
            rdmol_H,
            numConfs=n,
            params=Mol.etkdg_params(seed=seed, prune_rms=prune_rms),
        )
        if rdmol_H.GetNumConformers() == 0:
            raise RuntimeError(
                f"make_confs() failed to embed any conformer for {self.name!r} ({self.smiles})"
            )

        confs = []
        for rdConformer in rdmol_H.GetConformers():
            # number of atoms should match with conformer(s)
            rdmol_conf = Chem.Mol(rdmol_H)
            rdmol_conf.RemoveAllConformers()
            rdmol_conf.AddConformer(Chem.Conformer(rdConformer))
            confs.append(Conf(rdmol_conf))

        # fall back to UFF when MMFF has no parameters for this molecule
        if (
            isinstance(calculator, str)
            and calculator.lower() in ("mmff94", "mmff", "mmff94s")
            and not Chem.rdForceFieldHelpers.MMFFHasAllMoleculeParams(rdmol_H)
        ):
            logger.warning(
                f"make_confs() {calculator} has no parameters for {self.name!r}; using UFF instead"
            )
            calculator = "UFF"

        # replace existing conformers only after embedding succeeded;
        # restore them if the energy evaluation fails
        previous_confs = self.confs
        self.confs = confs
        try:
            # energy evaluations for ranking
            self.singlepoint_confs(
                calculator, water=water, batchsize_atoms=batchsize_atoms
            )
        except Exception:
            self.confs = previous_confs
            raise

        # set relative energy, E_rel(kcal/mol)
        sort_by = "E_tot(kcal/mol)"
        self.confs = sorted(
            self.confs, key=lambda c: c.props[sort_by]
        )  # ascending order
        lowest_energy = self.confs[0].props[sort_by]
        for conf in self.confs:
            conf.props.update({"E_rel(kcal/mol)": conf.props[sort_by] - lowest_energy})

        # rename conformers
        self.rename()

        if verbose:
            rot_bonds = rd_descriptor_f["RotBonds"](self.rdmol)
            nrb_suggested = int(8.481 * (rot_bonds**1.642))
            logger.info(
                f"make_confs() rotatable bonds {rot_bonds} (suggested conformers {nrb_suggested}) generated {self.num_confs}"
            )
            logger.info(
                f"make_confs() updated potential energies E_tot(kcal/mol) and E_rel(kcal/mol) by {calculator}"
            )

        return self

    @staticmethod
    def _map_singlepoint(conf: Conf, targs: tuple) -> Conf:
        """A map function to run conf.singlepoint().

        Args:
            mol (Mol): subject rdworks.Mol object.
            targs (tuple): a tuple of rdworks.Mol objects to compare.

        Returns:
            bool: True if molecule is similar with target molecules.
        """
        return conf.singlepoint(*targs)  # unpack tuple of arguments

    def singlepoint_confs(
        self,
        calculator: str | Callable,
        water: str | None = None,
        batchsize_atoms: int = 0,
    ) -> Self:
        """Evaluates potential energy of each conformer without geometry optimization.

        It sets `E_tot(kcal/mol)` property for each conformer.

        Args:
            calculator (str | Callable): MMFF94 (= MMFF), MMFF94s, UFF, xTB or ASE calculator.
                `MMFF94` or `MMFF` - Intended for general use, including organic molecules and proteins,
                    and primarily relies on data from quantum mechanical calculations.
                    It's often used in molecular dynamics simulations.
                `MMFF94s` - A "static" variant of MMFF94, with adjusted parameters for out-of-plane
                    bending and dihedral torsions to favor planar geometries for specific nitrogen atoms.
                    This makes it better suited for geometry optimization studies where a static,
                    time-averaged structure is desired. The "s" stands for "static".
                `UFF` - UFF refers to the "Universal Force Field," a force field model used for
                    molecular mechanics calculations. It's a tool for geometry optimization,
                    energy minimization, and exploring molecular conformations in 3D space.
                    UFF is often used to refine conformers generated by other methods,
                    such as random conformer generation, to produce more physically plausible
                    and stable structures.
            fmax (float, optional): fmax for the calculator convergence. Defaults to 0.05.
            max_iter (int, optional): max iterations for the calculator. Defaults to 1000.
            batchsize_atoms (int, optional): maximum number of atoms in a single batch.
                Setting any number smaller than conf.natoms to disable batch optimization.
                Defaults to 0.

        Args for xTB calculator:
            water (str, optional): water solvation model (choose 'gbsa' or 'alpb')
                alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
                gbsa: generalized Born (GB) model with Surface Area contributions.
                Defaults to None.

        Returns:
            Self: modified self.
        """
        if isinstance(calculator, Callable):
            if batchsize_atoms >= self.confs[0].natoms:
                self.confs = batch_singlepoint(self.confs, calculator, batchsize_atoms)
            else:
                self.confs = [
                    conf.singlepoint(calculator, water=water) for conf in self.confs
                ]
        else:
            if self.max_workers > 1:
                largs = [(calculator, water) for _ in self.confs]
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    if self.progress:
                        self.confs = list(
                            tqdm(
                                executor.map(
                                    Mol._map_singlepoint, self.confs, largs, chunksize=1
                                ),
                                desc="Singlepoint Conformers",
                                total=self.count(),
                            )
                        )
                    else:
                        self.confs = list(
                            executor.map(
                                Mol._map_singlepoint, self.confs, largs, chunksize=1
                            )
                        )
            else:
                self.confs = [
                    conf.singlepoint(calculator, water=water) for conf in self.confs
                ]

        return self

    @staticmethod
    def _map_optimize(conf: Conf, targs: tuple) -> Conf:
        """A map function to run conf.optimize().

        Args:
            mol (Mol): subject rdworks.Mol object.
            targs (tuple): a tuple of rdworks.Mol objects to compare.

        Returns:
            bool: True if molecule is similar with target molecules.
        """
        return conf.optimize(*targs)  # unpack tuple of arguments

    def optimize_confs(
        self,
        calculator: str | Callable = "MMFF94",
        fmax: float = 0.05,
        max_iter: int = 1000,
        water: str | None = None,
        batchsize_atoms: int = 0,
    ) -> Self:
        """Optimizes 3D geometry of conformers.

        Args:
            calculator (str | Callable): MMFF94 (= MMFF), MMFF94s, UFF, or ASE calculator.
                `MMFF94` or `MMFF` - Intended for general use, including organic molecules and proteins,
                    and primarily relies on data from quantum mechanical calculations.
                    It's often used in molecular dynamics simulations.
                `MMFF94s` - A "static" variant of MMFF94, with adjusted parameters for out-of-plane
                    bending and dihedral torsions to favor planar geometries for specific nitrogen atoms.
                    This makes it better suited for geometry optimization studies where a static,
                    time-averaged structure is desired. The "s" stands for "static".
                `UFF` - UFF refers to the "Universal Force Field," a force field model used for
                    molecular mechanics calculations. It's a tool for geometry optimization,
                    energy minimization, and exploring molecular conformations in 3D space.
                    UFF is often used to refine conformers generated by other methods,
                    such as random conformer generation, to produce more physically plausible
                    and stable structures.
            fmax (float, optional): fmax for the calculator convergence. Defaults to 0.05.
            max_iter (int, optional): max iterations for the calculator. Defaults to 1000.
            batchsize_atoms (int, optional): max number of atoms in one batch.
                Defaults to 16384(=16*1024). Disable batch optimization if zero or negative.

        Args for xTB calculator:
            water (str, optional): water solvation model (choose 'gbsa' or 'alpb')
                alpb: ALPB solvation model (Analytical Linearized Poisson-Boltzmann).
                gbsa: generalized Born (GB) model with Surface Area contributions.
                Defaults to None.

        Returns:
            Self: modified self.
        """
        if isinstance(calculator, Callable):
            if batchsize_atoms >= self.confs[0].natoms:
                self.confs = batch_optimize(self.confs, calculator, batchsize_atoms)
            else:
                self.confs = [
                    conf.optimize(calculator, fmax, max_iter, water)
                    for conf in self.confs
                ]
        else:
            if self.max_workers > 1:
                largs = [(calculator, fmax, max_iter, water) for _ in self.confs]
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    if self.progress:
                        self.confs = list(
                            tqdm(
                                executor.map(
                                    Mol._map_optimize, self.confs, largs, chunksize=1
                                ),
                                desc="Optimize Conformers",
                                total=self.count(),
                            )
                        )
                    else:
                        self.confs = list(
                            executor.map(
                                Mol._map_optimize, self.confs, largs, chunksize=1
                            )
                        )
            else:
                self.confs = [
                    conf.optimize(calculator, fmax, max_iter, water)
                    for conf in self.confs
                ]

        return self

    def sort_confs(self, calculator: str | Callable | None = None, **kwargs) -> Self:
        """Sorts by `E_tot(kcal/mol)` or `E_tot(eV)` and sets `E_rel(kcal/mol)`.

        Args:
            calculator (str | Callable | None): MMFF94 (= MMFF), MMFF94s, UFF, or ASE calculator.
                `MMFF94` or `MMFF` - Intended for general use, including organic molecules and proteins,
                    and primarily relies on data from quantum mechanical calculations.
                    It's often used in molecular dynamics simulations.
                `MMFF94s` - A "static" variant of MMFF94, with adjusted parameters for out-of-plane
                    bending and dihedral torsions to favor planar geometries for specific nitrogen atoms.
                    This makes it better suited for geometry optimization studies where a static,
                    time-averaged structure is desired. The "s" stands for "static".
                `UFF` - UFF refers to the "Universal Force Field," a force field model used for
                    molecular mechanics calculations. It's a tool for geometry optimization,
                    energy minimization, and exploring molecular conformations in 3D space.
                    UFF is often used to refine conformers generated by other methods,
                    such as random conformer generation, to produce more physically plausible
                    and stable structures.
        Raises:
            KeyError: if `E_tot(eV)` or `E_tot(kcal/mol)` is not defined.

        Returns:
            Self: modified self.
        """
        verbose = kwargs.get("verbose", False)

        if calculator is not None:
            # re-calculate potential energies
            if verbose:
                logger.info(f"sort_confs() calculate potential energy by {calculator}")
            self.singlepoint_confs(
                calculator,
                water=kwargs.get("water"),
                batchsize_atoms=kwargs.get("batchsize_atoms", 0),
            )

        if all(["E_tot(kcal/mol)" in conf.props for conf in self.confs]):
            sort_by = "E_tot(kcal/mol)"
            conversion = 1.0

        elif all(["E_tot(eV)" in conf.props for conf in self.confs]):
            sort_by = "E_tot(eV)"
            conversion = ev2kcalpermol  # eV to kcal/mol

        else:
            raise KeyError(
                f"sort_confs() requires `E_tot(eV)` or `E_tot(kcal/mol)` property"
            )

        # ascending order
        self.confs = sorted(self.confs, key=lambda c: c.props[sort_by])

        if self.num_confs > 0:
            E_min = self.confs[0].props[sort_by]
            for conf in self.confs:
                E_rel = conversion * (conf.props[sort_by] - E_min)
                conf.props.update({"E_rel(kcal/mol)": E_rel})

        return self

    def align_confs(self, method: str = "rigid_fragment") -> Self:
        """Aligns all conformers to the first conformer.

        Args:
            method (str, optional): alignment method:
                `rigid_fragment`, `CrippenO3A`, `MMFFO3A`, `best_rms`.
                Defaults to `rigid_fragment`.

        Returns:
            Self: modified self.
        """

        supported = ("rigid_fragment", "CrippenO3A", "MMFFO3A", "best_rms")
        if method not in supported:
            raise ValueError(
                f"align_confs() unsupported method {method!r}; supported: {supported}"
            )

        if self.num_confs < 2:
            # nothing to do
            return self

        if method == "rigid_fragment":
            indices = rigid_fragment_indices(self.confs[0].rdmol)[
                0
            ]  # 3D and H, largest fragment
            atomMap = [(i, i) for i in indices]
            for i in range(1, self.num_confs):
                rmsd = rdMolAlign.AlignMol(
                    prbMol=self.confs[i].rdmol,
                    refMol=self.confs[0].rdmol,
                    atomMap=atomMap,
                )
                # rdMolAlign.AlignMol does not take symmetry into account
                # but we will use atom indices for alignment anyway.
                # If atomMap is not given, AlignMol() will attempt to generate atomMap by
                # substructure matching.
            self.is_confs_aligned = True

        elif method == "CrippenO3A":
            crippen_ref_contrib = rdMolDescriptors._CalcCrippenContribs(
                self.confs[0].rdmol
            )
            for i in range(1, self.num_confs):
                crippen_prb_contrib = rdMolDescriptors._CalcCrippenContribs(
                    self.confs[i].rdmol
                )
                crippen_O3A = rdMolAlign.GetCrippenO3A(
                    prbMol=self.confs[i].rdmol,
                    refMol=self.confs[0].rdmol,
                    prbCrippenContribs=crippen_prb_contrib,
                    refCrippenContribs=crippen_ref_contrib,
                )
                crippen_O3A.Align()
                # crippen_O3A.Score()
            self.is_confs_aligned = True

        elif method == "MMFFO3A":
            mmff_ref_params = AllChem.MMFFGetMoleculeProperties(self.confs[0].rdmol)
            for i in range(1, self.num_confs):
                mmff_prb_params = AllChem.MMFFGetMoleculeProperties(self.confs[i].rdmol)
                mmff_O3A = rdMolAlign.GetO3A(
                    prbMol=self.confs[i].rdmol,
                    refMol=self.confs[0].rdmol,
                    prbPyMMFFMolProperties=mmff_prb_params,
                    refPyMMFFMolProperties=mmff_ref_params,
                )
                mmff_O3A.Align()
                # mmff_O3A.Score()
            self.is_confs_aligned = True

        elif method == "best_rms":
            for i in range(1, self.num_confs):
                # symmetry-aware alignment / speed can be improved by removing Hs
                rmsd = rdMolAlign.GetBestRMS(
                    prbMol=self.confs[i].rdmol, refMol=self.confs[0].rdmol
                )
            self.is_confs_aligned = True

        return self

    def cluster_confs(
        self,
        method: str = "QT",
        threshold: float = 1.0,
        sort: str = "size",
        symmetry_aware: bool = True,
    ) -> Self:
        """Clusters all conformers and sets cluster properties.

        Conformers are expected to be aligned.

        Following cluster properties will be added: `cluster`, `cluster_mean_energy`,
            `cluster_median_energy`, `cluster_IQR_energy`, `cluster_size`, `cluster_centroid` (True or False)

        `RCKMeans` algorithm is unreliable and not supported for now.

        Args:
            method (str, optional): clustering algorithm:
                `Butina`,
                `QT`,
                `NMRCLUST`,
                `DQT`,
                `BitQT`,
                `DynamicTreeCut`,
                `AutoGraph`.
                Defaults to `QT`.
            threshold (float, optional): RMSD threshold of a cluster. Defaults to 1.0.
            sort (str, optional): sort cluster(s) by mean `energy` or cluster `size`.
                Defaults to `size`.
            symmetry_aware (bool, optional): whether to use symmetry-aware rmsd.
                Defaults to True.

        Raises:
            NotImplementedError: if unsupported method is requested.

        Returns:
            Self: modified self.
        """
        if not self.is_confs_aligned:
            self = self.align_confs()

        if method != "DQT":  # rmsd of x,y,z coordinates (non-H)
            conf_rdmols_noH = [
                Chem.RemoveHs(Chem.Mol(conf.rdmol)) for conf in self.confs
            ]
            # copies are made for rmsd calculations to prevent coordinates changes
            lower_triangle_values = []
            if symmetry_aware:
                # symmetry-aware RMSD calculation using spyrmsd
                for i in range(1, self.num_confs):
                    mol_i = spyrmsd_Molecule.from_rdkit(self.confs[i].rdmol)
                    mol_js = [
                        spyrmsd_Molecule.from_rdkit(self.confs[j].rdmol)
                        for j in range(i)
                    ]
                    rmsd_list = spyrmsd_wrapper(
                        mol_i, mol_js, symmetry=True, minimize=False
                    )
                    # it just calculates rmsd without aligning (minimize=False)
                    lower_triangle_values += rmsd_list
            else:
                for i in range(1, self.num_confs):  # number of conformers
                    for j in range(i):
                        # rdMolAlign.GetBestRMS takes symmetry into account
                        # removed hydrogens to speed up
                        best_rms = rdMolAlign.GetBestRMS(
                            prbMol=conf_rdmols_noH[i], refMol=conf_rdmols_noH[j]
                        )
                        lower_triangle_values.append(best_rms)

        else:  # rmsd (radian) of dihedral angles
            torsion_angle_atom_indices_list = self.get_torsion_angle_atoms()
            # [(5, 4, 3, 1), ...]
            # symmmetry-related equivalence is not considered
            torsions = []
            for conf in self.confs:
                t_radians = []
                for i, j, k, l in torsion_angle_atom_indices_list:
                    t_radians.append(
                        rdMolTransforms.GetDihedralRad(
                            conf.rdmol.GetConformer(), i, j, k, l
                        )
                    )
                torsions.append(np.array(t_radians))
            # torsions: num.confs x num.torsions
            N = len(torsions)
            num_torsions = max(len(torsion_angle_atom_indices_list), 1)
            lower_triangle_values = []
            for i in range(N):
                for j in range(i):
                    # wrap angular difference into [-pi, pi) (e.g. 179 vs -179 deg -> 2 deg)
                    rad_diff = (torsions[i] - torsions[j] + np.pi) % (2.0 * np.pi) - np.pi
                    # root-mean-square over torsion angles
                    rmsd = np.sqrt(np.sum(rad_diff**2) / num_torsions)
                    # np.max(np.absolute(rad_diff))
                    lower_triangle_values.append(rmsd)

        cluster_assignment = None
        centroid_indices = None

        if method == "Butina":
            clusters = Butina.ClusterData(
                data=lower_triangle_values,
                nPts=self.num_confs,
                distThresh=threshold,
                isDistData=True,
                reordering=True,
            )
            cluster_assignment = [
                None,
            ] * self.num_confs
            centroid_indices = []
            for cluster_idx, indices in enumerate(clusters):
                for conf_idx in indices:
                    cluster_assignment[conf_idx] = cluster_idx
                centroid_indices.append(indices[0])

        elif method == "QT":
            # my implementation of the original QT algorithm
            # tighter than Butina
            symm_matrix = convert_tril_to_symm(lower_triangle_values)
            cluster_assignment, centroid_indices = QT(symm_matrix, threshold)

        elif method == "NMRCLUST":
            # looser than Butina
            # does not require threshold
            symm_matrix = convert_tril_to_symm(lower_triangle_values)
            cluster_assignment, centroid_indices = NMRCLUST(symm_matrix)

        elif method == "DQT":
            # issues with symmetry related multiplicities
            symm_matrix = convert_tril_to_symm(lower_triangle_values)
            cluster_assignment, centroid_indices = QT(symm_matrix, threshold)

        elif method == "BitQT":
            # supposed to produce identical result as QT but it does not
            symm_matrix = convert_tril_to_symm(lower_triangle_values)
            cluster_assignment, centroid_indices = BitQT(symm_matrix, threshold)

        elif method == "DynamicTreeCut":
            # often collapses into single cluster. so not very useful.
            symm_matrix = convert_tril_to_symm(lower_triangle_values)
            cluster_assignment, centroid_indices = DynamicTreeCut(symm_matrix)

        # elif method == 'RCKmeans':
        #     # buggy
        #     symm_matrix = convert_tril_to_symm(lower_triangle_values)
        #     cluster_assignment, centroid_indices = RCKmeans(symm_matrix)

        elif method == "AutoGraph":
            # not reliable
            symm_matrix = convert_tril_to_symm(lower_triangle_values)
            cluster_assignment, centroid_indices = AutoGraph(symm_matrix)

        else:
            raise NotImplementedError(f"{method} clustering is not implemented yet.")

        # cluster_assignment: ex. [0,1,0,0,2,..]
        # centroid_indices: ex. [10,5,..] i.e. centroids of clusters 0 and 1 are 10 and 5, respectively.

        if cluster_assignment is not None and centroid_indices is not None:
            cluster_raw_data = defaultdict(list)
            for conf_idx, cluster_idx in enumerate(cluster_assignment):
                cluster_raw_data[cluster_idx].append(conf_idx)
            cluster_list = []
            for i, k in enumerate(sorted(cluster_raw_data.keys())):
                energies = [
                    self.confs[conf_idx].props["E_rel(kcal/mol)"]
                    for conf_idx in cluster_raw_data[k]
                ]
                mean_energy = np.mean(energies)
                median_energy = np.median(energies)
                q75, q25 = np.percentile(energies, [75, 25])
                iqr_energy = q75 - q25  # interquartile range (IQR)
                cluster_list.append(
                    {
                        "confs": cluster_raw_data[k],
                        "centroid": centroid_indices[i],  # conformer index
                        "size": len(cluster_raw_data[k]),
                        "mean_energy": mean_energy,
                        "median_energy": median_energy,
                        "iqr_energy": iqr_energy,
                    }
                )
            # sort cluster index
            if sort == "size":
                cluster_list = sorted(
                    cluster_list, key=lambda x: x["size"], reverse=True
                )

            elif sort == "energy":
                cluster_list = sorted(
                    cluster_list, key=lambda x: x["median_energy"], reverse=False
                )

            else:
                raise NotImplementedError(f"{sort} is not implemented yet.")

            for cluster_idx, cluster_dict in enumerate(cluster_list, start=1):
                for conf_idx in cluster_dict["confs"]:
                    if conf_idx == cluster_dict["centroid"]:
                        self.confs[conf_idx].props.update(
                            {
                                "cluster": cluster_idx,
                                "cluster_mean_energy": cluster_dict["mean_energy"],
                                "cluster_median_energy": cluster_dict["median_energy"],
                                "cluster_IQR_energy": cluster_dict["iqr_energy"],
                                "cluster_size": cluster_dict["size"],
                                "cluster_centroid": True,
                            }
                        )
                    else:
                        self.confs[conf_idx].props.update(
                            {
                                "cluster": cluster_idx,
                                "cluster_mean_energy": cluster_dict["mean_energy"],
                                "cluster_median_energy": cluster_dict["median_energy"],
                                "cluster_IQR_energy": cluster_dict["iqr_energy"],
                                "cluster_size": cluster_dict["size"],
                                "cluster_centroid": False,
                            }
                        )
        return self

    def drop_confs(
        self,
        stereo_flipped: bool = True,
        unconverged: bool = True,
        similar: bool | None = None,
        similar_rmsd: float = 0.3,
        cluster: bool | None = None,
        k: int | None = None,
        window: float | None = None,
        **kwargs,
    ) -> Self:
        """Drop conformers that meet some condition(s).

        Args:
            stereo_flipped (bool): drop conformers whose R/S and cis/trans stereo is unintentionally flipped.
                For example, a trans double bond in a macrocyle can end up with both trans
                and cis isomers in the final optimized conformers.
            unconverged (bool): drop unconverged conformers. see `Converged` property.
            similar (bool, optional): drop similar conformers. see `similar_rmsd`.
            similar_rmsd (float): RMSD (A) below `similar_rmsd` is regarded similar (default: 0.3)
            cluster (bool, optional): drop all except for the lowest energy conformer in each cluster.
            k (int, optional): drop all except for `k` lowest energy conformers.
            window (float, optional): drop all except for conformers within `window` of relative energy.

        Examples:
            To drop similar conformers within rmsd of 0.5 A
            >>> mol.drop_confs(similar=True, similar_rmsd=0.5)

            To drop conformers beyond 5 kcal/mol
            >>> mol.drop_confs(window=5.0)

        Returns:
            Self: modified self.
        """

        verbose = kwargs.get("verbose", False)

        reasons = [
            f"stereo flipped",
            f"unconverged",
            f"similar({similar_rmsd})",
            f"cluster(non-centroid)",
            f"k and/or energy window",
        ]

        w = max([len(s) for s in reasons])

        if stereo_flipped and self.num_confs > 0:
            mask = [
                Chem.MolToSmiles(Chem.RemoveHs(_.rdmol)) == self.smiles
                for _ in self.confs
            ]
            self.confs = list(itertools.compress(self.confs, mask))
            if verbose:
                logger.info(
                    f"drop_confs() {mask.count(False):3d} {reasons[0]:<{w}} -> {self.num_confs}"
                )

        if unconverged and self.num_confs > 0:
            mask = [
                _.props["Converged"] if "Converged" in _.props else True
                for _ in self.confs
            ]
            self.confs = list(itertools.compress(self.confs, mask))
            if verbose:
                logger.info(
                    f"drop_confs() {mask.count(False):3d} {reasons[1]:<{w}} -> {self.num_confs}"
                )

        if similar and self.num_confs > 1:
            # it is observed that there are essentially identical conformers
            # such as 180-degree ring rotation and there is not minor conformational variations
            # in the RDKit ETKDG generated conformers.

            if not self.is_confs_aligned:
                self = self.align_confs()

            # symmetry-aware RMSD calculation using spyrmsd
            lower_triangle_values = []
            for i in range(1, self.num_confs):
                mol_i = spyrmsd_Molecule.from_rdkit(self.confs[i].rdmol)
                mol_js = [
                    spyrmsd_Molecule.from_rdkit(self.confs[j].rdmol) for j in range(i)
                ]
                rmsd_list = spyrmsd_wrapper(
                    mol_i, mol_js, symmetry=True, minimize=False
                )
                # it just calculates rmsd without aligning (minimize=False)
                lower_triangle_values += rmsd_list
            symm_matrix = convert_tril_to_symm(lower_triangle_values)
            cluster_assignment, centroid_indices = QT(symm_matrix, similar_rmsd)
            mask = [
                conf_idx in centroid_indices for conf_idx, conf in enumerate(self.confs)
            ]
            # keep the centroid of clusters and drop others
            self.confs = list(itertools.compress(self.confs, mask))
            if verbose:
                logger.info(
                    f"drop_confs() {mask.count(False):3d} {reasons[2]:<{w}} -> {self.num_confs}"
                )

            # note: it will retain the conformers with lower index
            # so, it should be sorted before dropping
            # obj = obj.sort_confs()
            # mask = []
            # retained_confs = []
            # for conf_i in obj.confs:
            #     is_dissimilar = True
            #     for conf_j_rdmol_noH in retained_confs:
            #         # symmetry-aware alignment / removing Hs speeds up the calculation
            #         rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(conf_i.rdmol), conf_j_rdmol_noH)
            #         if rmsd < similar_rmsd:
            #             is_dissimilar = False
            #             break
            #     mask.append(is_dissimilar)
            #     if is_dissimilar:
            #         retained_confs.append(Chem.RemoveHs(conf_i.rdmol)) # store a copy of H-removed rdmol
            # obj.confs = list(itertools.compress(obj.confs, mask))

        if cluster and self.num_confs > 1:
            # drop non-centroid cluster member(s)
            # cluster_confs() sets `cluster_centroid`; `centroid` kept for older data
            mask = [
                _.props.get("cluster_centroid", _.props.get("centroid", True))
                for _ in self.confs
            ]
            self.confs = list(itertools.compress(self.confs, mask))
            if verbose:
                logger.info(
                    f"drop_confs() {mask.count(False):3d} {reasons[3]:<{w}} -> {self.num_confs}"
                )

        if (k or window) and self.num_confs > 0:
            # confs must be sorted by energies
            if not all(["E_rel(kcal/mol)" in _.props for _ in self.confs]):
                self = self.sort_confs(**kwargs)
            if k:
                mask_k = [i < k for i, _ in enumerate(self.confs)]
            else:
                mask_k = [
                    True,
                ] * self.num_confs
            if window:
                mask_window = [
                    (
                        _.props["E_rel(kcal/mol)"] < window
                        if "E_rel(kcal/mol)" in _.props
                        else True
                    )
                    for _ in self.confs
                ]
            else:
                mask_window = [
                    True,
                ] * self.num_confs
            # retain conformer(s) that satisfy both k and window conditions
            mask = [(x and y) for (x, y) in zip(mask_k, mask_window)]
            self.confs = list(itertools.compress(self.confs, mask))
            if verbose:
                logger.info(
                    f"drop_confs() {mask.count(False):3d} {reasons[4]:<{w}} -> {self.num_confs}"
                )

        return self

    def compute(self, **kwargs) -> Self:
        """Change settings for parallel computing.

        Args:
            max_workers (int): max number of workers.
            chunksize (int): chunksize of splitted workload.
            progress (bool): whether to show progress bar.

        Returns:
            Self: modified self.
        """
        self.max_workers = kwargs.get("max_workers", self.max_workers)
        self.chunksize = kwargs.get("chunksize", self.chunksize)
        self.progress = kwargs.get("progress", self.progress)

        return self

    def calculate_torsion_energies(
        self,
        calculator: str | Callable = "MMFF94",
        torsion_angle_idx: int | None = None,
        simplify: bool = True,
        fmax: float = 0.05,
        interval: float = 20.0,
        use_converged_only: bool = True,
        batchsize_atoms: int = 0,
        water: str | None = None,
    ) -> Self:
        """Calculates potential energy profiles for each torsion angle using ASE optimizer.

        It uses the first conformer as a reference.

        Args:
            calculator (str | Callable): 'MMFF', 'UFF', 'xTB' or ASE calculator.
            torsion_angle_idx (int | None): torsion index to calculate. Defaults to None (all).
            simplify (bool, optional): whether to use fragment surrogate. Defaults to True.
            fmax (float, optional): fmax of ASE optimizer. Defaults to 0.05.
            interval (float, optional): interval of torsion angles in degree. Defaults to 15.0.
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
            Self: modified self.
        """
        assert self.num_confs > 0, (
            "calculate_torsion_energies() requires at least one conformer"
        )

        ref_conf = self.confs[0].copy()

        if batchsize_atoms >= ref_conf.natoms:
            assert not isinstance(calculator, str), (
                "Batch optimizer is required for batch calculation"
            )

        ref_conf = ref_conf.calculate_torsion_energies(
            calculator,
            torsion_angle_idx,
            simplify,
            fmax,
            interval,
            use_converged_only,
            water=water,
            batchsize_atoms=batchsize_atoms,
        )

        self.props["torsion"] = ref_conf.props["torsion"]

        return self

    def calculate_sp_torsion_energies(
        self,
        calculator: str | Callable = "MMFF94",
        torsion_angle_idx: int | None = None,
        simplify: bool = True,
        interval: float = 20.0,
        water: str | None = None,
        batchsize_atoms: int = 0,
    ) -> Self:
        """Calculates potential energy profiles for each torsion angle using ASE optimizer.

        It uses the first conformer as a reference.

        Args:
            calculator (str | Callable): 'MMFF', 'UFF', 'xTB' or ASE calculator.
            torsion_angle_idx (int | None): torsion index to calculate. Defaults to None (all).
            simplify (bool, optional): whether to use fragment surrogate. Defaults to True.
            interval (float, optional): interval of torsion angles in degree. Defaults to 15.0.
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
            Self: modified self.
        """
        assert self.num_confs > 0, (
            "calculate_sp_torsion_energies() requires at least one conformer"
        )

        ref_conf = self.confs[0].copy()

        if batchsize_atoms >= ref_conf.natoms:
            assert not isinstance(calculator, str), (
                "Batch singlepoint is required for batch calculation"
            )

        ref_conf = ref_conf.calculate_sp_torsion_energies(
            calculator,
            torsion_angle_idx,
            simplify,
            interval,
            water=water,
            batchsize_atoms=batchsize_atoms,
        )

        self.props["torsion"] = ref_conf.props["torsion"]

        return self

    def draw(
        self,
        coordgen: bool = False,
        rotate: bool = False,
        axis: str = "z",
        degree: float = 0.0,
    ) -> Self:
        """Draw molecule in 2D.

        Args:
            coordgen (bool, optional): whether to use `coordgen`. Defaults to False.
            rotate (bool, optional): whether to rotate drawing. Defaults to False.
            axis (str, optional): axis for rotation. Defaults to 'z'.
            degree (float, optional): degree for rotation. Defaults to 0.0.

        Returns:
            Self.
        """
        if rotate and axis.lower() not in ("x", "y", "z"):
            raise ValueError(f"draw() axis must be 'x', 'y', or 'z', got {axis!r}")

        rdDepictor.SetPreferCoordGen(coordgen)
        rdDepictor.Compute2DCoords(self.rdmol)

        if rotate:
            rad = (np.pi / 180.0) * degree
            c = np.cos(rad)
            s = np.sin(rad)
            if axis.lower() == "x":
                rotmat = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, c, -s, 0.0],
                        [0.0, s, c, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            elif axis.lower() == "y":
                rotmat = np.array(
                    [
                        [c, 0.0, s, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-s, 0.0, c, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            elif axis.lower() == "z":
                rotmat = np.array(
                    [
                        [c, -s, 0.0, 0.0],
                        [s, c, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            rdMolTransforms.TransformConformer(self.rdmol.GetConformer(), rotmat)

        return self

    ##################################################
    ### Endpoint Functions (returns some values)
    ##################################################

    def is_nnp_ready(self, model: str = "aimnet2") -> bool:
        """Check if a particular neural network model is applicable to current molecule.

        Args:
            model (str): neural network models: `ANI-2x`, `ANI-2xt`, `AIMNET`

        Raises:
            ValueError: if model is not supported.

        Returns:
            bool: True if applicable.
        """
        if model.lower() in ["ani-2x", "ani-2xt", "ani2x", "ani2xt"]:
            if self.props["charge"] != 0:
                return False
            # H, C, N, O, F, S, Cl
            atomic_numbers = [1, 6, 7, 8, 9, 16, 17]

        elif model.lower() in ["aimnet", "aimnet2"]:
            # H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
            atomic_numbers = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]

        else:
            raise ValueError("nnp_ready() supports ANI-2x, ANI-2xt, AIMNet, or AIMNet2")

        if all([a.GetAtomicNum() in atomic_numbers for a in self.rdmol.GetAtoms()]):
            return True
        else:
            return False

    def get_SASA(self) -> dict:
        """Get Solvent Accessible Surface Area"""
        results = defaultdict(list)
        for conf in self.confs:
            for k, v in conf.SASA.items():
                results[k].append(v)
        return results

    def get_torsion_angle_atoms(self, strict: bool = True) -> list[tuple]:
        """Determine torsion/dihedral angle atoms (i-j-k-l) and rotating group for each rotatable bond (j-k).

        Args:
            strict (bool): whether to exclude amide/imide/ester/acid bonds.

        Returns:
            [(i, j, k, l), ...]
        """
        return [d[:4] for d in get_torsion_angle_atom_indices(self.rdmol, strict)]

    def get_similarity(self, other: Self) -> float:
        """Returns Tanimoto similarity with other Mol object.

        Args:
            other (Mol): other Mol object.

        Raises:
            TypeError: if `other` is not Mol object type.

        Returns:
            float: Tanimoto similarity.
        """
        assert isinstance(other, Mol), "similarity() Error: invalid Mol object"

        if not self.fp:
            self.fp = self.MFP2.GetFingerprint(self.rdmol)

        if not other.fp:
            other.fp = other.MFP2.GetFingerprint(other.rdmol)

        return DataStructs.TanimotoSimilarity(self.fp, other.fp)

    def is_similar(self, other: Self, threshold: float) -> bool:
        """Check if other molecule is similar within Tanimoto similarity threshold.

        Args:
            other (Mol): other Mol object to compare with.
            threshold (float): Tanimoto similarity threshold.

        Returns:
            bool: True if similar.
        """
        return self.get_similarity(other) >= threshold

    def has_substr(self, substr: str) -> bool:
        """Determine if the molecule has the substructure match.

        Args:
            pattern (str): SMARTS or SMILES.

        Returns:
            bool: True if matches.
        """
        query = Chem.MolFromSmarts(substr)
        if query is None:
            raise ValueError(f"has_substr() invalid SMARTS/SMILES {substr!r}")
        return self.rdmol.HasSubstructMatch(query)

    def is_matching(self, terms: str | Path, invert: bool = False) -> bool:
        """Determines if the molecule matches the predefined substructure and/or descriptor ranges.

        invert | terms(~ or !) | effect
        ------ | ------------- | -------------
        True   |     ~         | No inversion
        True   |               | Inversion
        False  |     ~         | Inversion
        False  |               | No inversion

        Args:
            terms (str | Path):
                substructure SMARTS expression or a path to predefined descriptor ranges.
            invert (bool, optional): whether to invert the result. Defaults to False.

        Returns:
            bool: True if matches.
        """
        if isinstance(terms, Path):
            path = terms.as_posix()

        elif isinstance(terms, str):
            if terms.startswith("~") or terms.startswith("!"):
                terms = terms[1:]  # strip only the leading inversion marker
                invert = invert ^ True
            path = Path(terms)  # test if terms points to a xml file
            try:
                is_file = path.is_file()
            except OSError:  # e.g. ENAMETOOLONG for a long SMARTS string
                is_file = False
            if not is_file:
                path = get_predefined_xml(terms)
        else:
            logger.warning(
                "is_matching(): terms must be a str or Path; predefined terms:\n%s",
                list_predefined_xml(),
            )
            return False

        (lterms, combine) = parse_xml(path)
        mask = []
        for name, smarts, lb, ub in lterms:
            if smarts:
                query = Chem.MolFromSmarts(smarts)
                if len(self.rdmol.GetSubstructMatches(query)) > 0:
                    mask.append(True)
                else:
                    mask.append(False)
            else:  # descriptor lower and upper bounds
                if name not in self.props:
                    val = rd_descriptor_f[name](self.rdmol)
                    self.props.update({name: val})
                else:
                    val = self.props[name]
                # return if lower and upper boundaries are satisfied
                # None means no bound; 0.0 is a valid bound
                if (lb is None or val >= lb) and (ub is None or val <= ub):
                    mask.append(True)
                else:
                    mask.append(False)
            if combine.lower() == "or" and any(mask):
                # early termination if any term is satisfied
                return invert ^ True  # XOR(^) inverts only if invert is True

        if combine.lower() == "and" and all(mask):
            return invert ^ True

        return invert ^ False

    def to_geometry(self, index: int = 0) -> str:
        """Returns geometry input of a conformer for psi4 or other quantum chemistry software.

        Args:
            index (int, optional): conformer index. Defaults to 0.

        Returns:
            str: geometry input (atom symbol and X, Y, Z in Angstrom per line).
        """
        return rwio.to_geometry(self, index=index)

    def to_xyz(self, path: str | Path | None = None) -> str:
        """Returns multi-frame XYZ strings of all conformers.

        Args:
            path (str | Path | None, optional): also write to this file. Defaults to None.

        Returns:
            str: XYZ formatted strings (the comment line of each frame is the conformer name).
        """
        return rwio.to_xyz(self, path=path)

    def to_turbomole(self, index: int = 0, bohr: bool = False) -> str:
        """Returns TURBOMOLE coord file formatted strings of a conformer.

        Args:
            index (int, optional): conformer index. Defaults to 0.
            bohr (bool, optional): whether to use Bohr units. Defaults to False (Angstrom).

        Returns:
            str: TURBOMOLE coord formatted file.
        """
        return rwio.to_turbomole(self, index=index, bohr=bohr)

    def to_sdf(
        self, confs: bool = False, props: bool = True, path: str | Path | None = None
    ) -> str | None:
        """Returns strings of SDF output.

        Args:
            confs (bool, optional): whether to write 3D conformers (with molecule and
                conformer properties) instead of the 2D molecule. Defaults to False.
            props (bool, optional): whether to include properties. Defaults to True.
            path (str | Path | None, optional): write to this .sdf (.sdf.gz) file instead
                of returning the strings. Defaults to None.

        Returns:
            str | None: strings of SDF output if `path` is None.
        """
        return rwio.to_sdf(self, path=path, confs=confs, props=props)

    def to_png(
        self,
        width: int = 300,
        height: int = 300,
        legend: str = "",
        atom_index: bool = False,
        highlight_atoms: list[int] | None = None,
        highlight_bonds: list[int] | None = None,
        redraw: bool = False,
        coordgen: bool = False,
        trim: bool = True,
        path: str | Path | None = None,
    ) -> Image.Image:
        """Draw 2D molecule in PNG format.

        Args:
            width (int, optional): width. Defaults to 300.
            height (int, optional): height. Defaults to 300.
            legend (str, optional): legend. Defaults to ''.
            atom_index (bool, optional): whether to show atom index. Defaults to False.
            highlight_atoms (list[int] | None, optional): atom(s) to highlight. Defaults to None.
            highlight_bonds (list[int] | None, optional): bond(s) to highlight. Defaults to None.
            redraw (bool, optional): whether to redraw. Defaults to False.
            coordgen (bool, optional): whether to use coordgen. Defaults to False.
            trim (bool, optional): whether to trim white margins. Default to True.

        Returns:
            Image.Image: output PIL Image object.
        """

        return rwio.to_png(
            self,
            path=path,
            width=width,
            height=height,
            legend=legend,
            atom_index=atom_index,
            highlight_atoms=highlight_atoms,
            highlight_bonds=highlight_bonds,
            redraw=redraw,
            coordgen=coordgen,
            trim=trim,
        )

    def to_svg(
        self,
        width: int = 300,
        height: int = 300,
        legend: str = "",
        atom_index: bool = False,
        highlight_atoms: list[int] | None = None,
        highlight_bonds: list[int] | None = None,
        redraw: bool = False,
        coordgen: bool = False,
        optimize: bool = True,
        path: str | Path | None = None,
    ) -> str:
        """Draw 2D molecule in SVG format.

        Examples:
            For Jupyternotebook, wrap the output with SVG:

            >>> from IPython.display import SVG
            >>> SVG(libr[0].to_svg())

        Args:
            width (int, optional): width. Defaults to 300.
            height (int, optional): height. Defaults to 300.
            legend (str, optional): legend. Defaults to ''.
            atom_index (bool, optional): whether to show atom index. Defaults to False.
            highlight_atoms (list[int] | None, optional): atom(s) to highlight. Defaults to None.
            highlight_bonds (list[int] | None, optional): bond(s) to highlight. Defaults to None.
            redraw (bool, optional): whether to redraw. Defaults to False.
            coordgen (bool, optional): whether to use coordgen. Defaults to False.
            optimize (bool, optional): whether to optimize SVG string. Defaults to True.

        Returns:
            str: SVG string
        """
        return rwio.to_svg(
            self,
            path=path,
            width=width,
            height=height,
            legend=legend,
            atom_index=atom_index,
            highlight_atoms=highlight_atoms,
            highlight_bonds=highlight_bonds,
            redraw=redraw,
            coordgen=coordgen,
            optimize=optimize,
        )

    def to_df(
        self, name: str = "name", smiles: str = "smiles", confs: bool = False
    ) -> "pd.DataFrame":
        """Returns a pandas DataFrame of the molecule (or its conformers) and properties.

        Args:
            name (str, optional): column name for name. Defaults to 'name'.
            smiles (str, optional): column name for SMILES. Defaults to 'smiles'.
            confs (bool, optional): one row per conformer with conformer properties
                instead of one row with molecule properties. Defaults to False.

        Returns:
            pd.DataFrame: pandas DataFrame.
        """
        return rwio.to_df(self, name=name, smiles=smiles, confs=confs)

    def to_csv(self, path: str | Path, confs: bool = False, decimals: int = 3) -> None:
        """Writes the molecule (or its conformers) and properties to a .csv file.

        Args:
            path (str | Path): output filename or path.
            confs (bool, optional): one row per conformer. Defaults to False.
            decimals (int, optional): decimal places for float numbers. Defaults to 3.
        """
        rwio.to_csv(self, path, confs=confs, decimals=decimals)

    def to_plot_data_torsion_angle_vs_energy(self, idx: int | None = None) -> dict:
        """Returns plot data for torsion angle vs energy.

        For seaborn plot,

            ```py
            data = self.props['torsion'][torsion_angle_idx]
            df = pd.DataFrame({ax: data[ax] for ax in ['angle', 'E_rel(kcal/mol)']})

            plt.figure(**kwargs)
            plt.clf()  # Clear the current figure to prevent overlapping plots

            sns.set_theme()
            sns.color_palette("tab10")
            sns.set_style("whitegrid")

            if len(df['angle']) == len(df['angle'].drop_duplicates()):
                g = sns.lineplot(x="angle",
                                y="E_rel(kcal/mol)",
                                data=df,
                                marker='o',
                                markersize=10)
            else:
                g = sns.lineplot(x="angle",
                                y="E_rel(kcal/mol)",
                                data=df,
                                errorbar=('ci', 95),
                                err_style='bars',
                                marker='o',
                                markersize=10)
            g.xaxis.set_major_locator(ticker.MultipleLocator(30))
            g.xaxis.set_major_formatter(ticker.ScalarFormatter())
            if df["E_rel(kcal/mol)"].max() > upper_limit:
                g.set(title=self.name,
                    xlabel='Dihedral Angle (degree)',
                    ylabel='Relative Energy (Kcal/mol)',
                    xlim=(-190, 190),
                    ylim=(-1.5, upper_limit))
            elif df["E_rel(kcal/mol)"].max() < zoomin_limit:
                g.set(title=self.name,
                    xlabel='Dihedral Angle (degree)',
                    ylabel='Relative Energy (Kcal/mol)',
                    xlim=(-190, 190),
                    ylim=(-1.5, zoomin_limit))
            else:
                g.set(title=self.name,
                    xlabel='Dihedral Angle (degree)',
                    ylabel='Relative Energy (Kcal/mol)',
                    xlim=(-190, 190),)
            g.tick_params(axis='x', rotation=30)

            if svg:
                buf = StringIO()
                plt.savefig(buf, format='svg', bbox_inches='tight')
                plt.close() # prevents duplicate plot outputs in Jupyter Notebook
                svg_string = buf.getvalue()
                # optimize SVG string
                scour_options = {
                    'strip_comments': True,
                    'strip_ids': True,
                    'shorten_ids': True,
                    'compact_paths': True,
                    'indent_type': 'none',
                }
                svg_string = scourString(svg_string, options=scour_options)

                return svg_string

            else:
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close() # prevents duplicate plot outputs in Jupyter Notebook
                buf.seek(0)
                img = Image.open(buf)
                plt.imshow(img)
                plt.axis('off') # Optional: remove axes
                plt.show()
            ```

            Args:
                idx (int, optional) - 0-based torsion angle index.
                    Defaults to None (all torsion angles).

            Returns:
                {'idx': list, 'angle': list, 'E_rel(kcal/mol)': list}

        """
        return get_torsion_plot_data(self.props["torsion"], idx)


    def dumps(self, key: str = "", decimals: int = 2) -> str:
        """Returns JSON dumps of properties.

        Args:
            key (str | None): key for a subset of properties. Defaults to None.
            decimals (int, optional): decimal places for float numbers. Defaults to 2.

        Returns:
            str: JSON dumps.
        """
        props = recursive_round(self.props, decimals)

        if key:
            return json.dumps({key: props[key]})

        return json.dumps(props)

    def serialize(self, decimals: int = 3) -> str:
        """Serialize information necessary to rebuild a Mol object.

        Args:
            decimals (int, optional): number of decimal places for float data type. Defaults to 2.

        Returns:
            str: serialized string for json.loads()
        """
        return serialize(
            {
                "name": self.name,
                "smiles": self.smiles,
                "molblock": self.molblock,
                "InChIKey": self.InChIKey,
                "props": recursive_round(self.props, decimals),
                "confs": [conf.serialize(decimals=decimals) for conf in self.confs],
            }
        )

    ##################################################
    ### Report Functions (returns None)
    ##################################################

    def report_stereo(self) -> None:
        """Report stereochemistry information for debug"""
        num_chiral_centers = rdMolDescriptors.CalcNumAtomStereoCenters(self.rdmol)
        # Returns the total number of atomic stereocenters (specified and unspecified)
        num_unspecified_chiral_centers = (
            rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(self.rdmol)
        )
        print(
            f"chiral centers = unspecified {num_unspecified_chiral_centers} / total {num_chiral_centers}"
        )
        print(f"stereogenic double bonds =")
        for element in Chem.FindPotentialStereo(self.rdmol):
            # element.type= Atom_Octahedral, Atom_SquarePlanar, Atom_Tetrahedral,
            #               Atom_TrigonalBipyramidal,
            #               Bond_Atropisomer, Bond_Cumulene_Even, Bond_Double,
            #               Unspecified
            if element.type == Chem.StereoType.Bond_Double:
                bond = self.rdmol.GetBondWithIdx(element.centeredOn)
                atom1 = bond.GetBeginAtom().GetSymbol()
                atom2 = bond.GetEndAtom().GetSymbol()
                is_nitrogen = atom1 == "N" or atom2 == "N"
                print(f"  {element.type} bond: {element.centeredOn}", end=" ")
                print(f"ring: {bond.IsInRing()} N: {is_nitrogen}", end=" ")
            elif element.type == Chem.StereoType.Atom_Tetrahedral:
                print(f"  {element.type} atom: {element.centeredOn}", end=" ")
                print(f"atoms {list(element.controllingAtoms)}", end=" ")
            print(f"{element.specified} {element.descriptor}")  # type: Chem.StereoDescriptor

    def report_props(self) -> None:
        """Report properties"""
        if self.props:
            print(f"Properties({len(self.props)}):")
            fixed_width = max([len(k) for k in self.props]) + 4
            for k, v in self.props.items():
                while len(k) <= fixed_width:
                    k = k + " "
                print(f"  {k} {v}")
        else:
            print(f"Properties: None")
