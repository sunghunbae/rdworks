from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

if TYPE_CHECKING:
    # rdworks.mol imports this module, so Mol/MolLibr are imported lazily at runtime
    from rdworks import Mol, MolLibr


def correct_wedged_bonds(rdmol: Chem.Mol) -> None:
    """Corrects wedged bonds for chiral centers.
    Args:
        rdmol (Chem.Mol): input molecule.
    
    1. Clear all stereochemistry information
    2. Identify chiral centers and non-chiral atoms/bonds
    3. Reassign stereochemistry only where appropriate
    4. Return the modified molecule
    """
    chiral_centers = Chem.FindMolChiralCenters(rdmol, includeUnassigned=True)
    chiral_atom_idx = [atom_idx for atom_idx, chirality in chiral_centers]
    for bond in rdmol.GetBonds():
        ai = bond.GetBeginAtom()
        aj = bond.GetEndAtom()
        is_chiral = [ai.GetIdx() in chiral_atom_idx, aj.GetIdx() in chiral_atom_idx]
        if any(is_chiral):
            continue
        else:
            # clear atom chirality
            if not is_chiral[0]:
                ai.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if not is_chiral[1]:
                aj.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            # clear bond directions (wedge/dash) and bond stereo
            bond.SetBondDir(Chem.BondDir.NONE)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    # 3. Reassign stereochemistry only where appropriate
    Chem.AssignStereochemistry(rdmol, cleanIt=True, force=True)


def enumerate_stereoisomers(rdmol: Chem.Mol) -> list[Chem.Mol]:
    """Returns enumerated stereoisomers.

    Args:
        rdmol (Chem.Mol): input molecule.

    Returns:
        List[Chem.Mol]: a list of enumerated stereoisomers.
    """
    return list(
        EnumerateStereoisomers(
            rdmol,
            options=StereoEnumerationOptions(
                tryEmbedding=False,
                onlyUnassigned=True,
                maxIsomers=1024,
                rand=None,
                unique=True,
                onlyStereoGroups=False
            ),
        )
    )


def enumerate_ring_bond_stereoisomers(
    rdmol: Chem.Mol, ring_bond_stereo_info: list[tuple], override: bool = False
) -> list[Chem.Mol]:
    """Enumerates unspecified double bond stereochemistry (cis/trans).

    Small rings (< 8 atoms): 
        RDKit does NOT enumerate E/Z stereochemistry for double bonds 
        because the ring constraints make one configuration impossible
    
    Macrocycles (≥ 8 atoms): 
        RDKit CAN enumerate E/Z stereochemistry for double bonds 
        since both configurations are geometrically feasible

    Args:
        rdmol (Chem.Mol): input molecule.
        ring_bond_stereo_info (List[Tuple]): 
            ring_bond_stereo_info will be set when .remove_stereo() is called.
            bond_stereo_info = [(bond_idx, bond_stereo_descriptor), ..] where
            bond_stereo_descriptor is `Chem.StereoDescriptor.Bond_Cis` or
            `Chem.StereoDescriptor.Bond_Trans`, or `Chem.StereoDescriptor.NoValue`.
        override (bool, optional): _description_. Defaults to False.

    Returns:
        List[Chem.Mol]: list of enumerated stereoisomers.
    """
    isomers = []
    for bond_idx, bond_stereo_desc in ring_bond_stereo_info:
        if (bond_stereo_desc == Chem.StereoDescriptor.NoValue) or override:
            bond = rdmol.GetBondWithIdx(bond_idx)
            (a2, a3) = (bond.GetBeginAtom(), bond.GetEndAtom())
            a2_idx = a2.GetIdx()
            a3_idx = a3.GetIdx()
            a1_idx = sorted(
                [
                    (a.GetIdx(), a.GetAtomicNum())
                    for a in a2.GetNeighbors()
                    if a.GetIdx() != a3_idx
                ],
                key=lambda x: x[1],
                reverse=True,
            )[0][0]
            a4_idx = sorted(
                [
                    (a.GetIdx(), a.GetAtomicNum())
                    for a in a3.GetNeighbors()
                    if a.GetIdx() != a2_idx
                ],
                key=lambda x: x[1],
                reverse=True,
            )[0][0]
            bond.SetStereoAtoms(a1_idx, a4_idx)  # need to set reference atoms
            # cis
            bond.SetStereo(Chem.BondStereo.STEREOCIS)
            isomers.append(Chem.Mol(rdmol))
            # trans
            bond.SetStereo(Chem.BondStereo.STEREOTRANS)
            isomers.append(Chem.Mol(rdmol))
    return isomers


def can_have_stereo(bond: Chem.Bond) -> bool:
    """Check if a double bond can have E/Z stereochemistry.

    Uses RDKit's stereo perception, which correctly handles implicit H, lone pairs
    (e.g. imines C=N-R), identical substituents, and small-ring double bonds.
    """
    if bond.GetBondType() != Chem.BondType.DOUBLE:
        return False

    return any(
        si.type == Chem.StereoType.Bond_Double and si.centeredOn == bond.GetIdx()
        for si in Chem.FindPotentialStereo(bond.GetOwningMol())
    )


def _unspecified_double_bonds(rdmol: Chem.Mol) -> list[int]:
    """Returns indices of double bonds with potential but unassigned E/Z stereochemistry."""
    return [
        si.centeredOn
        for si in Chem.FindPotentialStereo(rdmol)
        if si.type == Chem.StereoType.Bond_Double
        and si.specified != Chem.StereoSpecified.Specified
    ]


def complete_stereoisomers(
    molecular_input: str | Chem.Mol | Mol,
    name: str | None = None,
    std: bool = False,
    override: bool = False,
    **kwargs,
) -> MolLibr:
    """Completes stereoisomers and returns a rdworks.MolLibr.

    Small rings (< 8 atoms): 
        RDKit does NOT enumerate E/Z stereochemistry for double bonds 
        because the ring constraints make one configuration impossible
    
    Macrocycles (≥ 8 atoms): 
        RDKit CAN enumerate E/Z stereochemistry for double bonds 
        since both configurations are geometrically feasible

    Args:
        molecular_input (Union[Mol, str, Chem.Mol]): input molecule.
        name (Optional[str], optional): name of the molecule. Defaults to None.
        std (bool, optional): whether to standardize the input. Only applies to SMILES or
            rdkit.Chem.Mol input. Defaults to False.
        override (bool, optional): whether to override input stereoisomers. Defaults to False.
        **kwargs: additional keyword arguments passed to MolLibr.compute().

    Raises:
        TypeError: if `molecular_input` is not rdworks.Mol, SMILES, or rdkit.Chem.Mol object.

    Returns:
        MolLibr: a library of complete stereoisomers.
    """
    from rdworks import Mol, MolLibr  # lazy import to avoid a circular import

    if isinstance(molecular_input, Mol):
        # work on a copy: rename() and remove_stereo() modify the Mol in place
        mol = molecular_input.copy()
        if name:
            mol = mol.rename(name)
    elif isinstance(molecular_input, (str, Chem.Mol)):
        mol = Mol(molecular_input, name or "", std)
    else:
        raise TypeError(
            "complete_stereoisomers() expects rdworks.Mol, SMILES or rdkit.Chem.Mol object"
        )

    if override:
        mol = mol.remove_stereo()

    rdmols = enumerate_stereoisomers(mol.rdmol)

    # if proper_wedged_bonds:
    #     for rdmol in rdmols:
    #         correct_wedged_bonds(rdmol)

    # ring_bond_stereo_info = mol.ring_bond_stereo_info  (must be read before remove_stereo)
    # if len(ring_bond_stereo_info) > 0:
    #     ring_cis_trans = []
    #     for rdmol in rdmols:
    #         ring_cis_trans += enumerate_ring_bond_stereoisomers(
    #             rdmol, ring_bond_stereo_info, override=override
    #         )
    #     if len(ring_cis_trans) > 0:
    #         rdmols = ring_cis_trans

    if len(rdmols) > 1:
        libr = MolLibr(rdmols).unique().rename(mol.name, sep=".").compute(**kwargs)
    else:
        libr = MolLibr(rdmols).rename(mol.name).compute(**kwargs)

    for _ in libr:
        _.props.update(mol.props)

    return libr
