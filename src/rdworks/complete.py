from rdworks import Mol, MolLibr
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)


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

    rdmols = list(
        EnumerateStereoisomers(
            mol.rdmol,
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


def complete_tautomers(mol: Mol, **kwargs) -> MolLibr:
    """Returns a library of enumerated tautomers.

    Args:
        mol (Mol): input molecule.

    Returns:
        MolLibr: a library of enumerated tautomers.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetRemoveSp3Stereo(False)
    enumerator.SetRemoveBondStereo(False)  # Don't remove existing stereo
    tautomers = list(enumerator.Enumerate(mol.rdmol))

    all_stereoisomers = []

    for tautomer in tautomers:
        # Find unspecified double bonds that could have E/Z stereo
        unspecified_bonds = _unspecified_double_bonds(tautomer)

        if unspecified_bonds:
            # Enumerate all possible stereoisomers
            opts = StereoEnumerationOptions(onlyUnassigned=True, maxIsomers=50)
            stereoisomers = list(EnumerateStereoisomers(tautomer, options=opts))
            all_stereoisomers.extend(stereoisomers)
        else:
            all_stereoisomers.append(tautomer)

    # rename() adds serial numbers only when more than one molecule remains;
    # a single result keeps the input name
    return (
        MolLibr(all_stereoisomers)
        .unique()
        .rename(mol.name, sep=".")
        .compute(**kwargs)
    )
