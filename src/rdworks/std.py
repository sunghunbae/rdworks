from rdkit import Chem
from rdkit.Chem import rdDepictor, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize


def generate_inchi_key(rdmol: Chem.Mol) -> str:
    """Generate InChIKey.

    Note:
        - An InChIKey is a 27-character string consisting of three parts:
            - 14 characters: Derived from the connectivity layer of the InChI.
            - Hyphen (-): Separates the first two blocks.
            - 9 characters: Derived from the remaining InChI layers.
            - A hyphen (-): Separates the second and third blocks.
            - A final checksum character: Ensures the integrity of the key.

    Args:
        rdmol (Chem.Mol): input molecule

    Returns:
        str: 27-character InChIKey, or an empty string if InChI cannot be
            generated (e.g. molecules with dummy atoms `*`).
    """
    InChI = Chem.MolToInchi(rdmol)
    if not InChI:
        return ""
    InChIKey = inchi.InchiToInchiKey(InChI)

    return InChIKey or ""


def desalt_smiles(smiles: str) -> tuple[str, Chem.Mol]:
    """Remove salt(s) from SMILES by keeping the largest fragment.

    The whole SMILES is parsed first, so ring-closure bonds across `.` (e.g. `C1.C1`)
    are handled correctly, and the returned SMILES is canonical, i.e. identical to
    the SMILES of the same molecule given without salts.

    Args:
        smiles (str): SMILES.

    Raises:
        ValueError: if `smiles` is invalid.

    Returns:
        (desalted canonical SMILES, desalted Chem.Mol)
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"desalt_smiles() Error: invalid SMILES {smiles}")

    # largest fragment by number of (heavy) atoms; ties keep the first fragment
    fragments = Chem.GetMolFrags(rdmol, asMols=True)
    desalted_rdmol = max(fragments, key=lambda frag: frag.GetNumAtoms())

    return (Chem.MolToSmiles(desalted_rdmol), desalted_rdmol)


def standardize_smiles(smiles: str) -> str:
    """Returns standardized SMILES string.

    The rdMolStandardize.StandardizeSmiles() function performs the following steps:

    1. mol = Chem.MolFromSmiles(sm)
    1. Chem.SanitizeMol(mol)
    1. mol = Chem.RemoveHs(mol)
    1. mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
    1. mol = rdMolStandardize.Normalize(mol)
    1. mol = rdMolStandardize.Reionize(mol)
    1. Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    1. Chem.MolToSmiles(mol)

    See [rdkit notebook](https://github.com/rdkit/rdkit/blob/master/Docs/Notebooks/MolStandardize.ipynb) and
    [greg's notebook](https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/Standardization%20and%20Validation%20with%20the%20RDKit.ipynb),
    and [youtube video](https://www.youtube.com/watch?v=eWTApNX8dJQ).

    Args:
        smiles (str): input SMILES string.

    Returns:
        str: standardized SMILES string.


    """
    return rdMolStandardize.StandardizeSmiles(smiles)


def standardize(rdmol: str | Chem.Mol) -> Chem.Mol:
    """Returns standardized rdkit.Chem.Mol object.

    Args:
        rdmol (str | Chem.Mol): input SMILES string or molecule (not modified).

    Raises:
        TypeError: if `rdmol` is neither a SMILES string nor a Chem.Mol.
        ValueError: if `rdmol` is an invalid SMILES.

    Returns:
        Chem.Mol: standardized rdkit.Chem.Mol object.
    """
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as
    if isinstance(rdmol, Chem.Mol):
        mol = Chem.Mol(rdmol)  # make a copy to avoid modifying the original
    elif isinstance(rdmol, str):
        mol = Chem.MolFromSmiles(rdmol)
        if mol is None:
            raise ValueError(f"standardize() Error: invalid SMILES {rdmol}")
    else:
        raise TypeError(
            f"standardize() expects SMILES or Chem.Mol, got {type(rdmol).__name__}"
        )

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol)

    # if many fragments, get the "parent" (the actual mol we are interested in)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

    # try to neutralize molecule
    uncharger = (
        rdMolStandardize.Uncharger()
    )  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

    # note that no attempt is made at reionization at this step
    # nor at ionization at some pH (rdkit has no pKa caculator)
    # the main aim to to represent all molecules from different sources
    # in a (single) standard way, for use in ML, catalogue, etc.

    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def neutralize_atoms(rdmol: Chem.Mol) -> Chem.Mol:
    """Neutralizes atoms.

    It is adapted from Noel O'Boyle's nocharge code:
    [rdkit cookbook](https://www.rdkit.org/docs/Cookbook.html),
    [no charge](https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html).
    It is a neutralization by atom approach and neutralizes atoms with a +1 or -1 charge
    by removing or adding hydrogen where possible. The SMARTS pattern checks for a hydrogen
    in +1 charged atoms and checks for no neighbors with a negative charge (for +1 atoms)
    and no neighbors with a positive charge (for -1 atoms), this is to avoid altering molecules
    with charge separation (e.g., nitro groups).

    The neutralize_atoms() function differs from the rdMolStandardize.Uncharger behavior.
    See the [MolVS documentation for Uncharger](https://molvs.readthedocs.io/en/latest/api.html#molvs-charge).

    > This class uncharges molecules by adding and/or removing hydrogens.
    In cases where there is a positive charge that is not neutralizable,
    any corresponding negative charge is also preserved. As an example,
    rdMolStandardize.Uncharger will not change charges on C[N+](C)(C)CCC([O-])=O,
    as there is a positive charge that is not neutralizable. In contrast, the neutralize_atoms()
    function will attempt to neutralize any atoms it can (in this case to C[N+](C)(C)CCC(=O)O).
    That is, neutralize_atoms() ignores the overall charge on the molecule, and attempts to neutralize
    charges even if the neutralization introduces an overall formal charge on the molecule.

    Args:
        rdmol (Chem.Mol) : molecule (not to be modified).

    Returns:
        Chem.Mol: neutralized copy of molecule.
    """
    mol = Chem.Mol(rdmol)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]

    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()

    return mol


def clean_2d(
    rdmol: Chem.Mol,
    reset_isotope: bool = True,
    remove_H: bool = True,
) -> tuple[Chem.Mol, list[Chem.Mol]]:
    """Clean molecule for 2D depiction.

    Args:
        rdmol (Chem.Mol): molecule (not to be modified)
        reset_isotope (bool, optional): whether to reset isotope information. Defaults to True.
        remove_H (bool, optional): whether to remove explicit hydrogens. Defaults to True.

    Returns:
        (cleaned 2D copy of molecule, list of Chem.Mol each holding one 3D conformer
        of the input molecule with its original atoms, ready for `Conf()`).
        3D conformers are returned only if all hydrogens are explicit, as `Conf()`
        requires; 2D conformers and 3D input without hydrogens yield an empty list.
    """
    mol = Chem.Mol(rdmol)

    # A molecule constructed from SMILES has no conformer; 2D conformers are skipped.
    conformers = []
    has_all_H = rdmol.GetNumAtoms() == rdmol.GetNumAtoms(onlyExplicit=False)
    for rdConformer in rdmol.GetConformers():
        if rdConformer.Is3D() and has_all_H:
            conf_mol = Chem.Mol(rdmol)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(Chem.Conformer(rdConformer), assignId=True)
            conformers.append(conf_mol)

    if reset_isotope:
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)

    if remove_H:
        mol = Chem.RemoveHs(mol)

    rdDepictor.Compute2DCoords(mol)

    return (mol, conformers)
