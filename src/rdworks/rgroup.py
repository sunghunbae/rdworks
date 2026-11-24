"""
This module provides functions to create molecules from core and R-groups.

Attributes:
    most_common (List[str]): a list of the most common R-groups, lazily loaded from data/Rgroups.json.
    most_common_in_NP (List[str]): a list of the most common R-groups in natural products, lazily loaded from data/Rgroups.json.
"""

import json
from collections import defaultdict
from functools import lru_cache
from importlib.resources import files
from itertools import product
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from rdkit.Chem.rdchem import EditableMol

from rdworks.mol import Mol
from rdworks.mollibr import MolLibr
from rdworks.std import desalt_smiles


def fragmented(
    rdmol: Chem.Mol, min_atoms: int = 5, max_atoms: int = 17
) -> list[Chem.Mol]:
    """BRICS decompose and returns fragments"""
    dummy = Chem.MolFromSmiles("*")
    hydro = Chem.MolFromSmiles("[H]")
    frag_set = BRICS.BRICSDecompose(rdmol)
    lfrag_rdmol = []
    for frag_smi in frag_set:
        # if '.' in frag_smi :
        (_, frag_rdmol) = desalt_smiles(frag_smi)
        # replace dummy atom(s) with [H]
        frag_rdmol_H = AllChem.ReplaceSubstructs(frag_rdmol, dummy, hydro, True)[0]
        frag_rdmol = Chem.RemoveHs(frag_rdmol_H)
        frag_smi = Chem.MolToSmiles(frag_rdmol)
        # filter out molecules which are too small or too big
        na = frag_rdmol.GetNumAtoms()
        if na < min_atoms or na > max_atoms:
            continue
        lfrag_rdmol.append(frag_rdmol)
    return lfrag_rdmol


def connect_core_and_chains(combi_smiles: str) -> Mol | None:
    """Connects scaffold core and chains(R-groups).

    `combi_smiles` has [`expr`:`n`] format where `expr` is an expression of atom mapping
    and n is the atom map number.
    It is modified from the Patrick Walters' work,
    [Re: [Rdkit-discuss] reassembling a molecule from R-groups](https://sourceforge.net/p/rdkit/mailman/message/36294482/).
    Examples:
        >>> combi_smiles = "CN(C)CC(Br)c1cc([*:2])c([*:1])cn1.[H]C([H])([H])[*:1].[H][*:2]"
        >>> m = connect_core_and_chains(combi_smiles)

    Args:
        combi_smiles (str): concatenated SMILES containing core followed by chains.

    Returns:
        Union[Mol, None]: connected rdworks.Mol object.
    """
    connect = defaultdict(list)
    core_and_chains = Chem.MolFromSmiles(combi_smiles, sanitize=False)
    # default sanitize=True causes WARNING: not removing hydrogen atom with dummy atom neighbors
    if core_and_chains is None:  # invalid SMILES
        return None
    for atom in core_and_chains.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num > 0:
            connect[atom_map_num].append(atom)
    # Second pass, transfer the atom maps to the neighbor atoms
    for atom_map_num, atom_list in connect.items():
        if len(atom_list) == 2:
            atm_1, atm_2 = atom_list
            if not atm_1.GetBonds() or not atm_2.GetBonds():
                continue  # a bare dummy atom has no neighbor to connect
            nbr_1 = [x.GetOtherAtom(atm_1) for x in atm_1.GetBonds()][0]
            nbr_1.SetAtomMapNum(atom_map_num)
            nbr_2 = [x.GetOtherAtom(atm_2) for x in atm_2.GetBonds()][0]
            nbr_2.SetAtomMapNum(atom_map_num)
    # Nuke all of the dummy atoms
    new_mol = Chem.DeleteSubstructs(core_and_chains, Chem.MolFromSmarts("[#0]"))
    # Third pass - arrange the atoms with AtomMapNum, these will be connected
    connect_bond = defaultdict(list)
    for atom in new_mol.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num > 0:
            connect_bond[atom_map_num].append(atom.GetIdx())
    # Make an editable molecule and add bonds between atoms with correspoing AtomMapNum
    em = EditableMol(new_mol)
    for atom_map_num, atom_list in connect_bond.items():
        if len(atom_list) == 2:
            start_atm, end_atm = atom_list
            em.AddBond(start_atm, end_atm, order=Chem.rdchem.BondType.SINGLE)
    bonded_mol = em.GetMol()
    # remove the AtomMapNum values
    for atom in bonded_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        return Mol(Chem.RemoveHs(bonded_mol))  # returns a new molecule
    except:
        return None


def expand_rgroup(
    core: str, r: dict[int, list[str]], prefix: str = "RGX", start: int = 1, **kwargs
) -> MolLibr:
    """Expands or enumerates the core with R groups.

    `core` SMILES has `[#0:n]` where n is the attachment index. All attachments are enumerated.

    Examples:
        >>> core = "[#0:1]C1C([#0:4])C([#0:3])CN2[C@@H]([C@H](C3=CC=C(C#CC4=CC=CC=C4)C=C3)[C@]2([H])CN1C(NC5=CC=C(C=C5)OC)=O)[#0:2]"
        >>> V = "[*]C#N.[*]C(O)=O.[*]CO.[*]C(NC)=O.[*]CNC(C)=O.[*]CC=C.[*][H]".split(".")
        >>> W = "[*][H].[*]COC".split(".")
        >>> X = "[*][H].[*]CC(F)(F)F.[*]OCCOC".split(".")
        >>> Y = "[*][H].[*]O.[*]OC".split(".")
        >>> output = expand_rgroup(core=core, r={1:V, 2:W, 3:X, 4:Y})

    Args:
        core (str): SMILES for the core.
        r (Dict[int, List[str]]): dictionary of {atom mapping number: list of SMILES for R groups}.
        prefix (str, optional): name prefix of generated molecules. Defaults to 'RGX'.
        start (int, optional): start number for serial numbers. Defaults to 1.

    Returns:
        MolLibr: a library of generated molecules.
    """
    obj = MolLibr().compute(**kwargs)
    r_groups = defaultdict(list)
    for atom_map_num, smiles_list in r.items():
        for smiles in smiles_list:
            smiles = smiles.replace("[H]", "[#1]")
            smiles = smiles.replace("[*]", f"[*:{atom_map_num}]")
            r_groups[atom_map_num].append(smiles)
    combi_smiles = [
        core + "." + ".".join(c) for c in product(*(r_groups[i] for i in r_groups))
    ]
    with ProcessPoolExecutor(max_workers=obj.max_workers) as executor:
        if obj.progress:
            obj.libr = list(
                tqdm(
                    executor.map(
                        connect_core_and_chains, combi_smiles, chunksize=obj.chunksize
                    ),
                    desc="Expand R-Group",
                    total=len(combi_smiles),
                )
            )
        else:
            obj.libr = list(
                executor.map(
                    connect_core_and_chains, combi_smiles, chunksize=obj.chunksize
                )
            )
    # combinations that cannot be connected (e.g., invalid valence) return None
    n_failed = sum(mol is None for mol in obj.libr)
    if n_failed:
        print(f"Expand R-Group: {n_failed} combination(s) could not be connected")
    obj.libr = [mol for mol in obj.libr if mol is not None]
    return obj.rename(prefix=prefix, start=start)


@lru_cache(maxsize=1)
def _load_rgroups() -> dict[str, dict]:
    """Loads R-group data from data/Rgroups.json.

    Returns:
        dict[str, dict]: R-group data keyed by attribute name.
    """
    resource = files("rdworks.data").joinpath("Rgroups.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def __getattr__(name: str) -> list[str]:
    """Lazily provides the R-group lists defined in data/Rgroups.json.

    Args:
        name (str): attribute name, 'most_common' or 'most_common_in_NP'.

    Returns:
        list[str]: list of R-group SMILES.

    Raises:
        AttributeError: if the attribute name is not 'most_common' or 'most_common_in_NP'.
    """
    if name in ("most_common", "most_common_in_NP"):
        return _load_rgroups()[name]["smiles"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
