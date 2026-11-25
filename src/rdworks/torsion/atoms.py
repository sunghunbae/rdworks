from rdkit import Chem


def get_torsion_angle_atom_indices(rdmol: Chem.Mol, strict: bool = True) -> list[tuple]:
    """Determine dihedral angle atoms (a-b-c-d) and rotating group for each rotatable bond (b-c).

    The bond b-c is split into two fragments. The smaller fragment (by heavy atom count)
    is the rotating group and must contain at least three heavy atoms.
    A rotatable bond is skipped when:
        - b or c has no neighbor other than H, F, Cl, Br, or I to define a or d
        - a-b or c-d is in a small ring (3 or 4 atoms)
        - a and d are in the same ring (macrocycle)

    Args:
        rdmol (Chem.Mol): molecule.
        strict (bool): whether to exclude amide/imide/ester/acid bonds. Defaults to True.

    Returns:
        list[tuple[int, int, int, int, tuple[int, ...], tuple[int, ...]]]:
            [   (a, b, c, d, rot_atom_indices, fix_atom_indices),
                (a, b, c, d, rot_atom_indices, fix_atom_indices),
                ...,
            ]
    """
    # https://github.com/rdkit/rdkit/blob/1bf6ef3d65f5c7b06b56862b3fb9116a3839b229/rdkit/Chem/Lipinski.py#L47%3E
    # https://github.com/rdkit/rdkit/blob/de602c88809ea6ceba1e8ed50fd543b6e406e9c4/Code/GraphMol/Descriptors/Lipinski.cpp#L108
    if strict:
        # excludes amide/imide/ester/acid bonds
        rotatable_bond_pattern = Chem.MolFromSmarts(
            (
                "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])("
                "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]="
                "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$"
                "(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])(["
                "CH3])[CH3])]"
            )
        )
    else:
        rotatable_bond_pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

    rotatable_bonds = rdmol.GetSubstructMatches(rotatable_bond_pattern)
    ri = rdmol.GetRingInfo()
    atom_rings = ri.AtomRings()

    torsion_angle_atom_indices = []

    # small rings (n=3 or 4)
    small_rings = [ring for ring in list(atom_rings) if len(ring) < 5]
    # ex. = [(1, 37, 35, 34, 3, 2), (29, 28, 30)]

    forbidden_terminal_nuclei = [1, 9, 17, 35, 53]  # H,F,Cl,Br,I

    for b_idx, c_idx in rotatable_bonds:
        # determine a atom ``a`` that define a dihedral angle
        a_candidates = []
        for neighbor in rdmol.GetAtomWithIdx(b_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx == c_idx:
                continue
            neighbor_atomic_num = neighbor.GetAtomicNum()
            if neighbor_atomic_num not in forbidden_terminal_nuclei:
                a_candidates.append((neighbor_atomic_num, neighbor_idx))

        if not a_candidates:
            continue

        (a_atomic_num, a_idx) = sorted(
            a_candidates, key=lambda x: (x[0], -x[1]), reverse=True
        )[0]

        # is a-b in a small ring (n=3 or 4)?
        is_in_small_ring = False
        for small_ring in small_rings:
            if (a_idx in small_ring) and (b_idx in small_ring):
                is_in_small_ring = True
                break

        if is_in_small_ring:
            continue

        # determine a atom ``d`` that define a dihedral angle
        d_candidates = []
        for neighbor in rdmol.GetAtomWithIdx(c_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx == b_idx:
                continue
            neighbor_atomic_num = neighbor.GetAtomicNum()
            if neighbor_atomic_num not in forbidden_terminal_nuclei:
                d_candidates.append((neighbor_atomic_num, neighbor_idx))

        if not d_candidates:
            continue

        (d_atomic_num, d_idx) = sorted(
            d_candidates, key=lambda x: (x[0], -x[1]), reverse=True
        )[0]

        # is c-d in a small ring?
        is_in_small_ring = False
        for small_ring in small_rings:
            if (c_idx in small_ring) and (d_idx in small_ring):
                is_in_small_ring = True
                break

        if is_in_small_ring:
            continue

        # check ring closure between a and d - avoid macrocycles
        if any(a_idx in ring and d_idx in ring for ring in atom_rings):
            continue

        # determine a group of atoms to be rotated
        # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
        em = Chem.EditableMol(rdmol)
        em.RemoveBond(b_idx, c_idx)
        fragmented = em.GetMol()
        (frag1, frag2) = Chem.GetMolFrags(
            fragmented, asMols=False
        )  # returns tuple of tuple
        hac1 = sum([1 for i in frag1 if rdmol.GetAtomWithIdx(i).GetAtomicNum() > 1])
        hac2 = sum([1 for i in frag2 if rdmol.GetAtomWithIdx(i).GetAtomicNum() > 1])

        # smaller fragment will be rotated and must contain at least three heavy atoms
        if min(hac1, hac2) >= 3:
            (frag_rot, frag_fix) = sorted([(hac1, frag1), (hac2, frag2)])
            torsion_angle_atom_indices.append(
                (a_idx, b_idx, c_idx, d_idx, frag_rot[1], frag_fix[1])
            )

    return torsion_angle_atom_indices