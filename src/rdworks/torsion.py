"""Torsion utilities: rotatable-bond detection, torsion fragmentation, and torsion scans.

Torsion scan functions take an `rdworks.Conf` and store potential energy profiles
in `conf.props['torsion']`. `Conf` methods of the same names delegate here.
"""

import itertools

import numpy as np

from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING
from rdkit import Chem
from rdworks.batch import prepare_batches
from rdworks.std import clean_2d
from rdworks.xtb.wrapper import GFN2xTB

if TYPE_CHECKING:
    # only for type hints; a runtime import would be circular (rdworks.conf imports this module)
    from rdworks.conf import Conf


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


def find_atoms_at_bond_distance(
    rdmol: Chem.Mol, start_atom_idx: int, distance: int
) -> list[int]:
    """Finds atoms at a specific bond distance from a starting atom.

    The distance is the length of any simple path (no repeated atoms) from the
    starting atom, not only the shortest path. Atoms reachable by more than one
    path of the given length appear more than once.

    Args:
        rdmol (Chem.Mol): molecule.
        start_atom_idx (int): index of the starting atom.
        distance (int): number of bonds from the starting atom.

    Returns:
        list[int]: atom indices at the specified distance (may contain duplicates).
    """
    assert start_atom_idx < rdmol.GetNumAtoms(), "start_atom_idx out of range."

    found_atoms = []
    visited = set()

    def dfs(curr_atom_idx: int, curr_bond_dist: int) -> None:
        """Depth-first search over simple paths, recording atoms at `distance`."""
        if curr_bond_dist == distance:
            found_atoms.append(curr_atom_idx)
            return

        visited.add(curr_atom_idx)
        curr_atom = rdmol.GetAtomWithIdx(curr_atom_idx)
        for next_atom in curr_atom.GetNeighbors():
            next_atom_idx = next_atom.GetIdx()
            if next_atom_idx not in visited:
                dfs(next_atom_idx, curr_bond_dist + 1)

        # Backtrack
        visited.remove(curr_atom_idx)

    dfs(start_atom_idx, 0)

    return found_atoms


def get_bond_distance(rdmol: Chem.Mol, start_atom_idx: int) -> dict:
    """Get bond distances of all atoms from a given atom.

    Distances are lengths of simple paths (no repeated atoms), not only shortest
    paths, so an atom can appear under more than one distance.

    Args:
        rdmol (Chem.Mol): molecule.
        start_atom_idx (int): index of the starting atom.

    Returns:
        dict[int, list[int]]: {bond distance: [atom indices], ...}
    """
    assert start_atom_idx < rdmol.GetNumAtoms(), "start_atom_idx out of range."

    bond_distance = {}
    visited = set()

    def dfs(curr_atom_idx: int, curr_bond_dist: int) -> None:
        """Depth-first search over simple paths, recording atoms by path length."""
        if curr_bond_dist in bond_distance:
            bond_distance[curr_bond_dist].append(curr_atom_idx)
        else:
            bond_distance[curr_bond_dist] = [curr_atom_idx]

        visited.add(curr_atom_idx)
        curr_atom = rdmol.GetAtomWithIdx(curr_atom_idx)
        for next_atom in curr_atom.GetNeighbors():
            next_atom_idx = next_atom.GetIdx()
            if next_atom_idx not in visited:
                dfs(next_atom_idx, curr_bond_dist + 1)

        # Backtrack
        visited.remove(curr_atom_idx)

    dfs(start_atom_idx, 0)

    return bond_distance


def find_bonds_to_prune(
    rdmol: Chem.Mol,
    torsion_indices: tuple,
    bond_dist_threshold: int = 4,
    bond_order_threshold: float = 1.75,
    electronegative: tuple[int, ...] = (7, 8, 9, 17, 35),
) -> dict[int, list[int]]:
    """Find candidate bonds to break to construct a fragment around a torsion.

    Rules for a candidate bond to break:

        For (i-j-k-l) torsion,

        1. NOT (bond distance from j or k < `bond_dist_threshold`)
        2. NOT (bond order > `bond_order_threshold`)
        3. NOT (any of bond atoms in `electronegative`, i.e. Pauling electronegativity > 2.9)
        4. NOT (bond in a ring)

    Args:
        rdmol (Chem.Mol): molecule.
        torsion_indices (tuple): (i, j, k, l, ...) torsion atom indices; only the first four are used.
        bond_dist_threshold (int, optional): minimum bond distance from j or k. Defaults to 4.
        bond_order_threshold (float, optional): maximum bond order of a breakable bond. Defaults to 1.75.
        electronegative (tuple[int, ...], optional): atomic numbers of electronegative atoms
            whose bonds are not broken. Defaults to (7, 8, 9, 17, 35) (N, O, F, Cl, Br).

    Pauling electronegativity:
        ```py
        from mendeleev import element
        for i in range(1, 119):  # 118 is the highest atomic number known
            el = element(i)
            if isinstance(el.en_pauling, float) and el.en_pauling > 2.9:
                print(f"Element {i}: {el.symbol} {el.atomic_number} {el.en_pauling}")
        ```
        Element 7: N 7 3.04
        Element 8: O 8 3.44
        Element 9: F 9 3.98
        Element 17: Cl 17 3.16
        Element 35: Br 35 2.96

    Returns:
        dict[int, list[int]]: {bond index: [atom index closer to the torsion, the other atom index], ...}
    """

    (i, j, k, l) = torsion_indices[:4]

    dist_from_j = get_bond_distance(rdmol, j)
    dist_from_k = get_bond_distance(rdmol, k)

    # sum(,[]) flattens a list of list
    forbidden = sum([v for d, v in dist_from_j.items() if d < bond_dist_threshold], [])
    forbidden += sum([v for d, v in dist_from_k.items() if d < bond_dist_threshold], [])
    forbidden = set(forbidden)

    start_atom_idx = k  # either j or k yields the same result

    found_bonds = {}
    visited = set()

    def ordered(p: int, q: int) -> list[int]:
        """Returns a list of atom indices by bond distance.

        Args:
            p (int): atom index
            q (int): atom index

        Returns:
            list[int]: (atom index closer to the torsion angle, the other)
        """
        dist_p = []
        dist_q = []
        for d, indices in dist_from_j.items():
            if p in indices:
                dist_p.append(d)
            if q in indices:
                dist_q.append(d)
        for d, indices in dist_from_k.items():
            if p in indices:
                dist_p.append(d)
            if q in indices:
                dist_q.append(d)
        if sum(dist_p) < sum(dist_q):
            return [p, q]
        else:
            return [q, p]

    def dfs(curr_atom_idx: int, bond_dist: int) -> None:
        """Depth-first recursive search of bonded atoms.

        Args:
            curr_atom_idx (int): atom index.
            bond_dist (int): bond distance.
        """
        curr_atom = rdmol.GetAtomWithIdx(curr_atom_idx)
        visited.add(curr_atom_idx)
        for next_atom in curr_atom.GetNeighbors():
            next_atom_idx = next_atom.GetIdx()
            bond = rdmol.GetBondBetweenAtoms(curr_atom_idx, next_atom_idx)
            n1 = curr_atom.GetAtomicNum()
            n2 = next_atom.GetAtomicNum()
            # forbidden (rule 1)
            too_close = (curr_atom_idx in forbidden) and (next_atom_idx in forbidden)
            # bond order (rule 2)
            bond_order = not (bond.GetBondTypeAsDouble() > bond_order_threshold)
            # Pauling electronegativity (rule 3)
            bond_pauling = not ((n1 in electronegative) or (n2 in electronegative))
            if (
                (bond_dist >= bond_dist_threshold)
                and (not too_close)
                and (not bond.IsInRing())
                and bond_order
                and bond_pauling
            ):
                # determine which atom has shorter bond distance to the torsion angle (j or k)
                found_bonds[bond.GetIdx()] = ordered(curr_atom_idx, next_atom_idx)
                return
            if next_atom_idx not in visited:
                dfs(next_atom_idx, bond_dist + 1)
        # Backtrack
        visited.remove(curr_atom_idx)

    dfs(start_atom_idx, 0)

    return found_bonds


def get_fragment_idx(parent: Chem.Mol, indices: tuple, fragment: Chem.Mol) -> tuple:
    """Get fragment atom indices corresponding to given parent indices.

    It uses 3D coordinates to find matching atoms between parent and fragment,
    so the fragment must keep the parent coordinates unchanged.
    It is much faster than the MCS-based method `get_fragment_idx_with_mcs()`.
    For example, below are the elapsed times for 12 torsion angles of atorvastatin:
            0 elapsed=0.0006455129478126764 sec.
            1 elapsed=0.0005964740412309766
            2 elapsed=0.0005442029796540737
            3 elapsed=0.000652436981908977
            4 elapsed=0.0006737819639965892
            5 elapsed=0.0004481689538806677
            6 elapsed=0.00035582599230110645
            7 elapsed=0.0003812289796769619
            8 elapsed=0.000359484925866127
            9 elapsed=0.0002818549983203411
            10 elapsed=0.000247497926466167
            11 elapsed=0.0003651580773293972

    Args:
        parent (Chem.Mol): rdkit Chem.Mol object.
        indices (tuple): parent atom indices to map.
        fragment (Chem.Mol): fragment originated from the parent.

    Returns:
        tuple[int, ...]: fragment atom indices in the same order as `indices`.
    """
    parent_xyz = parent.GetConformer().GetPositions()  # numpy.ndarray
    frag_xyz = fragment.GetConformer().GetPositions()  # numpy.ndarray
    qpos = [parent_xyz[i] for i in indices]

    frag_indices = tuple(
        j for q in qpos for j, f in enumerate(frag_xyz) if np.array_equal(f, q)
    )
    if len(frag_indices) != len(indices):
        raise ValueError(
            f"get_fragment_idx() Error: cannot map parent indices {indices} to the fragment"
        )

    return frag_indices


def get_fragment_idx_with_mcs(
    parent: Chem.Mol, indices: tuple, fragment: Chem.Mol
) -> tuple:
    """Get fragment atom indices corresponding to given parent indices.

    Warning:
        It uses MCS and can be extremely slow sometimes.
        For example, below are the elapsed times for 12 torsion angles of atorvastatin:
            0 elapsed=5.525973221054301 sec. **
            1 elapsed=1.9143556850031018 *
            2 elapsed=3.145250838017091 *
            3 elapsed=9.390580283012241 **
            4 elapsed=89.97735002799891 ***
            5 elapsed=0.19022215204313397
            6 elapsed=0.013428106089122593
            7 elapsed=0.023345661000348628
            8 elapsed=0.023358764010481536
            9 elapsed=0.0007965450640767813
            10 elapsed=0.0008196790004149079
            11 elapsed=0.04075543500948697

    Args:
        parent (Chem.Mol): rdkit Chem.Mol object.
        indices (tuple): parent atom indices to map within the MCS.
        fragment (Chem.Mol): fragment originated from the parent.

    Returns:
        tuple[int, ...]: fragment atom indices in the same order as `indices`.
    """
    mcs_result = Chem.rdFMCS.FindMCS([parent, fragment])
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    parent_matches = parent.GetSubstructMatches(mcs_mol)
    frag_matches = fragment.GetSubstructMatches(mcs_mol)

    indices_idx = None
    for parent_matched_indices in parent_matches:
        # It is possible to have more than one matches (i.e. methyl rotation).
        # However, even if there are more than one matches, the parent indices
        # should be the same.
        if indices_idx is None:
            indices_idx = {x: parent_matched_indices.index(x) for x in indices}
        else:
            assert all(
                [indices_idx[x] == parent_matched_indices.index(x) for x in indices]
            )

    indices_map = None
    for frag_matched_indices in frag_matches:
        # it is possible to have more than one matches (i.e. methyl rotation)
        if indices_map is None:
            indices_map = {x: frag_matched_indices[indices_idx[x]] for x in indices}
        else:
            assert all(
                [
                    indices_map[x] == frag_matched_indices[indices_idx[x]]
                    for x in indices
                ]
            )

    return tuple([indices_map[x] for x in indices])


def create_fragment_on_bonds(
    rdmol: Chem.Mol, bonds: dict, cap: bool = True
) -> Chem.Mol | None:
    """Create a fragment that preserves defined atoms.

    Args:
        rdmol (Chem.Mol): input molecule.
        bonds (dict): {bond_index : (preserved_atom_index, removed_atom_index), ...}
        cap (bool): whether to cap the dummy atom(s) with hydrogen(s). Defaults to True.

    Returns:
        Chem.Mol | None: resulting fragment molecule, or None if no fragment
            contains all preserved atoms.
    """
    fragments = Chem.FragmentOnBonds(rdmol, list(bonds))
    preserved_atoms = {preserved for bond_idx, (preserved, removed) in bonds.items()}
    for fragment_indices, fragment_mol in zip(
        Chem.GetMolFrags(fragments), Chem.GetMolFrags(fragments, asMols=True)
    ):
        if preserved_atoms.issubset(set(fragment_indices)):
            if cap:
                # cap dummy atoms with hydrogens
                for atom in fragment_mol.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        atom.SetAtomicNum(1)

            return fragment_mol

    return None


def create_torsion_fragment(
    rdmol: Chem.Mol, torsion_indices: tuple, wbo_tolerance: float = 0.03
) -> tuple[Chem.Mol, tuple[int, ...], bool, bool]:
    """Create a close surrogate fragment that captures the PES of the intended torsion.

    Fragmentation aims to preserve the local chemical environment around the targeted torsion
    while increasing calculation speed and avoiding potential complications. To avoid oversimplification
    and inaccurate approximation, two strategies are combined:
        - fragment candidates are generated by a set of reasonably empirical rules
        - further filtered by Wiberg bond order (WBO) calculated by semi-empirical QM. It has
        been shown that the Wiberg bond order (WBO) provides a fast and robust measure of
        whether a torsion profile has been disrupted by fragmentation. Any fragment that causes
        WBO difference larger than `wbo_tolerance` will be excluded.
        WBO filtering is used only if xTB is available.

    Args:
        rdmol (Chem.Mol): molecule.
        torsion_indices (tuple): (i, j, k, l, atoms to be rotated, atoms to be fixed)
        wbo_tolerance (float, optional): maximum allowed WBO change at the j-k bond.
            Defaults to 0.03.

    Returns:
        (Chem.Mol: fragment molecule,
        tuple[int, ...]: fragment atom indices of (i, j, k, l),
        bool: True if fragmented,
        bool: True if WBO filtering is used)

    References:
        https://pubs.acs.org/doi/10.1021/acs.jcim.2c01153
        https://www.biorxiv.org/content/10.1101/2020.08.27.270934v2
    """
    (i, j, k, l) = torsion_indices[:4]

    candidates = find_bonds_to_prune(rdmol, torsion_indices)

    if not candidates:
        # no fragmentation
        return (rdmol, tuple(torsion_indices[:4]), False, False)

    WBO_filtered = False
    if GFN2xTB().is_ready():
        # filter candidate(s) by Wiberg bond order (WBO) if xTB is available
        jk = tuple(sorted([j, k]))
        wbo_passed_candidates = {}
        parent = GFN2xTB(rdmol).singlepoint()
        assert hasattr(parent, "wbo"), (
            "create_torsion_fragment() Error: no wbo for parent"
        )
        for bond_idx, (p, q) in candidates.items():
            frag_single_break = create_fragment_on_bonds(rdmol, {bond_idx: (p, q)})
            fragment = GFN2xTB(frag_single_break).singlepoint()
            assert hasattr(fragment, "wbo"), (
                "create_torsion_fragment() Error: no wbo for fragment"
            )
            # WBO difference at the torsion angle bond
            frag_jk = get_fragment_idx(rdmol, jk, frag_single_break)
            frag_jk = tuple(sorted(frag_jk))
            if abs(fragment.wbo[frag_jk] - parent.wbo[jk]) < wbo_tolerance:
                wbo_passed_candidates[bond_idx] = (p, q)
        WBO_filtered = True
        if not wbo_passed_candidates:
            # every candidate disrupts the torsion bond (WBO): no fragmentation
            return (rdmol, tuple(torsion_indices[:4]), False, WBO_filtered)
        frag_multi_breaks = create_fragment_on_bonds(rdmol, wbo_passed_candidates)
    else:
        # skip WBO filtering
        frag_multi_breaks = create_fragment_on_bonds(rdmol, candidates)

    frag_indices = get_fragment_idx(rdmol, (i, j, k, l), frag_multi_breaks)

    return (frag_multi_breaks, frag_indices, True, WBO_filtered)


def find_shortest_path_excluding_torsion(
    rdmol: Chem.Mol, i: int, j: int, k: int, l: int
) -> list | None:
    """Find the shortest path between atoms i and l that avoids the torsion bonds i-j, j-k, k-l.

    Uses breadth-first search. A path exists only if i and l are connected
    through a ring or another route.

    Args:
        rdmol (Chem.Mol): molecule.
        i (int): first atom index of the torsion.
        j (int): second atom index of the torsion.
        k (int): third atom index of the torsion.
        l (int): fourth atom index of the torsion.

    Returns:
        list[int] | None: atom indices from i to l, or None if no alternative path exists.
    """
    # Get the bonds to exclude (the torsion path)
    excluded_bonds = set()
    bond_ij = rdmol.GetBondBetweenAtoms(i, j)
    bond_jk = rdmol.GetBondBetweenAtoms(j, k)
    bond_kl = rdmol.GetBondBetweenAtoms(k, l)

    if bond_ij:
        excluded_bonds.add(bond_ij.GetIdx())
    if bond_jk:
        excluded_bonds.add(bond_jk.GetIdx())
    if bond_kl:
        excluded_bonds.add(bond_kl.GetIdx())

    # BFS to find shortest path from i to l
    queue = deque([(i, [i])])  # (current_atom, path)
    visited = {i}

    while queue:
        current, path = queue.popleft()

        # Check neighbors
        atom = rdmol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()

            # Get the bond between current and neighbor
            bond = rdmol.GetBondBetweenAtoms(current, neighbor_idx)

            # Skip if this bond is part of the torsion path
            if bond and bond.GetIdx() in excluded_bonds:
                continue

            # Found the target
            if neighbor_idx == l:
                return path + [l]

            # Continue BFS
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append((neighbor_idx, path + [neighbor_idx]))

    return None  # No alternative path found


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


def get_torsion_plot_data(torsion_profiles: list[dict], idx: int | None = None) -> dict:
    """Returns plot data (torsion angle vs. relative energy) from torsion scan results.

    Args:
        torsion_profiles (list[dict]): torsion profiles, i.e. `props['torsion']` set by
            `calculate_torsion_energies()` or `calculate_sp_torsion_energies()`.
        idx (int, optional): 0-based torsion angle index.
            Defaults to None (all torsion angles).

    Returns:
        {'idx': list, 'angle': list, 'E_rel(kcal/mol)': list}
    """
    if isinstance(idx, int):
        _dict = torsion_profiles[idx]
        angles, energies = _dict["angle"], _dict["E_rel(kcal/mol)"]
        assert len(angles) == len(energies)
        data = {
            "idx": [
                idx,
            ]
            * len(angles),
            "angle": angles,
            "E_rel(kcal/mol)": energies,
        }
    else:
        data = {"idx": [], "angle": [], "E_rel(kcal/mol)": []}
        for idx, _dict in enumerate(torsion_profiles):  # list
            angles, energies = _dict["angle"], _dict["E_rel(kcal/mol)"]
            assert len(angles) == len(energies)
            data["idx"].extend(
                [
                    idx,
                ]
                * len(angles)
            )
            data["angle"].extend(angles)
            data["E_rel(kcal/mol)"].extend(energies)

    return data
