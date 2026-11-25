"""Torsion fragment creation around a rotatable bond."""

from collections import deque

import numpy as np

from rdkit import Chem
from rdworks.xtb.wrapper import GFN2xTB


def _get_bond_distance(rdmol: Chem.Mol, start_atom_idx: int) -> dict:
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


def _find_bonds_to_prune(
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

    dist_from_j = _get_bond_distance(rdmol, j)
    dist_from_k = _get_bond_distance(rdmol, k)

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


def _get_fragment_idx(parent: Chem.Mol, indices: tuple, fragment: Chem.Mol) -> tuple:
    """Get fragment atom indices corresponding to given parent indices.

    It uses 3D coordinates to find matching atoms between parent and fragment,
    so the fragment must keep the parent coordinates unchanged.
    It is much faster than the MCS-based method `_get_fragment_idx_with_mcs()`.

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
            f"_get_fragment_idx() Error: cannot map parent indices {indices} to the fragment"
        )

    return frag_indices


def _create_fragment_on_bonds(
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

    candidates = _find_bonds_to_prune(rdmol, torsion_indices)

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
            frag_single_break = _create_fragment_on_bonds(rdmol, {bond_idx: (p, q)})
            fragment = GFN2xTB(frag_single_break).singlepoint()
            assert hasattr(fragment, "wbo"), (
                "create_torsion_fragment() Error: no wbo for fragment"
            )
            # WBO difference at the torsion angle bond
            frag_jk = _get_fragment_idx(rdmol, jk, frag_single_break)
            frag_jk = tuple(sorted(frag_jk))
            if abs(fragment.wbo[frag_jk] - parent.wbo[jk]) < wbo_tolerance:
                wbo_passed_candidates[bond_idx] = (p, q)
        WBO_filtered = True
        if not wbo_passed_candidates:
            # every candidate disrupts the torsion bond (WBO): no fragmentation
            return (rdmol, tuple(torsion_indices[:4]), False, WBO_filtered)
        frag_multi_breaks = _create_fragment_on_bonds(rdmol, wbo_passed_candidates)
    else:
        # skip WBO filtering
        frag_multi_breaks = _create_fragment_on_bonds(rdmol, candidates)

    frag_indices = _get_fragment_idx(rdmol, (i, j, k, l), frag_multi_breaks)

    return (frag_multi_breaks, frag_indices, True, WBO_filtered)