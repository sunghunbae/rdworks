"""Torsion utilities: rotatable-bond detection, torsion fragmentation, and torsion scans.

Submodules:
    atoms:    rotatable-bond (dihedral a-b-c-d) detection
    fragment: torsion fragment creation around a rotatable bond
    scan:     torsion energy scans; they take an `rdworks.Conf` and store potential
              energy profiles in `conf.props['torsion']` (`Conf` methods of the same
              names delegate here)
    plot:     plot data from torsion scan results

The public functions are re-exported here, so `from rdworks.torsion import ...`
keeps working as before the split.
"""

from rdworks.torsion.atoms import get_torsion_angle_atom_indices
from rdworks.torsion.fragment import (
    create_fragment_on_bonds,
    create_torsion_fragment,
    find_atoms_at_bond_distance,
    find_bonds_to_prune,
    find_shortest_path_excluding_torsion,
    get_bond_distance,
    get_fragment_idx,
    get_fragment_idx_with_mcs,
)
from rdworks.torsion.scan import (
    calculate_sp_torsion_energies,
    calculate_torsion_energies,
    calculate_torsion_energies_batch,
    calculate_torsion_energies_one,
    sp_torsion_energies,
    sp_torsion_energies_batch,
)
from rdworks.torsion.plot import get_torsion_plot_data

__all__ = [
    # atoms
    "get_torsion_angle_atom_indices",
    # fragment
    "create_fragment_on_bonds",
    "create_torsion_fragment",
    "find_atoms_at_bond_distance",
    "find_bonds_to_prune",
    "find_shortest_path_excluding_torsion",
    "get_bond_distance",
    "get_fragment_idx",
    "get_fragment_idx_with_mcs",
    # scan
    "calculate_sp_torsion_energies",
    "calculate_torsion_energies",
    "calculate_torsion_energies_batch",
    "calculate_torsion_energies_one",
    "sp_torsion_energies",
    "sp_torsion_energies_batch",
    # plot
    "get_torsion_plot_data",
]
