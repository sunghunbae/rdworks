"""Torsion utilities: rotatable-bond detection, torsion fragmentation, and torsion scans.

Submodules:
    atom_indices:  rotatable-bond (dihedral a-b-c-d) detection
    fragment:      torsion fragment creation around a rotatable bond
    sp_relaxation: torsion energy scans; they take an `rdworks.Conf` and store potential
                   energy profiles in `conf.props['torsion']` (`Conf` methods of the
                   same names delegate here)
    plot:          plot data from torsion scan results

The public functions are re-exported here, so `from rdworks.torsion import ...`
keeps working as before the split.
"""

from rdworks.torsion.atom_indices import get_torsion_angle_atom_indices
from rdworks.torsion.fragment import create_torsion_fragment
from rdworks.torsion.sp_relaxation import torsion_energy_profile, torsion_scan
from rdworks.torsion.plot import get_torsion_plot_data

__all__ = [
    "get_torsion_angle_atom_indices",
    "create_torsion_fragment",
    "torsion_energy_profile",
    "torsion_scan",
    "get_torsion_plot_data",
]