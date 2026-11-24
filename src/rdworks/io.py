"""Reading and writing molecules.

Re-exports the readers from `rdworks.readers` and the writers from
`rdworks.writers`, so `from rdworks.io import read_sdf, to_sdf` keeps working.
Inside rdworks, `Conf`, `Mol`, and `MolLibr` import `rdworks.writers` directly:
importing this module loads the readers, which depend on those classes.
"""

from rdworks.readers import (
    conf_name_convention,
    validate_path,
    guess_molecular_id,
    read_csv,
    merge_csv,
    read_dataframe,
    read_smi,
    read_sdf,
    read_mae,
)
from rdworks.writers import (
    to_geometry,
    to_xyz,
    to_turbomole,
    to_sdf,
    to_png,
    to_svg,
    to_df,
    to_csv,
)

__all__ = [
    "conf_name_convention",
    "validate_path",
    "guess_molecular_id",
    "read_csv",
    "merge_csv",
    "read_dataframe",
    "read_smi",
    "read_sdf",
    "read_mae",
    "to_geometry",
    "to_xyz",
    "to_turbomole",
    "to_sdf",
    "to_png",
    "to_svg",
    "to_df",
    "to_csv",
]
