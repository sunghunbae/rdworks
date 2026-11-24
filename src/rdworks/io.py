"""Reading and writing molecules.

The writers (`to_geometry`, `to_xyz`, `to_turbomole`, `to_sdf`, `to_png`, `to_svg`,
`to_df`, `to_csv`) accept `rdworks.Conf`, `rdworks.Mol`, `rdworks.MolLibr`, or
`rdkit.Chem.Mol` objects and are the single implementation behind the `to_*`
methods of `Conf`, `Mol`, and `MolLibr`.

This module must not import `rdworks.Conf`, `Mol`, or `MolLibr` at module level:
those modules import this one, so the classes are imported inside functions.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING

from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

from rdworks.utils import compute
from rdworks.units import angstrom2bohr
from rdworks.view import render_matrix_grid, render_png, render_svg

if TYPE_CHECKING:
    from rdworks import Mol, MolLibr

import pandas as pd
import re
import gzip


conf_name_convention = re.compile(r"[a-zA-Z0-9-_.!@#$%^&*()+=]+.[0-9]+/[0-9]+")


def validate_path(path: str | Path) -> Path:
    """Prechecks filename or path and returns a string for the pathlib.PosixPath.

    Args:
        path (Union[str, PosixPath]): filename or path.

    Raises:
        FileNotFoundError: if the path is not found.

    Returns:
        str: a string for the path.
    """
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)

    if path.exists() and path.is_file():
        return path
    else:
        raise FileNotFoundError(f"File {path.as_posix()} does not exist.")


def guess_molecular_id(rdmols: list[Chem.Mol]) -> str | None:
    """Guesses molecular ID from SDF properties.

    Molecular ID is guessed by the coverage(=count of unique values divided by total count).

    Priority:
        `_Name` > ID property with shortest values > any ID property

    Args:
        rdmols (List[Chem.Mol]): a list of molecules.

    Returns:
        str: property suitable for id
    """
    U = defaultdict(list)
    for rdmol in rdmols:
        properties_dict = rdmol.GetPropsAsDict()
        if "_Name" not in properties_dict.keys():
            U["_Name"].append(rdmol.GetProp("_Name"))
        for k, v in properties_dict.items():
            # float is not suitable for molecular id
            if isinstance(v, float):
                continue
            # str(int) is acceptable for molecular id
            if isinstance(v, int):
                U[k].append(str(v))
            elif isinstance(v, str):
                U[k].append(v)

    id_properties = [k for k, v in U.items() if len(set(v)) == len(rdmols)]
    num_id_properties = len(id_properties)

    if num_id_properties == 0:
        return None

    elif num_id_properties == 1:
        return id_properties[0]

    elif num_id_properties > 1:
        if "_Name" in id_properties:
            return "_Name"
        else:
            # shorter id
            _, id_property = sorted(
                [(max([len(v) for v in U[k]]), k) for k in id_properties], reverse=True
            )[0]
            return id_property


def read_csv(
    path: str | Path, smiles: str, name: str, std: bool = False, **kwargs
) -> MolLibr:
    """Returns a library of molecules reading from a .csv file.

    Other columns will be read as properties.

    Args:
        path (Union[str, Path]): filename or path to a .csv file.
        smiles (str): column for SMILES.
        name (str): column for name.
        std (bool, optional): whether to standardize the input. Defaults to False.

    Raises:
        ValueError: if `smiles` or `name` column is not found in the csv file.

    Returns:
        MolLibr: a library of molecules.
    """
    from rdworks import Mol, MolLibr  # imported here to avoid a circular import

    path = validate_path(path)
    df = pd.read_csv(path)
    try:
        assert smiles in list(df.columns)
    except:
        raise ValueError(f"Cannot find SMILES column (`smiles=`) {smiles}")
    try:
        assert name in list(df.columns)
    except:
        raise ValueError(f"Cannot find NAME column (`name=`) {name}")

    largs = [
        (smiles, name, std) for smiles, name in zip(list(df[smiles]), list(df[name]))
    ]
    libr = MolLibr(compute(Mol, largs, desc="Reading CSV", **kwargs))

    # read other columns as properties
    # A list of dictionaries, where each dictionary represents a row,
    # with column names as keys and cell values as values:
    # [{column -> value}, ..., {column -> value}].
    csv_records = df.to_dict("records")
    for mol, row_dict in zip(libr, csv_records):
        mol.props.update(
            {k: v for (k, v) in row_dict.items() if k not in [smiles, name]}
        )

    return libr


def merge_csv(libr: MolLibr, path: str | Path, on: str = "name") -> MolLibr:
    """Returns a copy of MolLibr merged with properties from `on` column of a .csv file.

    Args:
        libr (MolLibr): library to be merged.
        path (Union[str, Path]): filename or path to a .csv file.
        on (str, optional): column for name. Defaults to 'name'.

    Raises:
        ValueError: if `on` column is not found in the csv file.

    Returns:
        MolLibr: a copy of library of molecules.
    """
    path = validate_path(path)
    df = pd.read_csv(path)
    try:
        assert on in list(df.columns)
    except:
        raise ValueError(f"Cannot find ON column (`on=`) {on}")
    # A list of dictionaries, where each dictionary represents a row,
    # with column names as keys and cell values as values:
    # [{column -> value}, ..., {column -> value}].
    csv_records = df.to_dict("records")
    data = {}
    for row_dict in csv_records:
        data[row_dict[on]] = {k: v for (k, v) in row_dict.items() if k != on}

    merged_libr = libr.copy()

    for mol in merged_libr:
        if mol.name in data:  # mol.props can be partly updated from csv
            mol.props.update(data[mol.name])

    return merged_libr


def read_dataframe(
    df: pd.DataFrame, smiles: str, name: str, std: bool = False
) -> MolLibr:
    """Returns rdworks.MolLibr object from a pandas DataFrame.

    Args:
        df (pd.DataFrame): pandas.DataFrame.
        smiles (str): column for SMILES.
        name (str): column for name.
        std (bool, optional): whether to standardize the input. Defaults to False.

    Raises:
        TypeError: if `df` is not pandas DataFrame.
        ValueError: if `smiles` or `name` column is not found.

    Returns:
        MolLibr: a library of molecules.
    """
    from rdworks import MolLibr  # imported here to avoid a circular import

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expects a pandas.DataFrame object")
    try:
        assert smiles in list(df.columns)
    except:
        raise ValueError(f"Cannot find SMILES column (`smiles=`) {smiles}")
    try:
        assert name in list(df.columns)
    except:
        raise ValueError(f"Cannot find NAME column (`name=`) {name}")

    return MolLibr(list(df[smiles]), list(df[name]), std=std)


def read_smi(path: str | Path, std: bool = False, **kwargs) -> MolLibr:
    """Read a SMILES file and create a molecular library.

    Args:
        path (str | Path): path to the SMILES file.
        std (bool, optional): whether to standardize. Defaults to False.

    Raises:
        FileNotFoundError: when path does not exist.

    Returns:
        MolLibr: a library of molecules.
    """
    from rdworks import Mol, MolLibr  # imported here to avoid a circular import

    path = validate_path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as gz:
            largs = [
                tuple(line.decode("utf-8").strip().split()[:2] + [std]) for line in gz
            ]
    else:
        with open(path, "r") as smi:
            largs = [tuple(line.strip().split()[:2] + [std]) for line in smi]
    return MolLibr(compute(Mol, largs, desc="Reading SMILES", **kwargs))


def _map_sdf(rdmol: Chem.Mol, name: str, std: bool, props: dict) -> Mol:
    """A map function for `read_sdf()` to return a rdworks.Mol object reading from a SDF entry.

    Args:
        rdmol (Chem.Mol): input molecule.
        name (str): name of the molecule.
        std (bool): whether to standardize the input SMILES.
        props (dict): dictionary of molecule properties.

    Returns:
        Mol: rdworks.Mol object.
    """
    from rdworks import Mol  # imported here to avoid a circular import

    obj = Mol(rdmol, name, std)
    obj.props = props
    return obj


def read_sdf(
    path: str | Path,
    name: str | None = None,
    prefix: str | None = None,
    std: bool = False,
    confs: bool = False,
    props: bool = True,
    **kwargs,
) -> MolLibr:
    """Returns a library of molecules reading from a SDF file.

    Args:
        path (Union[str, PosixPath]): filename or path to the .sdf file.
        name (str, optional): property name to be used for name. Defaults to None.
        prefix (str, optional): prefix to be used for name if necessary. Defaults to None.
        std (bool, optional): whether to standardize the input. Defaults to False.
        confs (bool, optional): whether to read 3D conformers and keep hydrogens. Defaults to False.
        props (bool, optional): whether to read SDF properties. Defaults to True.

    Returns:
        MolLibr: a library of molecules.
    """
    from rdworks import Conf, Mol, MolLibr  # imported here to avoid a circular import

    path = validate_path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as gz:
            # switch ^ True, XOR(^) inverts only if switch is True
            with Chem.ForwardSDMolSupplier(
                gz, sanitize=True, removeHs=(confs ^ True)
            ) as gzsdf:
                rdmols = [m for m in gzsdf if m is not None]
    else:
        # switch ^ True, XOR(^) inverts only if switch is True
        with Chem.SDMolSupplier(path, sanitize=True, removeHs=(confs ^ True)) as sdf:
            rdmols = [m for m in sdf if m is not None]

    if isinstance(name, str):
        id_property = name
    else:
        id_property = guess_molecular_id(rdmols)
    if isinstance(id_property, str):
        names = [m.GetProp(id_property) for m in rdmols]
    else:
        names = [f"{prefix}_{i + 1}" for i in range(len(rdmols))]

    if props:
        _props = [m.GetPropsAsDict() for m in rdmols]
    else:
        _props = [None] * len(rdmols)

    _args = [
        (rdmol, name, std, props) for rdmol, name, props in zip(rdmols, names, _props)
    ]

    obj = MolLibr()
    if confs:
        # reading 3D SDF (conformers)
        last_smiles = None
        new_mol = None
        for rdmol, name, props in zip(rdmols, names, _props):
            # rdworks name convention (e.g. xxxx.yy/zzz)
            if conf_name_convention.match(name):
                (isomer_name, _) = name.split("/")
            else:
                isomer_name = name
            smiles = Chem.MolToSmiles(rdmol)  # canonicalized SMILES
            if last_smiles is None or last_smiles != smiles:
                if new_mol:
                    obj.libr.append(new_mol.rename())
                # start a new molecule
                rdmol_2d = Chem.RemoveHs(rdmol)
                AllChem.Compute2DCoords(rdmol_2d)
                # initialize a new molecule with the H-removed 2D
                new_mol = Mol(
                    rdmol_2d, isomer_name, std=False
                )  # atom indices remain unchanged.
            new_conf = Conf(rdmol)
            new_conf.props.update(props)
            new_mol.confs.append(new_conf)
            last_smiles = smiles
        if new_mol:  # handle the last molecule
            obj.libr.append(new_mol.rename())
    else:
        # reading 2D SDF
        obj = MolLibr(compute(_map_sdf, _args, desc="Reading SDF", **kwargs))

    return obj


def read_mae(
    path: str | Path, std: bool = False, confs: bool = True, **kwargs
) -> MolLibr:
    """Returns a library of molecules reading from a Schrodinger Maestro file.

    Args:
        path (Union[str, Path]): filename or path to the .mae or .maegz file.
        std (bool, optional): whether to standardize the input. Defaults to False.
        confs (bool, optional): whether to read 3D conformers. Defaults to True.

    Returns:
        MolLibr: a library of molecules.
    """
    from rdworks import Conf, Mol, MolLibr  # imported here to avoid a circular import

    path = validate_path(path)

    if path.suffix == ".maegz":
        with gzip.open(path, "rb") as gz:
            # switch ^ True, XOR(^) inverts only if switch is True
            with rdmolfiles.MaeMolSupplier(
                gz, sanitize=True, removeHs=(confs ^ True)
            ) as maegz:
                rdmols = [m for m in maegz if m is not None]
    else:
        # switch ^ True, XOR(^) inverts only if switch is True
        with rdmolfiles.MaeMolSupplier(
            path, sanitize=True, removeHs=(confs ^ True)
        ) as mae:
            rdmols = [m for m in mae if m is not None]

    lnames = [m.GetProp("_Name") for m in rdmols]
    largs = [(rdmol, name, std) for rdmol, name in zip(rdmols, lnames)]

    obj = MolLibr()

    if confs:  # reading 3D SDF (conformers)
        last_smiles = None
        new_mol = None
        for rdmol, name in zip(rdmols, lnames):
            # rdworks name convention (e.g. xxxx.yy/zzz)
            if conf_name_convention.match(name):
                (isomer_name, _) = name.split("/")
            else:
                isomer_name = name
            smiles = Chem.MolToSmiles(rdmol)  # canonicalized SMILES
            if last_smiles is None or last_smiles != smiles:
                if new_mol:
                    obj.libr.append(new_mol.rename())
                # start a new molecule
                # !!!! rdmol and new_mol do not have consistent atom indices !!!
                # idxmap: original atom index -> canonicalized rdmol atom index
                # smiles = Chem.MolToSmiles(rdmol) # canonicalization creates `_smilesAtomOutputOrder` property
                # idxord_o = ast.literal_eval(rdmol.GetProp("_smilesAtomOutputOrder"))
                # idxmap_o = {o.GetIdx():idxord_o.index(o.GetIdx()) for o in rdmol.GetAtoms()}
                rdmol_2d = Chem.RemoveHs(rdmol)
                AllChem.Compute2DCoords(rdmol_2d)
                new_mol = Mol(
                    rdmol_2d, isomer_name, std=False
                )  # atom indices remain unchanged.

            new_mol.confs.append(Conf(rdmol))

            last_smiles = smiles
        if new_mol:  # handle the last molecule
            obj.libr.append(new_mol.rename())

    else:  # reading 2D SDF
        obj = MolLibr(compute(Mol, largs, desc="Reading Mae", **kwargs))

    return obj


##################################################
### Writers shared by Conf, Mol, and MolLibr
##################################################


def _classes() -> tuple:
    """Returns (Conf, Mol, MolLibr), imported lazily to avoid a circular import."""
    from rdworks import Conf, Mol, MolLibr

    return Conf, Mol, MolLibr


def _structures(obj, index: int | None = None) -> list[tuple[str, Chem.Mol]]:
    """Returns 3D structures of `obj` as a list of (name, rdkit.Chem.Mol).

    Args:
        obj (Conf | Mol | MolLibr | Chem.Mol): input.
        index (int | None): conformer index for a Mol; None means all conformers.

    Raises:
        TypeError: if `obj` is not a supported type.
        ValueError: if `obj` has no 3D conformer.
    """
    Conf, Mol, MolLibr = _classes()
    if isinstance(obj, Conf):
        structures = [(obj.name, obj.rdmol)]
    elif isinstance(obj, Mol):
        confs = obj.confs if index is None else [obj.confs[index]]
        structures = [(conf.name, conf.rdmol) for conf in confs]
    elif isinstance(obj, MolLibr):
        structures = [(conf.name, conf.rdmol) for mol in obj for conf in mol.confs]
    elif isinstance(obj, Chem.Mol):
        name = obj.GetProp("_Name") if obj.HasProp("_Name") else ""
        structures = [(name, obj)]
    else:
        raise TypeError(
            f"expects rdworks.Conf, Mol, MolLibr, or rdkit.Chem.Mol, got {type(obj).__name__}"
        )
    if not structures:
        raise ValueError("no 3D conformer to write; call make_confs() first")
    return structures


def _one_structure(obj, index: int = 0) -> tuple[str, Chem.Mol]:
    """Returns a single 3D structure (name, rdkit.Chem.Mol) for single-structure formats."""
    _, _, MolLibr = _classes()
    if isinstance(obj, MolLibr):
        raise TypeError("a single-structure format: use one Mol or Conf, not a MolLibr")
    return _structures(obj, index)[0]


def _symbols_positions(rdmol: Chem.Mol) -> zip:
    return zip(
        [atom.GetSymbol() for atom in rdmol.GetAtoms()],
        rdmol.GetConformer().GetPositions(),
    )


def _write_text(text: str, path: str | Path) -> None:
    """Writes `text` to `path`, gzip-compressed if the path ends with .gz."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        with open(path, "w") as f:
            f.write(text)


def to_geometry(obj, index: int = 0) -> str:
    """Returns geometry input for psi4 or other quantum chemistry software.

    Each line has atom symbol and its X, Y, and Z coordinates in Angstrom.

        O  0.0  0.0  0.0
        H  0.757  0.586  0.0
        H -0.757  0.586  0.0

    Example:
        import psi4
        mol = psi4.geometry(to_geometry(conf))

    Args:
        obj (Conf | Mol | Chem.Mol): a single 3D structure.
        index (int, optional): conformer index of a Mol. Defaults to 0.

    Returns:
        str: geometry input.
    """
    _, rdmol = _one_structure(obj, index)
    lines = [f"{e:5} {x:.3f} {y:.3f} {z:.3f}" for e, (x, y, z) in _symbols_positions(rdmol)]
    return "\n".join(lines)


def to_xyz(obj, path: str | Path | None = None) -> str:
    """Returns XYZ formatted strings; multiple conformers give a multi-frame XYZ.

    The comment line of each frame is the conformer name.

    Args:
        obj (Conf | Mol | MolLibr | Chem.Mol): input. A Mol or MolLibr writes all conformers.
        path (str | Path | None, optional): also write to this file (.gz compressed). Defaults to None.

    Returns:
        str: XYZ formatted strings.
    """
    frames = []
    for name, rdmol in _structures(obj):
        lines = [f"{rdmol.GetNumAtoms()}", name or " "]
        for e, (x, y, z) in _symbols_positions(rdmol):
            lines.append(f"{e:5} {x:23.14f} {y:23.14f} {z:23.14f}")
        frames.append("\n".join(lines))
    text = "\n".join(frames)
    if path is not None:
        _write_text(text + "\n", path)
    return text


def to_turbomole(obj, index: int = 0, bohr: bool = False) -> str:
    """Returns TURBOMOLE coord file formatted strings.

    Turbomole coord file format:

        - It starts with the keyword `$coord`.
        - Each line after the $coord line specifies an atom, consisting of:
            - Three real numbers representing the Cartesian coordinates (x, y, z).
            - A string for the element name.
            - Optional: an "f" label at the end to indicate that the atom's coordinates are frozen during optimization.
        - Coordinates can be given in Bohr (default), Angstrom (`$coord angs`), or fractional coordinates (`$coord frac`).
        - Regarding precision:
            The precision of the coordinates is crucial for accurate calculations, especially geometry optimizations.
            Tools like the TURBOMOLEOptimizer might check for differences in atomic positions with a tolerance of 1e-13.

    Args:
        obj (Conf | Mol | Chem.Mol): a single 3D structure.
        index (int, optional): conformer index of a Mol. Defaults to 0.
        bohr (bool, optional): whether to write coordinates in Bohr (`$coord`).
            Otherwise, Angstrom (`$coord angs`). Defaults to False.

    Returns:
        str: TURBOMOLE coord formatted file.
    """
    _, rdmol = _one_structure(obj, index)
    scale = angstrom2bohr if bohr else 1.0
    lines = ["$coord"] if bohr else ["$coord angs"]
    for e, (x, y, z) in _symbols_positions(rdmol):
        lines.append(f"{x * scale:20.15f} {y * scale:20.15f} {z * scale:20.15f} {e}")
    lines.append("$end")
    return "\n".join(lines)


def _sdf_records(obj, confs: bool, props: bool) -> list[tuple[str, Chem.Mol]]:
    """Returns (molecule name, rdkit.Chem.Mol ready to write) SDF records."""
    Conf, Mol, MolLibr = _classes()

    def with_props(rdmol: Chem.Mol, name: str, *dicts: dict) -> Chem.Mol:
        rdmol = Chem.Mol(rdmol)
        rdmol.SetProp("_Name", name)
        if props:
            for d in dicts:
                for k, v in d.items():
                    rdmol.SetProp(k, str(v))
        return rdmol

    if isinstance(obj, Conf):
        return [(obj.name, with_props(obj.rdmol, obj.name, obj.props))]
    if isinstance(obj, Mol):
        if confs:
            # molecule props, then conformer props
            return [
                (obj.name, with_props(conf.rdmol, conf.name, obj.props, conf.props))
                for conf in obj.confs
            ]
        return [(obj.name, with_props(obj.rdmol, obj.name, obj.props))]
    if isinstance(obj, MolLibr):
        return [record for mol in obj for record in _sdf_records(mol, confs, props)]
    if isinstance(obj, Chem.Mol):
        name = obj.GetProp("_Name") if obj.HasProp("_Name") else ""
        return [(name, Chem.Mol(obj))]
    raise TypeError(
        f"expects rdworks.Conf, Mol, MolLibr, or rdkit.Chem.Mol, got {type(obj).__name__}"
    )


def _sdf_string(rdmols: list[Chem.Mol]) -> str:
    buf = StringIO()
    with Chem.SDWriter(buf) as f:
        for rdmol in rdmols:
            f.write(rdmol)
    return buf.getvalue()


def to_sdf(
    obj,
    path: str | Path | None = None,
    confs: bool = False,
    props: bool = True,
    separate: bool = False,
) -> str | None:
    """Returns SDF strings or writes them to .sdf or .sdf.gz file(s).

    `dirname/filename.sdf` -> `dirname/filename_{molecule name}.sdf` if `separate`.

    Args:
        obj (Conf | Mol | MolLibr | Chem.Mol): input. A Conf is written with its 3D
            coordinates and conformer props.
        path (str | Path | None, optional): output file; None returns the SDF string.
            Defaults to None.
        confs (bool, optional): for Mol and MolLibr, write 3D conformers with molecule and
            conformer props instead of the 2D molecule. Defaults to False.
        props (bool, optional): whether to write SDF properties. Defaults to True.
        separate (bool, optional): write each molecule to a separate file. Defaults to False.

    Returns:
        str | None: SDF strings if `path` is None.
    """
    _, _, MolLibr = _classes()
    # one SDWriter per molecule, so the property record counters, e.g. `>  <QED>  (1)`,
    # restart for each molecule as in the per-molecule Mol.to_sdf() output
    molecules = list(obj) if isinstance(obj, MolLibr) else [obj]
    blocks = []  # (molecule name, SDF string)
    for molecule in molecules:
        records = _sdf_records(molecule, confs, props)
        # the molecule's own name, even if it has no conformer records
        if isinstance(molecule, Chem.Mol):
            name = molecule.GetProp("_Name") if molecule.HasProp("_Name") else ""
        else:
            name = molecule.name
        blocks.append((name, _sdf_string([rdmol for _, rdmol in records])))

    if path is None:
        return "".join(text for _, text in blocks)

    path = Path(path)
    if not separate:
        _write_text("".join(text for _, text in blocks), path)
        return None

    # PurePosixPath('my/dir/mol.sdf.gz').suffixes -> ['.sdf', '.gz']
    suffixes = "".join(path.suffixes)
    prefix = path.name.replace(suffixes, "")
    extension = ".sdf.gz" if path.suffix == ".gz" else ".sdf"
    for name, text in blocks:
        _write_text(text, path.with_name(f"{prefix}_{name}{extension}"))
    return None


def _depiction_inputs(obj) -> tuple[bool, list[Chem.Mol], list[str]]:
    """Returns (is_grid, rdkit molecules, legends) for 2D depiction."""
    Conf, Mol, MolLibr = _classes()
    if isinstance(obj, MolLibr):
        return True, [mol.rdmol for mol in obj], [mol.name for mol in obj]
    if isinstance(obj, (Conf, Mol)):
        return False, [obj.rdmol], [obj.name]
    if isinstance(obj, Chem.Mol):
        return False, [obj], [""]
    raise TypeError(
        f"expects rdworks.Conf, Mol, MolLibr, or rdkit.Chem.Mol, got {type(obj).__name__}"
    )


def to_svg(
    obj,
    path: str | Path | None = None,
    width: int | None = None,
    height: int | None = None,
    legend: str = "",
    atom_index: bool = False,
    highlight_atoms: list | None = None,
    highlight_bonds: list | None = None,
    redraw: bool = False,
    coordgen: bool = False,
    optimize: bool = True,
    mols_per_row: int = 5,
) -> str:
    """Returns a 2D depiction in SVG format (a grid for a MolLibr).

    Examples:
        For Jupyter notebook, wrap the output with SVG:

        >>> from IPython.display import SVG
        >>> SVG(to_svg(libr[0]))

    Args:
        obj (Conf | Mol | MolLibr | Chem.Mol): input.
        path (str | Path | None, optional): also write to this .svg file. Defaults to None.
        width (int | None, optional): width (per molecule for a grid).
            Defaults to 300 (200 for a MolLibr).
        height (int | None, optional): height (per molecule for a grid).
            Defaults to 300 (200 for a MolLibr).
        legend (str, optional): legend of a single molecule. Defaults to ''.
        atom_index (bool, optional): whether to show atom index. Defaults to False.
        highlight_atoms (list | None, optional): atom(s) to highlight
            (a list of lists for a MolLibr). Defaults to None.
        highlight_bonds (list | None, optional): bond(s) to highlight
            (a list of lists for a MolLibr). Defaults to None.
        redraw (bool, optional): whether to redraw. Defaults to False.
        coordgen (bool, optional): whether to use coordgen. Defaults to False.
        optimize (bool, optional): whether to optimize a single-molecule SVG string.
            Defaults to True.
        mols_per_row (int, optional): molecules per row of a grid. Defaults to 5.

    Returns:
        str: SVG string.
    """
    is_grid, rdmols, legends = _depiction_inputs(obj)
    if is_grid:
        svg = render_matrix_grid(
            rdmols,
            legends,
            highlight_atoms=highlight_atoms,
            highlight_bonds=highlight_bonds,
            mols_per_row=mols_per_row,
            width=width or 200,
            height=height or 200,
            atom_index=atom_index,
            redraw=redraw,
            coordgen=coordgen,
            svg=True,
        )
    else:
        svg = render_svg(
            rdmols[0],
            width=width or 300,
            height=height or 300,
            legend=legend,
            atom_index=atom_index,
            highlight_atoms=highlight_atoms,
            highlight_bonds=highlight_bonds,
            redraw=redraw,
            coordgen=coordgen,
            optimize=optimize,
        )
    if path is not None:
        _write_text(svg, path)
    return svg


def to_png(
    obj,
    path: str | Path | None = None,
    width: int | None = None,
    height: int | None = None,
    legend: str = "",
    atom_index: bool = False,
    highlight_atoms: list | None = None,
    highlight_bonds: list | None = None,
    redraw: bool = False,
    coordgen: bool = False,
    trim: bool = True,
    mols_per_row: int = 5,
) -> Image.Image:
    """Returns a 2D depiction as a PIL image (a grid for a MolLibr).

    Args:
        obj (Conf | Mol | MolLibr | Chem.Mol): input.
        path (str | Path | None, optional): also save to this .png file. Defaults to None.
        width (int | None, optional): width (per molecule for a grid).
            Defaults to 300 (200 for a MolLibr).
        height (int | None, optional): height (per molecule for a grid).
            Defaults to 300 (200 for a MolLibr).
        legend (str, optional): legend of a single molecule. Defaults to ''.
        atom_index (bool, optional): whether to show atom index. Defaults to False.
        highlight_atoms (list | None, optional): atom(s) to highlight
            (a list of lists for a MolLibr). Defaults to None.
        highlight_bonds (list | None, optional): bond(s) to highlight
            (a list of lists for a MolLibr). Defaults to None.
        redraw (bool, optional): whether to redraw. Defaults to False.
        coordgen (bool, optional): whether to use coordgen. Defaults to False.
        trim (bool, optional): whether to trim white margins of a single molecule.
            Defaults to True.
        mols_per_row (int, optional): molecules per row of a grid. Defaults to 5.

    Returns:
        Image.Image: PIL image.
    """
    is_grid, rdmols, legends = _depiction_inputs(obj)
    if is_grid:
        img = render_matrix_grid(
            rdmols,
            legends,
            highlight_atoms=highlight_atoms,
            highlight_bonds=highlight_bonds,
            mols_per_row=mols_per_row,
            width=width or 200,
            height=height or 200,
            atom_index=atom_index,
            redraw=redraw,
            coordgen=coordgen,
            svg=False,
        )
    else:
        img = render_png(
            rdmols[0],
            width=width or 300,
            height=height or 300,
            legend=legend,
            atom_index=atom_index,
            highlight_atoms=highlight_atoms,
            highlight_bonds=highlight_bonds,
            redraw=redraw,
            coordgen=coordgen,
            trim=trim,
        )
    if path is not None:
        img.save(Path(path).as_posix())
    return img


def _table_rows(obj, name: str, smiles: str, confs: bool) -> list[dict]:
    """Returns table rows: molecules, or conformers if `confs` (or for a Conf)."""
    Conf, Mol, MolLibr = _classes()
    exclude = {"coord"}  # bulky conformer data

    def conf_row(conf, mol_smiles: str) -> dict:
        row = {name: conf.name, smiles: mol_smiles}
        row.update({k: v for k, v in conf.props.items() if k not in exclude | {name, smiles}})
        return row

    if isinstance(obj, Conf):
        return [conf_row(obj, Chem.MolToSmiles(Chem.RemoveHs(obj.rdmol)))]
    if isinstance(obj, Mol):
        if confs:
            return [conf_row(conf, obj.smiles) for conf in obj.confs]
        row = {name: obj.name, smiles: obj.smiles}
        row.update({k: v for k, v in obj.props.items() if k not in {name, smiles}})
        return [row]
    if isinstance(obj, MolLibr):
        return [row for mol in obj for row in _table_rows(mol, name, smiles, confs)]
    raise TypeError(f"expects rdworks.Conf, Mol, or MolLibr, got {type(obj).__name__}")


def to_df(
    obj, name: str = "name", smiles: str = "smiles", confs: bool = False
) -> pd.DataFrame:
    """Returns a pandas DataFrame of names, SMILES, and properties.

    Columns are `name`, `smiles`, then properties in the order first seen;
    missing values are None (NaN in numeric columns).

    Args:
        obj (Conf | Mol | MolLibr): input. A Conf gives one conformer row.
        name (str, optional): column name for name. Defaults to 'name'.
        smiles (str, optional): column name for SMILES. Defaults to 'smiles'.
        confs (bool, optional): one row per conformer with conformer properties
            instead of one row per molecule. Defaults to False.

    Returns:
        pd.DataFrame: pandas DataFrame.
    """
    rows = _table_rows(obj, name, smiles, confs)
    columns = list(dict.fromkeys([name, smiles] + [k for row in rows for k in row]))
    return pd.DataFrame({c: [row.get(c) for row in rows] for c in columns}, columns=columns)


def to_csv(obj, path: str | Path, confs: bool = False, decimals: int = 3) -> None:
    """Writes names, SMILES, and properties to a .csv file.

    Args:
        obj (Conf | Mol | MolLibr): input.
        path (str | Path): output filename or path.
        confs (bool, optional): one row per conformer. Defaults to False.
        decimals (int, optional): decimal places for float numbers. Defaults to 3.
    """
    to_df(obj, confs=confs).to_csv(path, index=False, float_format=f"%.{decimals}f")
