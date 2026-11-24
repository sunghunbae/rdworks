"""Writers shared by `Conf`, `Mol`, and `MolLibr`.

The writers (`to_geometry`, `to_xyz`, `to_turbomole`, `to_sdf`, `to_png`, `to_svg`,
`to_df`, `to_csv`) accept `rdworks.Conf`, `rdworks.Mol`, `rdworks.MolLibr`, or
`rdkit.Chem.Mol` objects and are the single implementation behind the `to_*`
methods of `Conf`, `Mol`, and `MolLibr`.

`rdworks.conf`, `rdworks.mol`, and `rdworks.mollibr` import this module, so it must
not import them, not even inside functions. Objects are dispatched on the
`_rdworks_kind` class attribute ("conf", "mol", "libr") instead of `isinstance`.
"""

from __future__ import annotations

import gzip
from io import StringIO
from pathlib import Path

import pandas as pd
from PIL import Image
from rdkit import Chem

from rdworks.units import angstrom2bohr
from rdworks.view import render_matrix_grid, render_png, render_svg


def _kind(obj) -> str | None:
    """Returns "conf", "mol", or "libr" for rdworks objects, None otherwise.

    `Conf`, `Mol`, and `MolLibr` set the `_rdworks_kind` class attribute, so this
    module can dispatch on them without importing them.
    """
    return getattr(type(obj), "_rdworks_kind", None)


def _structures(obj, index: int | None = None) -> list[tuple[str, Chem.Mol]]:
    """Returns 3D structures of `obj` as a list of (name, rdkit.Chem.Mol).

    Args:
        obj (Conf | Mol | MolLibr | Chem.Mol): input.
        index (int | None): conformer index for a Mol; None means all conformers.

    Raises:
        TypeError: if `obj` is not a supported type.
        ValueError: if `obj` has no 3D conformer.
    """
    if _kind(obj) == "conf":
        structures = [(obj.name, obj.rdmol)]
    elif _kind(obj) == "mol":
        confs = obj.confs if index is None else [obj.confs[index]]
        structures = [(conf.name, conf.rdmol) for conf in confs]
    elif _kind(obj) == "libr":
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
    if _kind(obj) == "libr":
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

    def with_props(rdmol: Chem.Mol, name: str, *dicts: dict) -> Chem.Mol:
        rdmol = Chem.Mol(rdmol)
        rdmol.SetProp("_Name", name)
        if props:
            for d in dicts:
                for k, v in d.items():
                    rdmol.SetProp(k, str(v))
        return rdmol

    if _kind(obj) == "conf":
        return [(obj.name, with_props(obj.rdmol, obj.name, obj.props))]
    if _kind(obj) == "mol":
        if confs:
            # molecule props, then conformer props
            return [
                (obj.name, with_props(conf.rdmol, conf.name, obj.props, conf.props))
                for conf in obj.confs
            ]
        return [(obj.name, with_props(obj.rdmol, obj.name, obj.props))]
    if _kind(obj) == "libr":
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
    # one SDWriter per molecule, so the property record counters, e.g. `>  <QED>  (1)`,
    # restart for each molecule as in the per-molecule Mol.to_sdf() output
    molecules = list(obj) if _kind(obj) == "libr" else [obj]
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
    if _kind(obj) == "libr":
        return True, [mol.rdmol for mol in obj], [mol.name for mol in obj]
    if _kind(obj) in ("conf", "mol"):
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
    exclude = {"coord"}  # bulky conformer data

    def conf_row(conf, mol_smiles: str) -> dict:
        row = {name: conf.name, smiles: mol_smiles}
        row.update({k: v for k, v in conf.props.items() if k not in exclude | {name, smiles}})
        return row

    if _kind(obj) == "conf":
        return [conf_row(obj, Chem.MolToSmiles(Chem.RemoveHs(obj.rdmol)))]
    if _kind(obj) == "mol":
        if confs:
            return [conf_row(conf, obj.smiles) for conf in obj.confs]
        row = {name: obj.name, smiles: obj.smiles}
        row.update({k: v for k, v in obj.props.items() if k not in {name, smiles}})
        return [row]
    if _kind(obj) == "libr":
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
