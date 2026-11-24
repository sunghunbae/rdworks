"""Tests for the shared writers in rdworks.io and the Conf/Mol/MolLibr to_* wrappers."""

import gzip
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from rdkit import Chem

import rdworks.io as rwio
from rdworks import Mol, MolLibr
from rdworks.units import angstrom2bohr


@pytest.fixture
def mol() -> Mol:
    """Acetaminophen with 3 reproducible conformers (20 atoms with H)."""
    return Mol("CC(=O)Nc1ccc(O)cc1", "apap").make_confs(n=3, seed=1).qed(["QED"])


@pytest.fixture
def conf(mol):
    return mol.confs[0]


@pytest.fixture
def libr(mol) -> MolLibr:
    """apap (3 conformers), ethanol (2 conformers), benzene (no conformer)."""
    ethanol = Mol("CCO", "ethanol").make_confs(n=2, seed=1)
    return MolLibr([mol, ethanol, Mol("c1ccccc1", "benzene")])


def _sdf_mols(text: str) -> list[Chem.Mol]:
    supplier = Chem.SDMolSupplier()
    supplier.SetData(text, removeHs=False)
    return [m for m in supplier]


##################################################
### package structure
##################################################


def test_import_writers_first_is_not_circular():
    # conf.py/mol.py/mollibr.py import rdworks.writers, so it must load on its own
    result = subprocess.run(
        [sys.executable, "-c", "import rdworks.writers as m; print(m.to_xyz.__module__)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rdworks.writers"


def test_io_reexports_readers_and_writers():
    # `rdworks.io` stays the public entry point for both halves
    import rdworks.readers
    import rdworks.writers

    for name in rwio.__all__:
        source = rdworks.readers if hasattr(rdworks.readers, name) else rdworks.writers
        assert getattr(rwio, name) is getattr(source, name), name


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("to_geometry", {}),
        ("to_xyz", {}),
        ("to_turbomole", {}),
        ("to_turbomole", {"bohr": True}),
        ("to_sdf", {}),
        ("to_svg", {}),
    ],
)
def test_conf_methods_delegate_to_io(conf, method, kwargs):
    assert getattr(conf, method)(**kwargs) == getattr(rwio, method)(conf, **kwargs)


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("to_geometry", {"index": 1}),
        ("to_xyz", {}),
        ("to_turbomole", {"index": 2}),
        ("to_sdf", {}),
        ("to_sdf", {"confs": True}),
        ("to_svg", {}),
    ],
)
def test_mol_methods_delegate_to_io(mol, method, kwargs):
    assert getattr(mol, method)(**kwargs) == getattr(rwio, method)(mol, **kwargs)


@pytest.mark.parametrize(
    "method, kwargs",
    [("to_xyz", {}), ("to_sdf", {}), ("to_sdf", {"confs": True}), ("to_svg", {})],
)
def test_mollibr_methods_delegate_to_io(libr, method, kwargs):
    assert getattr(libr, method)(**kwargs) == getattr(rwio, method)(libr, **kwargs)


def test_dataframe_methods_delegate_to_io(conf, mol, libr):
    pd.testing.assert_frame_equal(conf.to_df(), rwio.to_df(conf))
    pd.testing.assert_frame_equal(mol.to_df(confs=True), rwio.to_df(mol, confs=True))
    pd.testing.assert_frame_equal(libr.to_df(confs=True), rwio.to_df(libr, confs=True))


##################################################
### to_geometry / to_xyz / to_turbomole
##################################################


def test_to_geometry(conf):
    lines = conf.to_geometry().split("\n")
    assert len(lines) == conf.natoms
    for line, symbol, xyz in zip(lines, conf.symbols, conf.positions):
        fields = line.split()
        assert fields[0] == symbol
        assert np.allclose([float(v) for v in fields[1:]], xyz, atol=5e-4)  # 3 decimals


def test_mol_to_geometry_index(mol):
    assert mol.to_geometry() == mol.confs[0].to_geometry()
    assert mol.to_geometry(index=2) == mol.confs[2].to_geometry()


def test_single_structure_formats_reject_mollibr(libr):
    with pytest.raises(TypeError, match="single-structure"):
        rwio.to_geometry(libr)
    with pytest.raises(TypeError, match="single-structure"):
        rwio.to_turbomole(libr)


def _parse_xyz(text: str) -> list[tuple[str, list[str], np.ndarray]]:
    """Returns (comment, symbols, coordinates) per frame."""
    lines, frames, i = text.split("\n"), [], 0
    while i < len(lines):
        n = int(lines[i])
        atoms = [line.split() for line in lines[i + 2 : i + 2 + n]]
        frames.append(
            (lines[i + 1], [a[0] for a in atoms], np.array([[float(v) for v in a[1:]] for a in atoms]))
        )
        i += 2 + n
    return frames


def test_to_xyz_conf(conf):
    [(comment, symbols, xyz)] = _parse_xyz(conf.to_xyz())
    assert comment == "apap/1"  # conformer name
    assert symbols == conf.symbols
    assert np.allclose(xyz, conf.positions, atol=1e-12)


def test_to_xyz_multiframe(mol, libr):
    frames = _parse_xyz(mol.to_xyz())
    assert [f[0] for f in frames] == ["apap/1", "apap/2", "apap/3"]
    for (_, _, xyz), conf in zip(frames, mol.confs):
        assert np.allclose(xyz, conf.positions, atol=1e-12)
    # all conformers of all molecules; benzene has none
    names = [f[0] for f in _parse_xyz(libr.to_xyz())]
    assert names == ["apap/1", "apap/2", "apap/3", "ethanol/1", "ethanol/2"]


@pytest.mark.parametrize("filename", ["out.xyz", "out.xyz.gz"])
def test_to_xyz_path(tmp_path, mol, filename):
    text = mol.to_xyz(path=tmp_path / filename)
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(tmp_path / filename, "rt") as f:
        assert f.read() == text + "\n"


def test_to_xyz_no_conformer():
    with pytest.raises(ValueError, match="no 3D conformer"):
        Mol("CCO").to_xyz()


def test_to_turbomole_angstrom(conf):
    lines = conf.to_turbomole().split("\n")
    assert lines[0] == "$coord angs"
    assert lines[-1] == "$end"
    xyz = np.array([[float(v) for v in line.split()[:3]] for line in lines[1:-1]])
    assert [line.split()[3] for line in lines[1:-1]] == conf.symbols
    assert np.allclose(xyz, conf.positions, atol=1e-12)


def test_to_turbomole_bohr(conf):
    # `$coord` means Bohr: coordinates must be converted (previously written in Angstrom)
    lines = conf.to_turbomole(bohr=True).split("\n")
    assert lines[0] == "$coord"
    xyz = np.array([[float(v) for v in line.split()[:3]] for line in lines[1:-1]])
    assert np.allclose(xyz, conf.positions * angstrom2bohr, atol=1e-12)
    assert angstrom2bohr == pytest.approx(1.8897261246, abs=1e-9)


##################################################
### to_sdf
##################################################


def test_to_sdf_conf(conf):
    [m] = _sdf_mols(conf.to_sdf())
    assert m.GetProp("_Name") == "apap/1"
    assert m.GetNumAtoms() == conf.natoms  # 3D with hydrogens
    assert np.allclose(m.GetConformer().GetPositions(), conf.positions, atol=1e-3)
    assert float(m.GetProp("E_tot(kcal/mol)")) == pytest.approx(conf.props["E_tot(kcal/mol)"])
    assert not _sdf_mols(conf.to_sdf(props=False))[0].HasProp("E_tot(kcal/mol)")


def test_to_sdf_mol(mol):
    [m] = _sdf_mols(mol.to_sdf())  # 2D molecule with molecule props
    assert m.GetProp("_Name") == "apap"
    assert not m.GetConformer().Is3D()
    assert float(m.GetProp("QED")) == pytest.approx(mol.props["QED"])


def test_to_sdf_mol_confs(mol):
    mols = _sdf_mols(mol.to_sdf(confs=True))
    assert [m.GetProp("_Name") for m in mols] == ["apap/1", "apap/2", "apap/3"]
    for m, conf in zip(mols, mol.confs):
        assert m.GetConformer().Is3D()
        assert m.HasProp("QED")  # molecule props
        assert float(m.GetProp("E_rel(kcal/mol)")) == pytest.approx(conf.props["E_rel(kcal/mol)"])


def test_to_sdf_mollibr_string(libr):
    assert [m.GetProp("_Name") for m in _sdf_mols(libr.to_sdf())] == ["apap", "ethanol", "benzene"]
    names = [m.GetProp("_Name") for m in _sdf_mols(libr.to_sdf(confs=True))]
    assert names == ["apap/1", "apap/2", "apap/3", "ethanol/1", "ethanol/2"]


def test_to_sdf_record_counter_restarts_per_molecule(libr):
    # one SDWriter per molecule, as in the original per-molecule output
    blocks = libr.to_sdf().split("$$$$\n")[:-1]
    assert all("(1)" in block and "(2)" not in block for block in blocks)


@pytest.mark.parametrize("filename", ["libr.sdf", "libr.sdf.gz"])
def test_to_sdf_path(tmp_path, libr, filename):
    assert libr.to_sdf(tmp_path / filename, confs=True) is None
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(tmp_path / filename, "rt") as f:
        assert f.read() == libr.to_sdf(confs=True)


@pytest.mark.parametrize("filename, extension", [("out.sdf", ".sdf"), ("out.sdf.gz", ".sdf.gz")])
def test_to_sdf_separate(tmp_path, libr, filename, extension):
    libr.to_sdf(tmp_path / filename, confs=True, separate=True)
    written = sorted(p.name for p in tmp_path.iterdir())
    # every molecule gets its own file, even benzene without conformers
    assert written == sorted(f"out_{name}{extension}" for name in ("apap", "ethanol", "benzene"))
    opener = gzip.open if extension == ".sdf.gz" else open
    with opener(tmp_path / f"out_ethanol{extension}", "rt") as f:
        assert [m.GetProp("_Name") for m in _sdf_mols(f.read())] == ["ethanol/1", "ethanol/2"]


def test_conf_and_mol_to_sdf_path(tmp_path, conf, mol):
    assert conf.to_sdf(path=tmp_path / "conf.sdf") is None
    assert (tmp_path / "conf.sdf").read_text() == conf.to_sdf()
    assert mol.to_sdf(confs=True, path=tmp_path / "mol.sdf") is None
    assert (tmp_path / "mol.sdf").read_text() == mol.to_sdf(confs=True)


##################################################
### to_svg / to_png
##################################################


def test_to_svg_single_and_grid(conf, mol, libr):
    for svg in (conf.to_svg(), mol.to_svg(), libr.to_svg()):
        assert svg.startswith("<?xml")
        assert "<svg" in svg
    assert 'width="300px"' in mol.to_svg()  # single-molecule default


def test_to_svg_path(tmp_path, mol, libr):
    assert (tmp_path / "m.svg").exists() is False
    svg = mol.to_svg(path=tmp_path / "m.svg")
    assert (tmp_path / "m.svg").read_text() == svg
    grid = libr.to_svg(path=tmp_path / "grid.svg")
    assert (tmp_path / "grid.svg").read_text() == grid


def test_to_png_single(mol, tmp_path):
    assert mol.to_png(trim=False).size == (300, 300)
    assert mol.to_png(width=200, height=150, trim=False).size == (200, 150)
    img = mol.to_png(path=tmp_path / "m.png")
    assert isinstance(img, Image.Image)
    assert Image.open(tmp_path / "m.png").size == img.size


@pytest.mark.parametrize("mols_per_row, size", [(1, (200, 600)), (2, (400, 400)), (3, (600, 200))])
def test_to_png_grid(libr, mols_per_row, size):
    assert libr.to_png(mols_per_row=mols_per_row).size == size


def test_mollibr_to_png_filename_saves_and_returns_image(libr, tmp_path):
    img = libr.to_png(tmp_path / "grid.png")
    assert isinstance(img, Image.Image)
    assert np.array_equal(np.array(Image.open(tmp_path / "grid.png")), np.array(img))


def test_writers_accept_rdkit_mol(conf):
    rdmol = Chem.Mol(conf.rdmol)
    rdmol.SetProp("_Name", "plain")
    assert _parse_xyz(rwio.to_xyz(rdmol))[0][0] == "plain"
    assert _sdf_mols(rwio.to_sdf(rdmol))[0].GetProp("_Name") == "plain"
    assert "<svg" in rwio.to_svg(Chem.MolFromSmiles("CCO"))


def test_writers_reject_unsupported_type():
    for writer in (rwio.to_xyz, rwio.to_sdf, rwio.to_svg, rwio.to_df):
        with pytest.raises(TypeError):
            writer("CCO")


##################################################
### to_df / to_csv
##################################################


def test_to_df_conf(conf):
    df = conf.to_df()
    assert len(df) == 1
    assert list(df.columns[:2]) == ["name", "smiles"]
    assert df.loc[0, "name"] == "apap/1"
    assert df.loc[0, "smiles"] == "CC(=O)Nc1ccc(O)cc1"  # without explicit hydrogens
    assert df.loc[0, "E_rel(kcal/mol)"] == conf.props["E_rel(kcal/mol)"]


def test_to_df_mol(mol):
    df = mol.to_df()
    assert df.shape[0] == 1
    assert df.loc[0, "QED"] == mol.props["QED"]
    df_confs = mol.to_df(confs=True)
    assert list(df_confs["name"]) == ["apap/1", "apap/2", "apap/3"]
    assert list(df_confs["smiles"]) == [mol.smiles] * 3
    assert "QED" not in df_confs.columns  # conformer rows hold conformer props


def test_to_df_mollibr(libr):
    df = libr.to_df()
    assert list(df["name"]) == ["apap", "ethanol", "benzene"]
    assert list(df.columns[:2]) == ["name", "smiles"]
    assert df.loc[0, "QED"] == libr[0].props["QED"]
    assert pd.isna(df.loc[1, "QED"])  # missing value (NaN in a numeric column)
    df_confs = libr.to_df(confs=True)
    assert list(df_confs["name"]) == ["apap/1", "apap/2", "apap/3", "ethanol/1", "ethanol/2"]


def test_to_df_column_order_is_deterministic(libr):
    # first-seen order, not set order (which varied between Python sessions)
    columns = list(libr.to_df().columns)
    expected = ["name", "smiles"] + list(dict.fromkeys(k for mol in libr for k in mol.props))
    assert columns == expected


def test_to_df_custom_column_names(libr):
    df = libr.to_df(name="ID", smiles="SMILES")
    assert list(df.columns[:2]) == ["ID", "SMILES"]
    assert list(df["ID"]) == ["apap", "ethanol", "benzene"]


@pytest.mark.parametrize("confs", [False, True])
def test_to_csv(tmp_path, libr, confs):
    path = tmp_path / "out.csv"
    libr.to_csv(path, confs=confs, decimals=2)
    df = pd.read_csv(path)
    expected = libr.to_df(confs=confs)
    assert list(df.columns) == list(expected.columns)
    assert list(df["name"]) == list(expected["name"])
    if confs:
        assert np.allclose(df["E_tot(kcal/mol)"], expected["E_tot(kcal/mol)"].astype(float), atol=0.005)
        assert all(len(v.split(".")[1]) <= 2 for v in path.read_text().splitlines()[1].split(",") if "." in v and v.replace(".", "").replace("-", "").isdigit())


def test_conf_and_mol_to_csv(tmp_path, conf, mol):
    conf.to_csv(tmp_path / "conf.csv")
    assert list(pd.read_csv(tmp_path / "conf.csv")["name"]) == ["apap/1"]
    mol.to_csv(tmp_path / "mol.csv", confs=True)
    assert list(pd.read_csv(tmp_path / "mol.csv")["name"]) == ["apap/1", "apap/2", "apap/3"]
