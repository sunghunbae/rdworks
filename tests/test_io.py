import tempfile
import math

from importlib.resources import files, as_file
from pathlib import Path

from rdkit import Chem

from rdworks.data.test.example import drugs
from rdworks import MolLibr
from rdworks.io import merge_csv, read_csv, read_sdf, read_smi, read_mae


drug_names = list(drugs.keys())
drug_smiles = list(drugs.values())


def test_read_sdf():
    with (
        as_file(files("rdworks.data.test") / "cdk2.sdf") as path1,
        as_file(files("rdworks.data.test") / "cdk2.sdf.gz") as path2,
    ):
        libr1 = read_sdf(path1, progress=False)
        assert libr1.count() == 47, "failed to read .sdf file"
        libr2 = read_sdf(path2, progress=False)
        assert libr2.count() == 47, "failed to read .sdf.gz file"
        assert libr1 == libr2


def test_read_smi():
    with (
        as_file(files("rdworks.data.test") / "cdk2.smi") as path1,
        as_file(files("rdworks.data.test") / "cdk2.smi.gz") as path2,
    ):
        libr1 = read_smi(path1, progress=False)
        assert libr1.count() == 47, "failed to read .smi file"
        libr2 = read_smi(path2, progress=False)
        assert libr2.count() == 47, "failed to read .smi.gz file"
        assert libr1 == libr2


def test_read_mae():
    with as_file(files("rdworks.data.test") / "sj506.mae") as path:
        libr = read_mae(path)
        print(libr.count())


def test_merge_csv():
    with as_file(files("rdworks.data.test") / "drugs_20.csv") as path:
        # drugs_20.csv is a subset of drugs dictionary
        libr = MolLibr(drug_smiles, drug_names)
        assert libr.count() == 22
        libr = merge_csv(libr, path, on="name")
        assert libr.count() == 22
        with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
            workdir = Path(temp_dir)
            libr.to_csv(workdir / "test_merge_csv.csv")
            assert libr.count() == 22


def test_to_csv():
    libr1 = MolLibr(drug_smiles, drug_names, progress=False)
    with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
        workdir = Path(temp_dir)
        libr1.qed(progress=False).to_csv(workdir / "test_to_csv.csv")
        libr2 = read_csv(
            workdir / "test_to_csv.csv", smiles="smiles", name="name", progress=False
        )
        assert libr1 == libr2


def test_to_png():
    libr = MolLibr(drug_smiles, drug_names, progress=False)
    with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
        workdir = Path(temp_dir)
        libr.to_png(workdir / "test_to_png.png")
        libr.to_png(workdir / "test_to_png_with_index.png", atom_index=True)


def test_to_svg():
    libr = MolLibr(drug_smiles, drug_names, progress=False)
    with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
        workdir = Path(temp_dir)
        with open(workdir / "test_to_svg.svg", "w") as svg:
            svg.write(libr.to_svg())
        head = libr.to_svg()[:100]
        assert head.startswith("<?xml") and ("<svg" in head)


def test_to_smi():
    libr = MolLibr(drug_smiles, drug_names, progress=False)
    with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
        workdir = Path(temp_dir)
        libr.to_smi(workdir / "test_to_smi.smi.gz")
        libr.to_smi(workdir / "test_to_smi.smi")


def test_to_sdf():
    libr = MolLibr(drug_smiles, drug_names, progress=False)
    with tempfile.TemporaryDirectory() as temp_dir:  # tmpdir is a string
        workdir = Path(temp_dir)
        libr.to_sdf(workdir / "test_to_sdf.sdf.gz")
        libr.to_sdf(workdir / "test_to_sdf.sdf")
        libr.qed().to_sdf(
            workdir / "test_to_sdf_with_qed.sdf"
        )  # QED and other properties should be here
        supp = Chem.SDMolSupplier((workdir / "test_to_sdf_with_qed.sdf").as_posix())
        for m, mol in zip(supp, libr):
            assert math.isclose(
                float(m.GetProp("MolWt")),
                mol.props["MolWt"],
                rel_tol=1e-6,
                abs_tol=1e-3,
            )
            assert math.isclose(
                float(m.GetProp("QED")), mol.props["QED"], rel_tol=1e-6, abs_tol=1e-3
            )
