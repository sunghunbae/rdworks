"""Tests for rdworks.std (standardization, desalting, InChIKey, 2D cleanup)."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rdworks import Mol
from rdworks.std import (
    clean_2d,
    desalt_smiles,
    generate_inchi_key,
    neutralize_atoms,
    standardize,
    standardize_smiles,
)


def smi(rdmol: Chem.Mol) -> str:
    return Chem.MolToSmiles(rdmol)


def embedded_ethanol(num_confs: int = 1) -> Chem.Mol:
    """Ethanol with explicit hydrogens and `num_confs` 3D conformers."""
    rdmol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(rdmol, num_confs, randomSeed=1)
    return rdmol


##################################################
### generate_inchi_key
##################################################


def test_generate_inchi_key():
    key = generate_inchi_key(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))  # aspirin
    assert key == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert len(key) == 27


def test_generate_inchi_key_unavailable():
    # InChI cannot represent dummy atoms; an empty string (not None) is returned
    assert generate_inchi_key(Chem.MolFromSmiles("*C")) == ""


def test_mol_equality_without_inchi_key():
    # without an InChIKey, Mol equality falls back to canonical SMILES
    assert Mol("*C") != Mol("*CC")
    assert Mol("*C") == Mol("C*")
    assert len({Mol("*C"), Mol("*CC")}) == 2


##################################################
### desalt_smiles
##################################################


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("OC(=O)c1ccccc1.[Na+]", "O=C(O)c1ccccc1"),  # canonical SMILES
        ("[Na+].[O-]C(=O)c1ccccc1", "O=C([O-])c1ccccc1"),  # fragment order
        ("C[C@H](N)C(=O)O.Cl", "C[C@H](N)C(=O)O"),  # stereo kept
        ("C1.C1", "CC"),  # ring closure across '.' is a single molecule
        ("c1ccccc1", "c1ccccc1"),  # nothing to remove
    ],
)
def test_desalt_smiles(smiles, expected):
    desalted_smiles, desalted_rdmol = desalt_smiles(smiles)
    assert desalted_smiles == expected
    assert smi(desalted_rdmol) == expected


def test_desalt_smiles_tie_keeps_first_fragment():
    assert desalt_smiles("CCO.OCC")[0] == "CCO"


@pytest.mark.parametrize("smiles", ["CCO.xyz", "xyz", "C(C"])
def test_desalt_smiles_invalid(smiles):
    with pytest.raises(ValueError):
        desalt_smiles(smiles)


def test_mol_salt_and_free_form_have_same_smiles():
    # MolLibr.unique() keys on Mol.smiles
    assert Mol("OC(=O)c1ccccc1.[Na+]").smiles == Mol("OC(=O)c1ccccc1").smiles


##################################################
### standardize_smiles / standardize
##################################################


def test_standardize_smiles():
    assert standardize_smiles("C[N+](=O)[O-]") == "C[N+](=O)[O-]"
    # metal disconnected and reionized
    assert standardize_smiles("[Na]OC(=O)c1ccccc1") == "O=C([O-])c1ccccc1.[Na+]"


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("OC(=O)c1ccccc1.[Na+]", "O=C(O)c1ccccc1"),  # parent fragment
        ("CC(=O)[O-].[Na+]", "CC(=O)O"),  # neutralized
        ("C[NH3+]", "CN"),  # neutralized
        ("Oc1ccccn1", "O=c1cccc[nH]1"),  # canonical tautomer
    ],
)
def test_standardize(smiles, expected):
    assert smi(standardize(smiles)) == expected
    assert smi(standardize(Chem.MolFromSmiles(smiles))) == expected


def test_standardize_does_not_modify_input():
    rdmol = Chem.MolFromSmiles("C[NH3+]")
    standardize(rdmol)
    assert smi(rdmol) == "C[NH3+]"


def test_standardize_invalid_input():
    with pytest.raises(ValueError):
        standardize("xyz")
    with pytest.raises(TypeError):
        standardize(123)


##################################################
### neutralize_atoms
##################################################


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("CC(=O)[O-]", "CC(=O)O"),
        ("C[NH3+]", "CN"),
        # unlike Uncharger, neutralizes the carboxylate next to a quaternary N
        ("C[N+](C)(C)CCC([O-])=O", "C[N+](C)(C)CCC(=O)O"),
        ("C[N+](=O)[O-]", "C[N+](=O)[O-]"),  # charge-separated nitro kept
        ("CCO", "CCO"),  # nothing to neutralize
    ],
)
def test_neutralize_atoms(smiles, expected):
    rdmol = Chem.MolFromSmiles(smiles)
    before = smi(rdmol)
    assert smi(neutralize_atoms(rdmol)) == expected
    assert smi(rdmol) == before  # input not modified


##################################################
### clean_2d
##################################################


def test_clean_2d_from_smiles():
    mol, conformers = clean_2d(Chem.MolFromSmiles("CCO"))
    assert smi(mol) == "CCO"
    assert not mol.GetConformer().Is3D()  # 2D depiction coordinates
    assert conformers == []


def test_clean_2d_isotope():
    rdmol = Chem.MolFromSmiles("[2H]C([2H])([2H])O")
    assert smi(clean_2d(rdmol)[0]) == "CO"  # reset, then removed as H
    assert smi(clean_2d(rdmol, reset_isotope=False)[0]) == "[2H]C([2H])([2H])O"


def test_clean_2d_hydrogens():
    rdmol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert clean_2d(rdmol)[0].GetNumAtoms() == 3
    assert clean_2d(rdmol, remove_H=False)[0].GetNumAtoms() == 9
    assert rdmol.GetNumAtoms() == 9  # input not modified


@pytest.mark.parametrize("num_confs", [1, 3])
def test_clean_2d_returns_3d_conformers(num_confs):
    rdmol = embedded_ethanol(num_confs)
    mol, conformers = clean_2d(rdmol)
    assert not mol.GetConformer().Is3D()
    assert len(conformers) == num_confs
    for i, conf_mol in enumerate(conformers):
        assert conf_mol.GetNumConformers() == 1
        assert conf_mol.GetNumAtoms() == rdmol.GetNumAtoms()  # hydrogens kept
        assert np.allclose(
            conf_mol.GetConformer().GetPositions(),
            rdmol.GetConformer(i).GetPositions(),
        )
    assert rdmol.GetNumConformers() == num_confs  # input not modified


def test_clean_2d_skips_3d_conformers_without_hydrogens():
    rdmol = Chem.RemoveHs(embedded_ethanol())
    assert clean_2d(rdmol)[1] == []


@pytest.mark.parametrize("num_confs", [1, 3])
def test_mol_from_3d_rdmol_keeps_conformers(num_confs):
    rdmol = embedded_ethanol(num_confs)
    m = Mol(rdmol, "ethanol")
    assert m.num_confs == num_confs
    for i, conf in enumerate(m.confs):
        assert conf.rdmol.GetNumAtoms() == rdmol.GetNumAtoms()
        assert np.allclose(
            conf.rdmol.GetConformer().GetPositions(),
            rdmol.GetConformer(i).GetPositions(),
        )


def test_mol_from_3d_molblock():
    molblock = Chem.MolToMolBlock(embedded_ethanol())
    assert Mol().from_molblock(molblock).num_confs == 0  # MolFromMolBlock drops H
    with_h = Chem.MolFromMolBlock(molblock, removeHs=False)
    assert Mol(with_h).num_confs == 1
