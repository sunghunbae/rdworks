from rdkit import Chem
from rdworks import Mol
from rdworks.data.test.example import drugs


drug_smiles = list(drugs.values())
drug_names = list(drugs.keys())


def test_init_mol():
    mol = Mol(drug_smiles[0], drug_names[0])
    assert mol.num_confs == 0
    assert mol.name == drug_names[0]
    rdmol = Chem.MolFromSmiles(drug_smiles[0])
    rdmol.SetProp("_Name", drug_names[0])
    mol = Mol(rdmol, drug_names[0])
    assert mol.rdmol.GetProp("_Name") == drug_names[0]
    assert mol.name == drug_names[0]


def test_has_substr():
    mol = Mol("c1cc(C(=O)O)c(OC(=O)C)cc1")
    assert mol.has_substr("OC(=O)C")


def test_optimize_confs():
    mol = Mol("N=C1O[C@@H](CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    mol = (
        mol.make_confs()
        .compute(max_workers=4, chunksize=1, progress=True)
        .optimize_confs(calculator="MMFF94")
    )
