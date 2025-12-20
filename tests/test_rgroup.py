from rdworks.scaffold import scaffold_tree
from rdworks.rgroup import expand_rgroup
from rdworks.data.test.example import drugs
from rdworks import MolLibr, MatchedSeries


def test_expand_rgroup():
    X = [
        "[*]C#N",
        "[*]C(O)=O",
        "[*]CO",
        "[*]COC",
        "[*]C(NC)=O",
        "[*]CNC(C)=O",
        "[*]CC=C",
        "[*][H]",
    ]  # (8)
    Y = ["[*][H]", "[*]O", "[*]OC", "[*]CC(F)(F)F", "[*]OCCOC"]  # (5)
    core = "[*:1]-c1ccc2ccn(-[*:2])c2c1"
    libr = expand_rgroup(core=core, r={1: X, 2: Y}, prefix="RGX", progress=False)
    assert libr.count() == 40  # 8x5


def test_scaffold_tree():
    drug_smiles = list(drugs.values())
    drug_names = list(drugs.keys())
    libr = MolLibr(drug_smiles[:3], drug_names[:3])
    for mol in libr:
        adhoc_libr = MolLibr(scaffold_tree(mol.rdmol)).rename(prefix=mol.name)


def test_MatchedSeries():
    # https://greglandrum.github.io/rdkit-blog/posts/2023-01-09-rgd-tutorial.html
    X = [
        "[*]C#N",
        "[*]C(O)=O",
        "[*]CO",
        "[*]COC",
        "[*]C(NC)=O",
        "[*]CNC(C)=O",
        "[*]CC=C",
        "[*][H]",
    ]  # (8)
    Y = ["[*][H]", "[*]O", "[*]OC", "[*]CC(F)(F)F", "[*]OCCOC"]  # (5)
    core = "[*:1]-c1ccc2ccn(-[*:2])c2c1"
    libr = expand_rgroup(core=core, r={1: X, 2: Y}, prefix="RGX", progress=False)
    series = MatchedSeries(libr, sort_props=["QED", "HAC"])
    assert series.count() == 10
