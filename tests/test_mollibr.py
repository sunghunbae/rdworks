import copy
import math
from importlib.resources import files, as_file

from rdkit import Chem
from rdworks import Mol, MolLibr
from rdworks.descriptor import rd_descriptor_f
from rdworks.data.test.example import drugs


drug_smiles = list(drugs.values())
drug_names = list(drugs.keys())


def test_init_mollibr():
    libr = MolLibr(drug_smiles[:5], drug_names[:5])
    assert libr.count() == 5
    libr = MolLibr([Chem.MolFromSmiles(_) for _ in drug_smiles[5:10]], drug_names[5:10])
    assert libr.count() == 5
    libr = MolLibr(
        [Mol(smi, name) for smi, name in zip(drug_smiles[10:15], drug_names[10:15])]
    )
    assert libr.count() == 5


def test_operators():
    libr = MolLibr(drug_smiles, drug_names)
    assert libr.count() == 22

    # other library has 5 overlapping molecules
    other = MolLibr(drug_smiles[5:10], drug_names[5:10])
    assert other.count() == 5

    # index or slice and equality
    assert libr[10] == Mol(drug_smiles[10], drug_names[10])
    assert libr[5:10] == other

    assert (libr + other).count() == 27  # appended, 5 duplicates
    assert (libr - other).count() == 17
    assert (libr & other).count() == 5
    assert libr.count() == 22  # libr object is unchanged.

    # libr will be changed by +=, -=, &= operators
    libr += other
    assert libr.count() == 27  # appended, 5 duplicates

    libr -= other
    assert libr.count() == 17  # now libr has no overlap with other

    libr &= other
    assert libr.count() == 0  # previous operator removed common molecules


def test_copy():
    libr1 = MolLibr(drug_smiles[:5], drug_names[:5])
    for i in range(5):
        libr1.libr[i].rdmol.SetProp("_Name", f"_Name_{i}")
        libr1.libr[i].rdmol.SetProp("Property", f"Property_{i}")
    libr2 = copy.deepcopy(libr1)  # copied
    for i in range(5):
        assert libr2.libr[i].rdmol.GetProp("_Name") == f"_Name_{i}"
        assert libr2.libr[i].rdmol.GetProp("Property") == f"Property_{i}"
        assert libr1.libr[i].smiles == libr2.libr[i].smiles


def test_unique():
    libr = MolLibr(
        drug_smiles[-4:]
        + ["N[C@@H](CC(=O)N1CCN2C(C1)=NN=C2C(F)(F)F)CC1=CC(F)=C(F)C=C1F"],
        drug_names[-4:] + ["Januvia"],
    )  # = sitagliptin
    assert libr.count() == 5
    libr_unique = libr.unique()
    assert libr_unique.count() == 4
    assert ["Januvia"] in [_.props["aka"] for _ in libr_unique]


def test_drop():
    libr = MolLibr(drug_smiles, drug_names, progress=False)
    not_druglike_names = [
        "Atorvastatin",
        "Sofosbuvir",
        "Rifampin",
        "Cefdinir",
        "Famotidine",
        "Atovaquone",
        "Chlorprothixene",
        "Methixene",
        "Ethopropazine",
    ]
    cnsmpo_compliant_names = [
        "Sitagliptin",
        "Simvastatin",
        "Paroxetine",
        "Leflunomide",
        "Granisetron",
        "Molindone",
        "Cimetidine",
        "Fluconazole",
        "Linezolid",
    ]

    obj = libr.drop()

    obj = libr.drop("ZINC_druglike")
    assert obj.count() == 9
    assert set([_.name for _ in obj]) == set(not_druglike_names)

    obj = libr.drop("~ZINC_druglike")
    assert obj.count() == 13
    assert set([_.name for _ in obj]) == set(drug_names) - set(not_druglike_names)

    # Keep CNS compliant compounds
    # Below three drop() functions have the same effect
    # and obj1, obj2, and obj3 should be identical
    obj1 = libr.drop("CNS", invert=True)
    assert obj1.count() == 9
    assert set([_.name for _ in obj1]) == set(cnsmpo_compliant_names)

    obj2 = libr.drop("~CNS")
    assert obj2.count() == 9
    assert set([_.name for _ in obj2]) == set(cnsmpo_compliant_names)

    with as_file(files("rdworks.data.test") / "cns.xml") as xml_path:
        obj3 = libr.drop(xml_path, invert=True)
        assert obj3.count() == 9
        assert set([_.name for _ in obj3]) == set(cnsmpo_compliant_names)


def test_similar():
    libr = MolLibr(drug_smiles, drug_names, progress=False)
    query = Mol(
        "[H][C@@]1(CC[C@]([H])(C2=C(F)C=C(F)C(F)=C2)[C@@]([H])(N)C1)N1CCN2C(C1)=NN=C2C(F)(F)F",
        "DB07072",
    )
    assert libr.similar(query, threshold=0.2).count() == 1
    query = libr[drug_names.index("Methixene")]
    sim = libr.similar(query, threshold=0.2)
    sim_expected = ["Pergolide", "Methixene", "Ethopropazine"]
    sim_names = [_.name for _ in sim]
    assert set(sim_names) == set(sim_expected)


def test_nnp_ready():
    libr = MolLibr(drug_smiles, drug_names)
    libr_subset = libr.nnp_ready("ANI-2x", progress=False)
    assert libr_subset.count() == 21


def test_qed():
    libr = MolLibr(drug_smiles[:3], drug_names[:3]).qed(progress=False)
    i = drug_names.index("Acetaminophen")
    assert math.isclose(libr[i].props["MolWt"], 151.165, rel_tol=1.0e-6, abs_tol=1.0e-3)
    assert math.isclose(libr[i].props["QED"], 0.5950, rel_tol=1.0e-6, abs_tol=1.0e-3)
    # calculate all available descriptors:
    # "MolWt", "TPSA", "LogP", "HBA", "HBD", "QED", "LipinskiHBA", "LipinskiHBD",
    # "HAC", "RotBonds", "RingCount", "Hetero", "FCsp3"
    libr = MolLibr(drug_smiles, drug_names).qed(
        properties=[k for k in rd_descriptor_f], progress=False
    )
