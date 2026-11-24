from rdworks import Mol, MolLibr
from rdworks.complete import complete_stereoisomers, complete_tautomers
from rdworks.data.test.example import drugs

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

import logging

drug_smiles = list(drugs.values())
drug_names = list(drugs.keys())


logger = logging.getLogger(__name__)


def test_rdkit():
    logger.info("Single chiral center (bromochlorofluoromethane)")
    mol = Chem.MolFromSmiles('FC(Cl)Br')
    isomers = list(EnumerateStereoisomers(mol))
    assert len(isomers) == 2, "There are two stereoisomers of bromochlorofluoromethane"
    for i, iso in enumerate(isomers, 1):
        logger.info(f"  {i}. {Chem.MolToSmiles(iso)}")

    logger.info("Double bond stereoisomers (2-butene)")
    mol = Chem.MolFromSmiles('CC=CC')
    isomers = list(EnumerateStereoisomers(mol))
    assert len(isomers) == 2, "There are two stereoisomers of 2-butene"
    for i, iso in enumerate(isomers, 1):
        logger.info(f"  {i}. {Chem.MolToSmiles(iso)}")
    
    logger.info("Ring double bond stereoisomers (cyclohexene, n=6)")
    mol = Chem.MolFromSmiles('C1CCC=CC1')
    isomers = list(EnumerateStereoisomers(mol))
    assert len(isomers) == 1, "cannot enumerate for small rings (<8 atoms)"
    for i, iso in enumerate(isomers, 1):
        logger.info(f"  {i}. {Chem.MolToSmiles(iso)}")

    logger.info("Ring double bond stereoisomers (cycloheptene, n=7)")
    mol = Chem.MolFromSmiles('C1CCCC=CC1')
    isomers = list(EnumerateStereoisomers(mol))
    assert len(isomers) == 1, "cannot enumerate for small rings (<8 atoms)"
    for i, iso in enumerate(isomers, 1):
        logger.info(f"  {i}. {Chem.MolToSmiles(iso)}")

    logger.info("Ring double bond stereoisomers (cyclooctene, n=8)")
    mol = Chem.MolFromSmiles('C1CCCCCC=CC1')
    isomers = list(EnumerateStereoisomers(mol))
    assert len(isomers) == 2, "can enumerate for macrocycles (≥8 atoms)"
    for i, iso in enumerate(isomers, 1):
        logger.info(f"  {i}. {Chem.MolToSmiles(iso)}")

    logger.info("Two chiral centers (2,3-dibromobutane)")
    # There are three stereoisomers of 2,3-dibromobutane, not four, 
    # because its symmetrical structure leads to one achiral (meso) compound, 
    # which is superimposable on its mirror image, 
    # reducing the total count from the theoretical \(2^{n}=4\). 
    # These three are a pair of enantiomers (chiral) and a meso compound (achiral), 
    # consisting of (R,R), (S,S) enantiomers, and the (R,S) or (S,R) meso form. 
    mol = Chem.MolFromSmiles('CC(Br)C(Br)C')
    isomers = list(EnumerateStereoisomers(mol))
    assert len(isomers) == 3, "There are three stereoisomers of 2,3-dibromobutane"
    for i, iso in enumerate(isomers, 1):
        logger.info(f"  {i}. {Chem.MolToSmiles(iso)}")

    logger.info("Chiral center + double bond")
    mol = Chem.MolFromSmiles('BrC=CC(F)Cl')
    isomers = list(EnumerateStereoisomers(mol))
    assert len(isomers) == 4, "There are four stereoisomers (2 chiral + 2 E/Z)"
    for i, iso in enumerate(isomers, 1):
        logger.info(f"  {i}. {Chem.MolToSmiles(iso)}")

    # logger.info("5: Using tryEmbedding option (ring system)")
    # mol = Chem.MolFromSmiles('BrC=CC1OC(C2)(F)C2(Cl)C1')
    # opts = StereoEnumerationOptions(tryEmbedding=True)
    # isomers_filtered = list(EnumerateStereoisomers(mol5, options=opts))
    # isomers_all = list(EnumerateStereoisomers(mol))
    # assert len(isomers_all) == 16, "There are eight stereoisomers without tryEmbedding"
    # assert len(isomers_filtered) == 8, "There are four stereoisomers with tryEmbedding"


def test_remove_stereo():
    m = Mol("C/C=C/C=C\\C", "double_bond")
    assert m.remove_stereo().smiles == "CC=CC=CC"


def test_complete_stereoisomers():
    m = Mol("CC=CC", "double_bond")
    logger.info(m.ring_bond_stereo_info)
    assert m.is_stereo_specified is False
    libr = complete_stereoisomers(m)
    for _ in libr:
        logger.info(f"{_.name}: {_.smiles}")
    assert libr.count() == 2, "cis and trans are expected"
    assert all([_.is_stereo_specified for _ in libr])

    expected_canonical_smiles = [r"C/C=C/C", r"C/C=C\C"]
    canonical_smiles = [_.smiles for _ in libr]
    difference = set(expected_canonical_smiles) - set(canonical_smiles)
    assert len(difference) == 0

    # 0 out of 3 atom stereocenters is specified
    m = Mol("N=C1OC(CN2CC(C)OC(C)C2)CN1", "stereoisomer")
    assert m.is_stereo_specified is False
    libr = complete_stereoisomers(m)
    assert libr.count() == 6, "0 out of 3 atom stereocenters is specified"
    assert all([_.is_stereo_specified for _ in libr])
    expected_names = [f"stereoisomer.{i}" for i in [1, 2, 3, 4, 5, 6]]
    names = [_.name for _ in libr]
    assert names == expected_names
    expected_canonical_smiles = [
        "C[C@@H]1CN(C[C@@H]2CNC(=N)O2)C[C@H](C)O1",
        "C[C@@H]1CN(C[C@H]2CNC(=N)O2)C[C@H](C)O1",
        "C[C@@H]1CN(C[C@@H]2CNC(=N)O2)C[C@@H](C)O1",
        "C[C@@H]1CN(C[C@H]2CNC(=N)O2)C[C@@H](C)O1",
        "C[C@H]1CN(C[C@@H]2CNC(=N)O2)C[C@H](C)O1",
        "C[C@H]1CN(C[C@H]2CNC(=N)O2)C[C@H](C)O1",
    ]
    canonical_smiles = [_.smiles for _ in libr]
    difference = set(expected_canonical_smiles) - set(canonical_smiles)
    assert len(difference) == 0

    # 1 out of 3 atom stereocenters is specified
    m = Mol("N=C1OC(CN1)CN2CC(O[C@H](C2)C)C", "stereoisomer")
    assert m.is_stereo_specified is False
    libr = complete_stereoisomers(m)
    assert libr.count() == 4, "1 out of 3 atom stereocenters is specified"
    assert all([_.is_stereo_specified for _ in libr])
    expected_names = [f"stereoisomer.{i}" for i in [1, 2, 3, 4]]
    names = [_.name for _ in libr]
    assert names == expected_names
    expected_canonical_smiles = [
        "C[C@@H]1CN(C[C@@H]2CNC(=N)O2)C[C@H](C)O1",
        "C[C@@H]1CN(C[C@H]2CNC(=N)O2)C[C@H](C)O1",
        "C[C@H]1CN(C[C@@H]2CNC(=N)O2)C[C@H](C)O1",
        "C[C@H]1CN(C[C@H]2CNC(=N)O2)C[C@H](C)O1",
    ]
    canonical_smiles = [_.smiles for _ in libr]
    difference = set(expected_canonical_smiles) - set(canonical_smiles)
    assert len(difference) == 0

    # 2 out of 3 atom stereocenters are specified
    m = Mol("N=C1OC(CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    assert m.is_stereo_specified is False
    libr = complete_stereoisomers(m)
    assert libr.count() == 2, "2 out of 3 atom stereocenters is specified"
    assert all([_.is_stereo_specified for _ in libr])
    expected_names = [f"stereoisomer.{i}" for i in [1, 2]]
    names = [_.name for _ in libr]
    assert names == expected_names
    expected_canonical_smiles = [
        "C[C@@H]1CN(C[C@@H]2CNC(=N)O2)C[C@H](C)O1",
        "C[C@@H]1CN(C[C@H]2CNC(=N)O2)C[C@H](C)O1",
    ]
    canonical_smiles = [_.smiles for _ in libr]
    difference = set(expected_canonical_smiles) - set(canonical_smiles)
    assert len(difference) == 0

    # 3 out of 3 atom stereocenters are specified
    m = Mol("N=C1O[C@@H](CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    assert m.is_stereo_specified is True
    libr = complete_stereoisomers(m)
    assert libr.count() == 1, "3 out of 3 atom stereocenters is specified"
    assert all([_.is_stereo_specified for _ in libr])
    expected_names = [f"stereoisomer"]
    names = [_.name for _ in libr]
    assert names == expected_names
    expected_canonical_smiles = ["C[C@@H]1CN(C[C@@H]2CNC(=N)O2)C[C@H](C)O1"]
    canonical_smiles = [_.smiles for _ in libr]
    difference = set(expected_canonical_smiles) - set(canonical_smiles)
    assert len(difference) == 0

    # for 20 molecules
    isomer_libr = MolLibr()
    for mol in MolLibr(drug_smiles, drug_names):
        isomer_libr += complete_stereoisomers(mol)
    assert isomer_libr.count() >= 25


def test_num_stereoisomers():
    """count all possible stereoisomers ignoring current stereochemistry assignment"""
    m = Mol("Cc1nc2c(-c3ccc(Cl)cc3F)nc(N3CCN(C(=O)C(F)F)CC3)cn2c(=O)c1C")
    assert m.num_stereoisomers == 1
    m = Mol("CN1C=C([C@H]2CN(C3=NC(C4=CC=C(Cl)C=C4F)=C4N=C5CCCCN5C(=O)C4=C3)CCO2)C=N1")
    assert m.num_stereoisomers == 2
    m = Mol(
        "Cc1nc2c(-c3ccc(Cl)cc3F)nc(N3CCN(S(=O)(=O)N4C[C@@H](C)O[C@@H](C)C4)CC3)cn2c(=O)c1C"
    )
    assert m.num_stereoisomers == 3
    m = Mol("Cc1cc([C@H]2CN(c3cc4nc(C)c(C)c(=O)n4c(-c4ccc(Cl)cc4F)n3)C[C@@H](C)O2)ccn1")
    assert m.num_stereoisomers == 4


def test_complete_tautomers():
    m = Mol("Oc1c(cccc3)c3nc2ccncc12", "tautomer")
    libr = complete_tautomers(m)
    assert libr.count() == 3
    expected_names = ["tautomer.1", "tautomer.2", "tautomer.3"]
    names = [_.name for _ in libr]
    assert names == expected_names
    expected_canonical_smiles = [
        "O=c1c2c[nH]ccc-2nc2ccccc12",
        "O=c1c2ccccc2[nH]c2ccncc12",
        "Oc1c2ccccc2nc2ccncc12",
    ]
    canonical_smiles = [_.smiles for _ in libr]
    difference = set(expected_canonical_smiles) - set(canonical_smiles)
    assert len(difference) == 0

    m = Mol("CCCCNC(=O)[C@@H]1CCCN1[C@@H](CC)c1nnc(Cc2ccc(C)cc2)o1")
    libr = complete_tautomers(m)
    assert libr.count() == 5