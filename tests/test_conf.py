import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rdworks import Mol
from rdworks.conf import (
    Conf,
    DummyBatchOptimizer,
    DummyBatchSinglePointer,
    batch_optimize,
    batch_singlepoint,
    prepare_batches,
)
from rdworks.utils import recursive_round


def test_align_and_cluster_confs():
    mol = Mol("N=C1O[C@@H](CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    mol = mol.make_confs()
    mol = mol.drop_confs(similar=True, similar_rmsd=0.5, window=15.0)
    mol = mol.sort_confs().align_confs().cluster_confs().rename()
    mol.to_sdf(confs=True)  # string output


def test_make_confs():
    mol = Mol("N=C1O[C@@H](CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    mol = mol.make_confs(method="ETKDG")
    assert mol.num_confs > 1


def test_singlepoint():
    mol = Mol("N=C1O[C@@H](CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    mol = mol.make_confs(n=10, method="ETKDG")
    mol = mol.singlepoint_confs(calculator="MMFF94")
    assert all([_.props.get("E_tot(kcal/mol)") is not None for _ in mol.confs])


def test_singlepoint():
    mol = Mol("N=C1O[C@@H](CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    mol = mol.make_confs(n=10, method="ETKDG")
    mol = mol.compute(max_workers=4).singlepoint_confs(calculator="MMFF94")
    assert all([_.props.get("E_tot(kcal/mol)") is not None for _ in mol.confs])


def test_optimize_confs():
    mol = Mol("N=C1O[C@@H](CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    mol = mol.make_confs().optimize_confs(calculator="MMFF94")


def test_optimize_confs_concurrent():
    mol = Mol("N=C1O[C@@H](CN1)CN2C[C@H](O[C@H](C2)C)C", "stereoisomer")
    mol = mol.make_confs().compute(max_workers=4).optimize_confs(calculator="MMFF94")


def test_workflow():
    state_mol = Mol("Cc1nc2cc(Cl)nc(Cl)c2nc1C", "A-1250")
    state_mol = state_mol.make_confs(n=10, method="ETKDG")
    state_mol = state_mol.drop_confs(similar=True, similar_rmsd=0.3)
    state_mol = state_mol.sort_confs().rename()
    state_mol = state_mol.align_confs(method="rigid_fragment")
    state_mol = state_mol.cluster_confs(
        "QT", threshold=1.0, sort="energy", symmetry_aware=True
    )
    print(state_mol.name, {k: v for k, v in state_mol.props.items()})
    for conf in state_mol.confs:
        conf.props = recursive_round(conf.props, decimals=2)
        print(conf.name, {k: v for k, v in conf.props.items()})
        print(conf.SASA)


def test_protonate_deprotonate():
    mol = Mol("c1ccc(CNc2ncnc3ccccc23)cc1")  # SAMPL6 SM07
    mol = mol.make_confs(n=1)
    # SMILES: c1ccc(CNc2ncnc3ccccc23)cc1
    # Formal charge: 0
    # Origin: None
    # Transformation: None
    # Ionizable sites:
    # - atom_idx=  5, atom=  N, q= +0, hs= 1, pr= 1, de= 1, acid_base= B:A:A, name= Amine:Amide:Amide vinylogue
    # - atom_idx=  7, atom=  N, q= +0, hs= 0, pr= 1, de= 0, acid_base= B, name= Aza-aromatics
    # - atom_idx=  9, atom=  N, q= +0, hs= 0, pr= 1, de= 0, acid_base= B, name= Aza-aromatics

    conf = mol.confs[0].copy()
    assert conf.positions.shape == (31, 3)
    assert conf.charge == 0

    conf = conf.protonate([7])
    assert conf.positions.shape == (32, 3)
    assert conf.charge == 1

    conf = mol.confs[0].copy()
    conf = conf.protonate([7, 9])
    assert conf.positions.shape == (33, 3)
    assert conf.charge == 2

    conf = mol.confs[0].copy()
    conf = conf.deprotonate([5])
    assert conf.positions.shape == (30, 3)
    assert conf.charge == -1


def test_from_molblock():
    mol = Mol("c1ccc(CNc2ncnc3ccccc23)cc1")  # SAMPL6 SM07
    mol = mol.make_confs(n=1)
    conf = mol.confs[0].copy()
    mol2 = Mol().from_molblock(conf.molblock)


def test_serialization():
    smiles = "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
    name = "atorvastatin"
    mol = Mol(smiles, name)
    assert mol.num_confs == 0
    assert mol.name == name
    mol = mol.make_confs(n=10)
    assert mol.num_confs == 10
    serialized = mol.serialize()
    rebuilt = Mol().deserialize(serialized)
    assert rebuilt.num_confs == 10
    assert rebuilt.name == name
    assert rebuilt == mol
    print(mol.confs[0].positions.shape)


##################################################
### Regression tests for conf.py fixes
##################################################


def _embedded_rdmol(smiles: str = "CCO", num_confs: int = 1) -> Chem.Mol:
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(rdmol, num_confs, randomSeed=1)
    return rdmol


@pytest.fixture
def apap_conf() -> Conf:
    """First conformer of acetaminophen (reproducible)."""
    return Mol("CC(=O)Nc1ccc(O)cc1", "apap").make_confs(n=3, seed=1).confs[0]


def test_conf_cog_and_rg(apap_conf):
    xyz = apap_conf.positions
    assert np.allclose(apap_conf.COG, xyz.mean(axis=0))
    expected_rg = np.sqrt(np.mean(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1)))
    assert np.isclose(apap_conf.Rg, expected_rg)


def test_conf_invalid_molblock():
    with pytest.raises(ValueError, match="invalid MolBlock"):
        Conf("not a molblock")


@pytest.mark.parametrize("num_confs", [0, 2])
def test_conf_requires_exactly_one_conformer(num_confs):
    rdmol = _embedded_rdmol(num_confs=num_confs)
    with pytest.raises(ValueError, match="exactly one conformer"):
        Conf(rdmol)


def test_conf_does_not_alias_input_molecule():
    rdmol = _embedded_rdmol()
    original = rdmol.GetConformer().GetPositions().copy()
    conf = Conf(rdmol, name="ethanol")
    conf.sync(conf.positions + 1.0)
    assert np.allclose(rdmol.GetConformer().GetPositions(), original)
    assert conf.rdmol.GetProp("_Name") == "ethanol"


def test_conf_positions_follow_optimize(apap_conf):
    before = apap_conf.positions.copy()
    apap_conf.optimize("UFF")
    current = apap_conf.rdmol.GetConformer().GetPositions()
    assert not np.allclose(current, before)  # precondition: geometry changed
    assert np.allclose(apap_conf.positions, current)
    assert f"{current[0][0]:.14f}" in apap_conf.to_xyz()
    assert np.allclose(Chem.MolFromMolBlock(apap_conf.molblock, removeHs=False).GetConformer().GetPositions(), current, atol=1e-3)


def test_conf_serialize_keeps_current_coordinates(apap_conf):
    apap_conf.optimize("UFF")
    optimized = apap_conf.rdmol.GetConformer().GetPositions()  # ground truth
    rebuilt = Conf().deserialize(apap_conf.serialize())
    assert np.allclose(rebuilt.positions, optimized, atol=1e-3)


def test_mol_serialize_keeps_optimized_coordinates():
    mol = Mol("CCO", "ethanol").make_confs(n=1, seed=1).optimize_confs("UFF")
    optimized = mol.confs[0].rdmol.GetConformer().GetPositions()  # ground truth
    rebuilt = Mol().deserialize(mol.serialize())
    assert np.allclose(rebuilt.confs[0].positions, optimized, atol=1e-3)


def test_conf_positions_follow_sync_and_torsion(apap_conf):
    shifted = apap_conf.positions + 1.0
    apap_conf.sync(shifted)
    assert np.allclose(apap_conf.positions, shifted)
    torsion = apap_conf.get_torsion_angle_atoms()[0]
    apap_conf.set_torsion_angle(*torsion, 90.0)
    xyz = apap_conf.positions
    i, j, k, l = torsion
    # dihedral recomputed from the `positions` property itself
    b0, b1, b2 = xyz[i] - xyz[j], xyz[k] - xyz[j], xyz[l] - xyz[k]
    b1 /= np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    angle = np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
    assert np.isclose(abs(angle), 90.0, atol=1e-3)


@pytest.mark.parametrize("smiles", ["CC(=O)Nc1ccc(O)cc1", "CC#N", "CC#CC"])
def test_has_acceptable_bond_lengths(smiles):
    conf = Mol(smiles).make_confs(n=1, seed=1).confs[0]
    assert conf.has_acceptable_bond_lengths()
    xyz = conf.positions
    xyz[0] += np.array([1.5, 0.0, 0.0])  # stretch the bonds of atom 0
    assert not conf.sync(xyz).has_acceptable_bond_lengths()


class _FakeConf:
    def __init__(self, natoms):
        self.natoms = natoms
        self.rdmol = natoms


@pytest.mark.parametrize(
    "natoms, batchsize_atoms, expected_sizes",
    [
        ([20, 5], 10, [1, 1]),  # first conf larger than a batch: no empty batch
        ([4, 4, 4], 10, [2, 1]),
        ([4, 4], 8, [2]),
    ],
)
def test_prepare_batches(natoms, batchsize_atoms, expected_sizes):
    batches = prepare_batches([_FakeConf(n) for n in natoms], batchsize_atoms)
    assert [b.size for b in batches] == expected_sizes
    assert all(b.size > 0 for b in batches)


@pytest.mark.parametrize(
    "batch_function, calculator, energy_keys",
    [
        (batch_singlepoint, DummyBatchSinglePointer, {"E_tot(kcal/mol)"}),
        (
            batch_optimize,
            DummyBatchOptimizer,
            {"E_tot(kcal/mol)", "E_tot_init(kcal/mol)", "Converged"},
        ),
    ],
)
def test_batch_results_keep_names_and_props(batch_function, calculator, energy_keys):
    mol = Mol("CC(=O)Nc1ccc(O)cc1", "apap").make_confs(n=3, seed=1)
    mol.confs[0].props["custom"] = "keep"
    mol.confs[0].props["cluster"] = 1  # stale after recalculation
    results = batch_function(mol.confs, calculator, batchsize_atoms=10000)
    assert [c.name for c in results] == ["apap/1", "apap/2", "apap/3"]
    assert results[0].props["custom"] == "keep"
    assert "cluster" not in results[0].props
    assert "E_rel(kcal/mol)" not in results[0].props  # stale relative energy dropped
    assert energy_keys <= set(results[0].props)
    assert {"atoms", "charge"} <= set(results[0].props)


def test_batch_result_count_mismatch():
    class DropOne(DummyBatchSinglePointer):
        def run(self):
            calculated = super().run()
            return calculated._replace(mols=calculated.mols[:-1])

    mol = Mol("CCO").make_confs(n=2, seed=1)
    with pytest.raises(RuntimeError, match="returned 1 molecules for 2"):
        batch_singlepoint(mol.confs, DropOne, batchsize_atoms=10000)


def test_conf_str_shows_smiles(apap_conf):
    text = str(apap_conf)
    assert apap_conf.smiles in text
    assert "rdkit.Chem.rdchem.Mol object" not in text
