from rdworks import Mol, Conf
from rdworks.xtb.wrapper import GFN2xTB
from rdworks.data.test.example import drugs

import pytest

testmol = Mol()
testconf = Conf()


@pytest.fixture(scope="module", autouse=True)
def required_test_data():
    global testmol
    global testconf
    name = "Acetaminophen"
    testmol = Mol(drugs[name], name).make_confs(n=4).optimize_confs()
    testmol = testmol.drop_confs(similar=True, verbose=True).sort_confs()
    testconf = testmol.confs[0]


def test_xtb_wrapper(required_test_data):
    conf = testconf.copy()
    gfn2xtb = GFN2xTB(conf.rdmol)

    assert gfn2xtb.is_xtb_ready() == True
    assert gfn2xtb.is_cpx_ready() == True
    assert gfn2xtb.is_cpcmx_ready() == True
    assert gfn2xtb.is_ready() == True
    assert gfn2xtb.version() is not None

    print("GFN2xTB.singlepoint()")
    outdict = gfn2xtb.singlepoint()
    print(outdict)
    print()


def test_singlepoint(required_test_data):
    conf = testconf.copy()

    print("number of atoms=", conf.natoms)

    gfn2xtb = GFN2xTB(conf.rdmol)

    print("GFN2xTB.singlepoint()")
    outdict = gfn2xtb.singlepoint()
    print(outdict)
    print()

    print("GFN2xTB.singlepoint(water='gbsa')")
    outdict = gfn2xtb.singlepoint(water="gbsa")
    print(outdict)
    print()

    print("GFN2xTB.singlepoint(water='alpb')")
    outdict = gfn2xtb.singlepoint(water="alpb")
    print(outdict)
    print()

    print("GFN2xTB.singlepoint(water='cpcmx')")
    outdict = gfn2xtb.singlepoint(water="cpcmx")
    print(outdict)
    print()


def test_optimize(required_test_data):
    conf = testconf.copy()
    print("GFN2xTB.optimize()")
    outdict = GFN2xTB(conf.rdmol).optimize(verbose=True)
    print(outdict)
    print()


def test_singlepoint_confs_concurrency(required_test_data):
    mol = testmol.copy()
    print("Mol.singlepoint_confs with xTB/cpcmx")
    mol = mol.compute(max_workers=4, progress=True).singlepoint_confs(
        calculator="xTB", water="cpcmx"
    )


def test_optimize_confs_concurrency(required_test_data):
    mol = testmol.copy()
    print("Mol.optimize_confs with xTB/alpb")
    mol = mol.compute(max_workers=4, progress=True).optimize_confs(
        calculator="xTB", water="alpb"
    )


def test_esp(required_test_data):
    conf = testconf.copy()
    print("GFN2xTB.esp()")
    outdict = GFN2xTB(conf.rdmol).esp_volumetric(max_iterations=500, water="cpcmx")
    print(outdict)
    print()
