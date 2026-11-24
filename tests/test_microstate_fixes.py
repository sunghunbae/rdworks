"""Regression tests for the microstate.py fixes."""

import math
import zlib
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from rdworks.microstate import (
    Boltzmann_weighted_average,
    Boltzmann_weights,
    InflectionDetector,
    QupkakeMicrostates,
    State,
    StateEnsemble,
    StateNetwork,
    beta_constant,
)


# 1-methyl-4-(4-hydroxyphenyl)piperazine: its non-canonical SMILES numbers atoms
# differently from the canonical SMILES of the generated microstates
PIPERAZINE = "CN1CCN(CC1)c1ccc(O)cc1"
PH = np.linspace(-2, 16, 181)


def _energy(state: State) -> float:
    """Deterministic energy per molecule, independent of the SMILES atom order."""
    return (zlib.crc32(Chem.CanonSmiles(state.smiles).encode()) % 1000) / 100.0


def _site_charge_curves(smiles: str) -> tuple[StateNetwork, dict[int, np.ndarray]]:
    network = StateNetwork().build(smiles=smiles, max_formal_charge=2, min_formal_charge=-2)
    network.set_energies([_energy(st) for st in network.visited_states])
    data, _ = network.get_plot_data_pH_vs_site_charge(PH, network.get_population(PH))
    curves = {}
    for site, charge in zip(data["site"], data["charge"]):
        curves.setdefault(site, []).append(charge)
    return network, {site: np.array(c) for site, c in curves.items()}


##################################################
### Boltzmann helpers
##################################################


def test_beta_constant_is_inverse_kT():
    kB = 1.987204259e-3  # kcal/(mol K)
    assert beta_constant(298.15) == pytest.approx(1.0 / (kB * 298.15))
    assert beta_constant() == pytest.approx(1.6878, abs=1e-4)  # default 298.15 K
    assert beta_constant(350.0) < beta_constant(298.15)


def test_boltzmann_accepts_integer_lists():
    weights = Boltzmann_weights([0, 1])
    assert np.allclose(weights, Boltzmann_weights(np.array([0.0, 1.0])))
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert Boltzmann_weighted_average([0, 1]) == pytest.approx(weights[1])


def test_boltzmann_invalid_input():
    with pytest.raises(TypeError):
        Boltzmann_weights("0 1")
    with pytest.raises(ValueError):
        Boltzmann_weights([])


def test_inflection_detector_input_checks():
    with pytest.raises(ValueError, match="at least 5 points"):
        InflectionDetector([0, 1, 2, 3], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        InflectionDetector(range(10), range(9))


##################################################
### State
##################################################


def test_update_does_not_duplicate_sites():
    state = State("c1ccc(CNc2ncnc3ccccc23)cc1")
    before = [(s.atom_idx, s.name, s.acid_base) for s in state.sites]
    state.update()
    assert [(s.atom_idx, s.name, s.acid_base) for s in state.sites] == before


def test_deserialize_does_not_duplicate_sites():
    state = State("CC(=O)O")
    rebuilt = State().deserialize(state.serialize())
    assert [(s.atom_idx, s.name) for s in rebuilt.sites] == [(s.atom_idx, s.name) for s in state.sites]


def test_state_identity_ignores_atom_order():
    assert State("CCO") == State("OCC")
    assert len({State("CCO"), State("OCC"), State("CCN")}) == 2  # hashable
    assert StateEnsemble([State("OCC"), State("CCO")]).drop().size() == 1
    assert State("OCC").smiles == "OCC"  # input SMILES (atom order) is kept


def test_charge_limits():
    assert State("C[NH3+]", max_formal_charge=0).get_protonated() == []
    assert State("CC(=O)[O-]", min_formal_charge=0).get_deprotonated() == []


def test_tautomers_inherit_settings():
    state = State(
        "Oc1ccccn1",
        tautomer_rule="rdkit",
        protomer_rule="simple",
        min_formal_charge=-1,
        max_formal_charge=1,
    )
    tautomers = state.get_tautomers()
    assert tautomers
    for t in tautomers:
        assert t.transformation == "Tautomer"
        assert (t.tautomer_rule, t.protomer_rule) == ("rdkit", "simple")
        assert (t.min_formal_charge, t.max_formal_charge) == (-1, 1)


def test_tautomers_exclude_noncanonical_input_itself():
    tautomers = State("n1ccccc1O", tautomer_rule="rdkit").get_tautomers()
    assert tautomers
    assert Chem.CanonSmiles("n1ccccc1O") not in [Chem.CanonSmiles(t.smiles) for t in tautomers]


##################################################
### StateNetwork / StateEnsemble
##################################################


def test_network_has_no_duplicate_molecules():
    noncanonical = StateNetwork().build(smiles=PIPERAZINE)
    canonical = StateNetwork().build(smiles=Chem.CanonSmiles(PIPERAZINE))
    counts = Counter(Chem.CanonSmiles(st.smiles) for st in noncanonical.visited_states)
    assert max(counts.values()) == 1  # the initial molecule was added twice
    assert noncanonical.size() == canonical.size()
    assert noncanonical.get_num_nodes() == noncanonical.size()


def test_site_charge_curves_do_not_depend_on_input_atom_order():
    from rdworks.microstate import atom_index_map

    network1, curves1 = _site_charge_curves(PIPERAZINE)
    network2, curves2 = _site_charge_curves(Chem.CanonSmiles(PIPERAZINE))
    assert len(curves1) == len(curves2) == 3  # two piperazine N and the phenol O
    # site indices refer to the atoms of each network's initial state
    imap = atom_index_map(network1.initial_state.rdmol, network2.initial_state.rdmol)
    assert {imap[site] for site in curves1} == set(curves2)
    for site, curve in curves1.items():
        assert np.allclose(curve, curves2[imap[site]], atol=1e-9)


def test_site_indices_refer_to_initial_state_atoms():
    network, curves = _site_charge_curves(PIPERAZINE)
    symbols = {network.initial_state.rdmol.GetAtomWithIdx(i).GetSymbol() for i in curves}
    assert symbols == {"N", "O"}
    # each site's charge goes from protonated (+1 or 0) to deprotonated (0 or -1)
    for curve in curves.values():
        assert curve[0] > curve[-1]


def test_atom_index_map_preserves_elements():
    from rdworks.microstate import atom_index_map

    network = StateNetwork().build(smiles=PIPERAZINE)
    ref = network.initial_state.rdmol
    for state in network.visited_states:
        imap = atom_index_map(ref, state.rdmol)
        assert len(imap) == ref.GetNumAtoms()
        for i, j in imap.items():
            assert ref.GetAtomWithIdx(i).GetSymbol() == state.rdmol.GetAtomWithIdx(j).GetSymbol()


def test_plot_data_accepts_list_ph():
    ensemble = StateEnsemble([State("CC(=O)O"), State("CC(=O)[O-]")]).set_energies([0.0, 1.0])
    p = ensemble.get_population(PH)
    data, _ = ensemble.get_plot_data_pH_vs_charge(list(PH), p)
    assert data["pH"] == pytest.approx(list(PH))
    assert len(data["charge"]) == len(PH)


##################################################
### QupkakeMicrostates
##################################################


def test_qupkake_get_populations():
    q = QupkakeMicrostates.__new__(QupkakeMicrostates)  # skip xTB-dependent setup
    q.states = [
        SimpleNamespace(PE=0.0, charge=0),
        SimpleNamespace(PE=1.0, charge=-1),
    ]
    populations = q.get_populations(7.0)  # previously UnboundLocalError
    assert sum(p for _, p in populations) == pytest.approx(1.0)
    probabilities = [p for _, p in populations]
    assert probabilities == sorted(probabilities, reverse=True)
    # at pH 7 the anion is favored: dG = 1.0 - ln(10) kT pH < 0
    assert populations[0][0] == 1
