"""Regression tests for logging and exception-handling best practices.

* no bare ``except:`` and no swallowed ``except Exception`` in the package
* no ``print`` in the package; report methods return ``str``
* the package logger only has a NullHandler and RDKit logging is untouched
* narrowed handlers still convert RDKit failures into the documented errors
"""

import ast
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "rdworks"

# (module, function) pairs allowed to call print(); report methods return str instead
PRINT_ALLOWED: set[tuple[str, str]] = set()


def _modules():
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith(("data/", "xtb/example/")):
            continue
        yield rel, ast.parse(p.read_text(encoding="utf-8"))


def test_no_catch_all_exception_handlers():
    offenders = []
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is None:
                offenders.append(f"{rel}:{node.lineno} bare except")
            elif getattr(node.type, "id", "") in ("Exception", "BaseException"):
                # allowed only when the handler re-raises (cleanup / wrap-and-chain)
                if not (node.body and isinstance(node.body[-1], ast.Raise)):
                    offenders.append(f"{rel}:{node.lineno} swallowed Exception")
    assert offenders == []


def test_no_print_in_package():
    offenders = []
    for rel, tree in _modules():
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
                enclosing = [
                    f.name for f in funcs if f.lineno <= node.lineno <= f.end_lineno
                ]
                key = (rel, enclosing[-1] if enclosing else None)
                if key not in PRINT_ALLOWED:
                    offenders.append(f"{rel}:{node.lineno} in {key[1]}")
    assert offenders == []


def test_package_logger_only_has_null_handler():
    import rdworks  # noqa: F401

    pkg_logger = logging.getLogger("rdworks")
    assert [type(h) for h in pkg_logger.handlers] == [logging.NullHandler]
    assert pkg_logger.level == logging.NOTSET  # level is left to the application


def test_import_does_not_change_rdkit_logging():
    # fresh interpreter: compare RDKit log status before/after importing rdworks
    code = textwrap.dedent(
        """
        from rdkit import rdBase
        before = rdBase.LogStatus()
        import rdworks
        after = rdBase.LogStatus()
        print(before == after)
        """
    )
    env = {"PYTHONPATH": str(SRC.parent), "PYTHONDONTWRITEBYTECODE": "1"}
    import os

    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "True"


def test_invalid_smiles_raises_value_error_with_cause():
    from rdworks import Mol

    with pytest.raises(ValueError, match="invalid SMILES") as exc:
        Mol("C1CC")
    assert exc.value.__cause__ is not None


def test_invalid_molblock_raises_value_error():
    from rdworks import Conf

    with pytest.raises(ValueError, match="invalid MolBlock"):
        Conf("not a molblock")


def test_qed_unknown_property_is_key_error():
    from rdworks import Mol

    with pytest.raises(KeyError, match="unknown property"):
        Mol("CCO").qed(properties=["NotADescriptor"])


def test_is_matching_bad_terms_logs_warning(caplog):
    from rdworks import Mol

    with caplog.at_level(logging.WARNING, logger="rdworks.mol"):
        assert Mol("CCO").is_matching(123) is False
    assert "predefined terms" in caplog.text


def test_is_matching_long_name_is_not_treated_as_file(monkeypatch):
    import rdworks.mol as rdmol_module
    from rdworks import Mol

    # very long non-path string: Path.is_file() may raise ENAMETOOLONG
    seen = []

    cns_xml = SRC / "data" / "test" / "cns.xml"

    def fake_predefined(terms):
        seen.append(terms)
        return cns_xml

    monkeypatch.setattr(rdmol_module, "get_predefined_xml", fake_predefined)
    Mol("CCO").is_matching("C" * 5000)
    assert seen == ["C" * 5000]


def test_read_dataframe_missing_columns():
    from rdworks.readers import read_dataframe

    df = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
    with pytest.raises(ValueError, match="SMILES column"):
        read_dataframe(df, smiles="smi", name="name")
    with pytest.raises(ValueError, match="NAME column"):
        read_dataframe(df, smiles="smiles", name="title")


def test_drop_without_terms_logs_warning(caplog):
    from rdworks import MolLibr

    libr = MolLibr(["CCO", "c1ccccc1"], ["a", "b"])
    with caplog.at_level(logging.WARNING, logger="rdworks.mollibr"):
        assert libr.drop() is libr
    assert "no terms given" in caplog.text


def test_connect_core_and_chains_narrowed_handler(monkeypatch):
    import rdworks.rgroup as rgroup

    assert rgroup.connect_core_and_chains("c1ccccc1[*:1].[*:1]C") is not None

    def sanitize_failure(*args, **kwargs):
        raise ValueError("simulated AtomValenceException")  # RDKit sanitize errors

    monkeypatch.setattr(rgroup.Chem, "RemoveHs", sanitize_failure)
    assert rgroup.connect_core_and_chains("c1ccccc1[*:1].[*:1]C") is None

    def programming_error(*args, **kwargs):
        raise ZeroDivisionError("bug")  # unrelated errors must no longer be swallowed

    monkeypatch.setattr(rgroup.Chem, "RemoveHs", programming_error)
    with pytest.raises(ZeroDivisionError):
        rgroup.connect_core_and_chains("c1ccccc1[*:1].[*:1]C")


def test_scaffold_breakup_verbose_logs_and_acyclic(caplog):
    from rdkit import Chem
    from rdworks.scaffold import breakup

    assert breakup(Chem.MolFromSmiles("CCCC")) == []
    with caplog.at_level(logging.INFO, logger="rdworks.scaffold"):
        children = breakup(Chem.MolFromSmiles("c1ccc2ccccc2c1CC1CCNCC1"), verbose=True)
    assert children
    assert "breakup parent" in caplog.text


def test_mol_reports_return_str(capsys):
    from rdworks import Mol

    m = Mol("C/C=C/C[C@H](N)O", "m1").qed()
    stereo = m.report_stereo()
    props = m.report_props()
    assert capsys.readouterr().out == ""  # nothing printed
    assert stereo.splitlines()[0] == "chiral centers = unspecified 0 / total 1"
    assert "Bond_Double bond: 1 ring: False N: False Specified Bond_Trans" in stereo
    assert props.splitlines()[0] == f"Properties({len(m.props)}):"
    # keys padded to a common width (longest key + 5)
    width = max(len(k) for k in m.props) + 5
    assert props.splitlines()[1] == f"  {'aka':<{width}} []"
    assert not props.endswith("\n")


def test_microstate_reports_return_str(capsys):
    from rdworks.microstate import StateNetwork

    sn = StateNetwork()
    sn.build(smiles="c1ccc(CNc2ncnc3ccccc23)cc1", max_formal_charge=3, tautomer_rule=None)
    network = sn.info()
    state = sn.get_initial_state().info(index=0)
    ensemble = sn.get_state_ensemble().info()
    assert capsys.readouterr().out == ""
    assert network == "StateNetwork - nodes: 12 edges: 40"
    assert state.splitlines()[0] == "[ 0] SMILES: c1ccc(CNc2ncnc3ccccc23)cc1"
    # ensemble = state reports joined by blank lines, starting with state 0
    assert ensemble.startswith(state + "\n\n")
    assert ensemble.count("SMILES:") == 12


def test_mollibr_report_duplicates(capsys, caplog):
    from rdworks import MolLibr

    libr = MolLibr(["CCO", "OCC", "c1ccccc1", "C(O)C"], ["a", "b", "c", "d"])
    expected = "duplicates:\n  a(2) - b,d\nde-duplicated to 2 molecules"
    assert libr.unique().report_duplicates() == expected
    with caplog.at_level(logging.INFO, logger="rdworks.mollibr"):
        uniq = libr.unique(report=True)
    assert uniq.count() == 2
    assert capsys.readouterr().out == ""
    assert expected in caplog.text


def test_show_styles_returns_str(capsys):
    from rdkit import Chem
    from rdworks.view import DescriptiveDraw

    dd = DescriptiveDraw(Chem.MolFromSmiles("CCN"))
    table = dd.show_styles()
    assert capsys.readouterr().out == ""
    lines = table.splitlines()
    assert lines[0].split() == ["Name", "r", "rgba", "linewidth", "fill"]
    assert len(lines) == len(dd.style) + 1


def test_xml_combine_defaults_to_or(tmp_path):
    from rdworks.xml import parse_xml

    xml = tmp_path / "t.xml"
    xml.write_text(
        '<rule><descriptor name="MolWt"><max>500</max></descriptor></rule>',
        encoding="utf-8",
    )
    _, combine = parse_xml(xml)
    assert combine == "OR"
