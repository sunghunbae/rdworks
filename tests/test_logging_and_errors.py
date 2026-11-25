"""Regression tests for logging and exception-handling best practices.

* no bare ``except:`` and no swallowed ``except Exception`` in the package
* ``print`` only in explicit user-facing report methods
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

# user-facing report methods that still print by design (pending decision)
PRINT_ALLOWED = {
    ("microstate.py", "info"),
    ("mol.py", "report_stereo"),
    ("mol.py", "report_props"),
    ("mollibr.py", "unique"),
    ("view.py", "show_styles"),
}


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


def test_print_only_in_report_methods():
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


def test_xml_combine_defaults_to_or(tmp_path):
    from rdworks.xml import parse_xml

    xml = tmp_path / "t.xml"
    xml.write_text(
        '<rule><descriptor name="MolWt"><max>500</max></descriptor></rule>',
        encoding="utf-8",
    )
    _, combine = parse_xml(xml)
    assert combine == "OR"
