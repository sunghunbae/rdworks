"""Guards against circular imports and import-order dependencies inside rdworks.

Every submodule must be importable on its own, even before `rdworks/__init__.py`
has run, so it must import classes from their defining modules
(`from rdworks.mol import Mol`), not from the package (`from rdworks import Mol`).
Imports inside functions or under `if TYPE_CHECKING:` are not affected.
"""

import ast
import importlib.util
import pathlib
import subprocess
import sys

import pytest


PACKAGE_DIR = pathlib.Path(importlib.util.find_spec("rdworks").origin).parent

# imports `name` with a stub `rdworks` package whose __init__.py has not run
STUB_IMPORT = r"""
import importlib, importlib.util, sys, types
spec = importlib.util.find_spec("rdworks")
stub = types.ModuleType("rdworks")
stub.__path__ = list(spec.submodule_search_locations)
stub.__spec__ = spec
sys.modules["rdworks"] = stub
importlib.import_module(sys.argv[1])
"""


def _source_files() -> list[pathlib.Path]:
    # data/ holds data files and stand-alone build scripts, not library code
    return [
        p
        for p in sorted(PACKAGE_DIR.rglob("*.py"))
        if p.relative_to(PACKAGE_DIR).parts[0] != "data"
    ]


def _module_name(path: pathlib.Path) -> str:
    parts = ["rdworks", *path.relative_to(PACKAGE_DIR).with_suffix("").parts]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


SUBMODULES = [
    _module_name(p) for p in _source_files() if p != PACKAGE_DIR / "__init__.py"
]


def _is_type_checking(node: ast.If) -> bool:
    return ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")


def _module_level_imports(tree: ast.Module):
    """Yields imports executed at import time (skips functions/classes and TYPE_CHECKING)."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If):
            if not _is_type_checking(node):
                stack.extend(node.body)
            stack.extend(node.orelse)
        elif isinstance(node, ast.Try):
            stack.extend(node.body)
            stack.extend(node.orelse)
            stack.extend(node.finalbody)
            for handler in node.handlers:
                stack.extend(handler.body)
        elif isinstance(node, ast.With):
            stack.extend(node.body)


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports_without_package_init(name):
    result = subprocess.run(
        [sys.executable, "-c", STUB_IMPORT, name],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "path",
    [p for p in _source_files() if p != PACKAGE_DIR / "__init__.py"],
    ids=lambda p: str(p.relative_to(PACKAGE_DIR)),
)
def test_no_module_level_import_from_package(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offending = [
        f"line {node.lineno}: from rdworks import {', '.join(a.name for a in node.names)}"
        for node in _module_level_imports(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "rdworks"
    ]
    assert not offending, (
        "import from the defining module instead (e.g. `from rdworks.mol import Mol`): "
        + "; ".join(offending)
    )
