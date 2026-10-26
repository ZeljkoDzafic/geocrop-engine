"""Architecture fitness tests: layer boundaries are enforced on every module, not just documented.

The layer map lives in `geocrop._layers`; see docs/adr/0001-layered-architecture.md.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from geocrop import _layers

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "geocrop"
MODULES = sorted(p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__")
FIX_HINT = ("Fix: move the needed code into a module of an allowed layer (re-export it from the old place for "
            "backward compatibility), invert the dependency with a registry in geocrop.plugins, or - only if the "
            "module really belongs elsewhere - reclassify it in src/geocrop/_layers.py.")


@dataclass(frozen=True)
class Import:
    target: str        # top-level module name (geocrop submodule or third-party/stdlib package)
    internal: bool
    line: int
    lazy: bool         # inside a function body


def _imports(module: str) -> list[Import]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf8"))
    found: list[Import] = []

    def visit(node: ast.AST, lazy: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner = lazy or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                for alias in child.names:
                    parts = alias.name.split(".")
                    if parts[0] == "geocrop":
                        if len(parts) > 1:
                            found.append(Import(parts[1], True, child.lineno, lazy))
                    else:
                        found.append(Import(parts[0], False, child.lineno, lazy))
            elif isinstance(child, ast.ImportFrom):
                if child.level == 0 and not (child.module or "").startswith("geocrop"):
                    found.append(Import((child.module or "").split(".")[0], False, child.lineno, lazy))
                    continue
                parts = (child.module or "").split(".")
                if child.level == 0:
                    parts = parts[1:]          # absolute geocrop.<module>
                if parts and parts[0]:
                    found.append(Import(parts[0], True, child.lineno, lazy))
                else:                          # from . import a, b  /  from geocrop import a
                    for alias in child.names:
                        if (PACKAGE / f"{alias.name}.py").is_file():
                            found.append(Import(alias.name, True, child.lineno, lazy))
            visit(child, inner)

    visit(tree, False)
    return found


def test_every_module_has_a_layer():
    unclassified = [m for m in MODULES if _layers.layer_of(m) is None]
    assert not unclassified, (
        f"Modules without a layer: {unclassified}. Add each to exactly one layer in src/geocrop/_layers.py "
        "(core = pure geospatial/ML on local files; adapters = network, external tools, exchange formats; "
        "application = workflows; interface = cli/web).")
    stale = sorted(set().union(*_layers.LAYERS.values()) - set(MODULES))
    assert not stale, f"src/geocrop/_layers.py lists modules that do not exist: {stale}"
    duplicated = [m for m in MODULES if sum(m in mods for mods in _layers.LAYERS.values()) > 1]
    assert not duplicated, f"Modules in more than one layer: {duplicated}"


@pytest.mark.parametrize("module", MODULES)
def test_layer_dependencies(module):
    layer = _layers.layer_of(module)
    allowed = _layers.ALLOWED[layer]
    problems = []
    for imp in _imports(module):
        if not imp.internal:
            continue
        target_layer = _layers.layer_of(imp.target)
        if target_layer is None or target_layer in allowed:
            continue
        if (module, imp.target) in _layers.EXCEPTIONS and imp.lazy:
            continue
        problems.append(f"geocrop.{module} ({layer}) imports geocrop.{imp.target} ({target_layer}) at line "
                        f"{imp.line}; {layer} may import only {sorted(allowed)}.")
    assert not problems, "\n".join(problems + [FIX_HINT])


@pytest.mark.parametrize("module", MODULES)
def test_web_modules_are_imported_only_by_the_cli(module):
    if module in _layers.WEB_IMPORTERS:
        return
    bad = [f"geocrop.{module} imports geocrop.{i.target} at line {i.line}" for i in _imports(module)
           if i.internal and i.target in _layers.WEB_MODULES]
    assert not bad, "\n".join(bad + ["Web modules are an entry point: only geocrop.cli (`geocrop serve`) may "
                                     "import them. Put shared logic in a core or application module instead."])


@pytest.mark.parametrize("module", sorted(_layers.LAYERS[_layers.CORE]))
def test_core_uses_only_scientific_libraries(module):
    problems = []
    for imp in _imports(module):
        if imp.internal or not imp.target:
            continue
        name = imp.target
        if name in _layers.NETWORK_LIBRARIES or name in _layers.CORE_FORBIDDEN_STDLIB:
            problems.append(f"geocrop.{module} (core) imports {name!r} at line {imp.line}: network access and "
                            "process execution belong in an adapter (see geocrop.snap, geocrop.catalog).")
        elif name not in sys.stdlib_module_names and name not in _layers.CORE_THIRD_PARTY:
            problems.append(f"geocrop.{module} (core) imports third-party {name!r} at line {imp.line}, which is "
                            "not in _layers.CORE_THIRD_PARTY. Add it there if it is a scientific library, or "
                            "move the code to an adapter.")
    assert not problems, "\n".join(problems)


def test_exceptions_are_lazy_and_still_needed():
    for (importer, imported), reason in _layers.EXCEPTIONS.items():
        uses = [i for i in _imports(importer) if i.internal and i.target == imported]
        assert uses, f"Stale exception ({importer} -> {imported}: {reason}); remove it from _layers.EXCEPTIONS"
        eager = [i.line for i in uses if not i.lazy]
        assert not eager, (f"geocrop.{importer} imports geocrop.{imported} at module level (lines {eager}); the "
                           "documented exception only covers imports inside functions.")


def test_core_import_pulls_in_no_network_library():
    """Transitive check at runtime: importing all of core loads no network or web framework."""
    # unet and the encoder ports import torch at module level (optional `deep` extra)
    needs_torch = {"unet", "presto_encoder", "galileo_encoder", "anysat_encoder"}
    skip = {"unet"} if importlib.util.find_spec("torch") else needs_torch
    core = sorted(_layers.LAYERS[_layers.CORE] - skip)
    code = ("import sys\n" + "".join(f"import geocrop.{m}\n" for m in core) +
            f"bad = sorted(m for m in {sorted(_layers.NETWORK_LIBRARIES)!r} if m in sys.modules)\n"
            "print(','.join(bad))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(PACKAGE.parent)})
    assert out.stdout.strip() == "", f"Importing core loaded network libraries: {out.stdout.strip()}"


def test_import_parser_sees_known_edges():
    """Guard the guard: the parser must detect relative, absolute and lazy imports."""
    model = {i.target for i in _imports("model") if i.internal}
    assert {"estimators", "raster", "samples", "applicability"} <= model
    assert any(i.target == "snap" and i.lazy for i in _imports("s1"))
    assert any(i.target == "web" for i in _imports("cli"))
