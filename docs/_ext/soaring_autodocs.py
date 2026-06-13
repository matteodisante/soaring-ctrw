"""Build-time generator for the parts of the docs that must never go stale.

This is a tiny Sphinx extension. On every build (``builder-inited`` event)
it re-derives, directly from the source tree:

* ``_generated/architecture.dot`` — the internal import graph of the whole
  project (library modules + scripts), rendered on the *Architecture* page.
* ``_generated/script_<name>.dot`` — a focused import map for each script.
* ``_generated/scripts/<name>.rst`` — one reference page per script, pulling
  in the script's module docstring (``automodule``), its import map, and its
  live ``--help`` output (``program-output``).

Because everything is regenerated from the current source on each build,
adding/renaming/removing a module or script is reflected automatically: there
is no hand-maintained list to fall out of sync. New scripts appear via the
``:glob:`` toctree in ``scripts.rst``.

The only thing this file hard-codes is the *layout* convention
(``src/soaring_ctrw`` for the library, ``scripts/*.py`` for the entry points);
the contents are always read from disk.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = "soaring_ctrw"


# ---------------------------------------------------------------------------
# Source scanning
# ---------------------------------------------------------------------------


def _internal_imports(path: Path, self_module: str) -> set[str]:
    """Return the set of ``soaring_ctrw`` submodule names imported by ``path``.

    Recognises ``import soaring_ctrw.X``, ``from soaring_ctrw.X import ...``
    and (for files inside the package) relative ``from .X import ...``.
    Only the leaf submodule name (``X``) is returned.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE + "."):
                    found.add(alias.name.split(".")[1])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level and mod:  # from .X import ...
                found.add(mod.split(".")[0])
            elif mod.startswith(PACKAGE + "."):
                found.add(mod.split(".")[1])
    found.discard(self_module)
    return found


def _scan(repo_root: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Scan the repo and return ``(library, scripts)`` import maps.

    ``library`` maps each ``soaring_ctrw`` submodule to the set of sibling
    submodules it imports; ``scripts`` maps each script stem to the set of
    library submodules it imports.
    """
    pkg_dir = repo_root / "src" / PACKAGE
    scripts_dir = repo_root / "scripts"

    library: dict[str, set[str]] = {}
    for py in sorted(pkg_dir.glob("*.py")):
        if py.stem == "__init__":
            continue
        library[py.stem] = _internal_imports(py, py.stem)

    scripts: dict[str, set[str]] = {}
    for py in sorted(scripts_dir.glob("*.py")):
        scripts[py.stem] = _internal_imports(py, py.stem)

    return library, scripts


# ---------------------------------------------------------------------------
# Graphviz emission
# ---------------------------------------------------------------------------

_LIB_NODE = '"lib::{m}" [label="{m}", fillcolor="#dbeafe"];'
_SCRIPT_NODE = '"scr::{s}" [label="{s}", shape=note, fillcolor="#fef9c3"];'
_GRAPH_HEADER = (
    "  rankdir=LR;\n"
    '  node [style="filled,rounded", shape=box,'
    ' fontname="Helvetica", fontsize=10];\n'
    '  edge [color="#94a3b8", arrowsize=0.7];'
)


def _architecture_dot(
    library: dict[str, set[str]], scripts: dict[str, set[str]]
) -> str:
    """Render the full project import graph as Graphviz source."""
    lines = [
        "digraph architecture {",
        _GRAPH_HEADER,
        '  subgraph cluster_lib {',
        '    label="library  (src/soaring_ctrw)"; style=dashed; color="#3b82f6";',
    ]
    for m in sorted(library):
        lines.append("    " + _LIB_NODE.format(m=m))
    lines.append("  }")
    lines.append('  subgraph cluster_scripts {')
    lines.append('    label="scripts"; style=dashed; color="#ca8a04";')
    for s in sorted(scripts):
        lines.append("    " + _SCRIPT_NODE.format(s=s))
    lines.append("  }")
    for m, deps in sorted(library.items()):
        for d in sorted(deps):
            lines.append(f'  "lib::{m}" -> "lib::{d}";')
    for s, deps in sorted(scripts.items()):
        for d in sorted(deps):
            lines.append(f'  "scr::{s}" -> "lib::{d}" [color="#eab308"];')
    lines.append("}")
    return "\n".join(lines)


def _script_dot(script: str, deps: set[str], library: dict[str, set[str]]) -> str:
    """Render one script's transitive library import map as Graphviz source."""
    # transitive closure within the library
    reachable: set[str] = set()
    stack = list(deps)
    while stack:
        m = stack.pop()
        if m in reachable:
            continue
        reachable.add(m)
        stack.extend(library.get(m, set()))

    lines = [
        f"digraph script_{script} {{",
        _GRAPH_HEADER,
        f'  "scr::{script}" [label="{script}", shape=note, fillcolor="#fef9c3"];',
    ]
    for m in sorted(reachable):
        lines.append("  " + _LIB_NODE.format(m=m))
    for d in sorted(deps):
        lines.append(f'  "scr::{script}" -> "lib::{d}" [color="#eab308"];')
    for m in sorted(reachable):
        for d in sorted(library.get(m, set())):
            if d in reachable:
                lines.append(f'  "lib::{m}" -> "lib::{d}";')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-script reference page
# ---------------------------------------------------------------------------

_SCRIPT_PAGE = """\
{title_underline}
``scripts/{name}.py``
{title_underline}

.. The narrative ("what / why") for this script is its module docstring,
   shown below; the import map and CLI are derived from source on each build.

Import map
----------

.. graphviz:: ../script_{name}.dot

Module reference
----------------

.. automodule:: {name}
   :members:
   :show-inheritance:

Command-line interface
----------------------

.. program-output:: {python} {repo_root}/scripts/{name}.py --help
"""


def _write_script_pages(
    out_dir: Path,
    scripts: dict[str, set[str]],
    library: dict[str, set[str]],
    repo_root: Path,
) -> None:
    """Write one ``.rst`` page and one ``.dot`` import map per script."""
    pages_dir = out_dir / "scripts"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for name, deps in sorted(scripts.items()):
        (out_dir / f"script_{name}.dot").write_text(
            _script_dot(name, deps, library), encoding="utf-8"
        )
        title = f"``scripts/{name}.py``"
        page = _SCRIPT_PAGE.format(
            name=name,
            title_underline="=" * len(title),
            repo_root=repo_root.as_posix(),
            python=Path(sys.executable).as_posix(),
        )
        (pages_dir / f"{name}.rst").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------------------
# Sphinx hook
# ---------------------------------------------------------------------------


def _generate(app) -> None:
    """``builder-inited`` callback: regenerate all derived doc inputs."""
    srcdir = Path(app.srcdir)
    repo_root = srcdir.parent
    out_dir = srcdir / "_generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    library, scripts = _scan(repo_root)
    (out_dir / "architecture.dot").write_text(
        _architecture_dot(library, scripts), encoding="utf-8"
    )
    _write_script_pages(out_dir, scripts, library, repo_root)


def setup(app):
    """Register the extension with Sphinx."""
    app.connect("builder-inited", _generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
