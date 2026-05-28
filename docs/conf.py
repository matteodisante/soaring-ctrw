"""Sphinx configuration for soaring-ctrw.

Design goal: the documentation is generated from the source of truth
(docstrings + the live code) so it cannot silently go stale. See
``docs/contributing-docs.md`` for the maintenance contract.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DOCS_DIR.parent

# Make the local extension, the scripts (a flat directory, not a package),
# and the src package importable by autodoc.
sys.path.insert(0, str(_DOCS_DIR / "_ext"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# The ``program-output`` directive runs each script's ``--help`` in a fresh
# subprocess. Export ``src`` on PYTHONPATH so that subprocess can import
# ``soaring_ctrw`` regardless of how (or whether) the package is installed
# editable — in particular under Python 3.14, which ignores the
# ``__editable__*.pth`` files setuptools writes by default. This keeps the
# docs build self-sufficient.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(_REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p
)

# -- Project information -----------------------------------------------------

project = "soaring-ctrw"
author = "Matteo Di Sante"
copyright = "2026, Matteo Di Sante"
# Version is read from the installed package metadata, never hard-coded here.
release = _pkg_version("soaring-ctrw")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",        # pull docstrings from the code
    "sphinx.ext.autosummary",    # auto-generate API stub pages (recursive)
    "sphinx.ext.napoleon",       # NumPy / Google docstring sections
    "sphinx.ext.viewcode",       # "[source]" links next to each object
    "sphinx.ext.intersphinx",    # cross-link to numpy / scipy / python docs
    "sphinx.ext.mathjax",        # render the LaTeX in the docstrings
    "sphinx.ext.graphviz",       # render the generated dependency graphs
    "sphinx.ext.githubpages",    # emit .nojekyll so GitHub Pages serves _static/
    "myst_parser",               # narrative pages written in Markdown
    "sphinxcontrib.programoutput",  # capture live ``--help`` output
    "soaring_autodocs",          # local: regenerate dot graphs + script pages
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True          # build API stubs on every run
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"     # render type hints in the body
autodoc_class_signature = "separated"
# class docstring only (the __init__ is documented separately)
autoclass_content = "class"
autodoc_inherit_docstrings = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = True

# Unresolved cross-references (e.g. external numpy types in type hints)
# are left as plain text rather than warned about, so the strict ``-W``
# build only fails on real problems (broken directives, dead toctree
# entries, missing modules, malformed docstrings).
nitpicky = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
intersphinx_timeout = 10

# -- Graphviz ----------------------------------------------------------------

graphviz_output_format = "svg"

# -- programoutput -----------------------------------------------------------

programoutput_prompt_template = "$ %(command)s\n\n%(output)s"

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"soaring-ctrw {release}"
html_static_path = ["_static"]
# Canonical URL of the published site (GitHub Pages, project page).
html_baseurl = "https://matteodisante.github.io/soaring-ctrw/"
