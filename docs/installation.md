# Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" --config-settings editable_mode=compat
```

The package installs as `soaring_ctrw`; import submodules directly, e.g.
`from soaring_ctrw.model import SoaringConfig`.

## The `editable_mode=compat` flag (Python 3.14)

The flag is **required on Python 3.14**, which ignores hidden `.pth` files
(those starting with `_`) for security reasons. Without it, setuptools' default
editable install writes a `__editable__.<project>.pth` file that is silently
skipped, and `import soaring_ctrw` fails with `ModuleNotFoundError` even though
`pip show soaring-ctrw` reports it as installed. The compat mode falls back to
the legacy `easy-install.pth` scheme that Python 3.14 still loads. Requires
`setuptools >= 64`.

## Building the documentation

The docs have their own optional-dependency group and need the Graphviz `dot`
binary (for the dependency graphs):

```bash
pip install -e ".[docs]" --config-settings editable_mode=compat
# macOS:  brew install graphviz
# Debian/Ubuntu:  sudo apt-get install graphviz
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. To reproduce the strict CI build (which
turns warnings into errors), add `-W --keep-going`. See {doc}`contributing-docs`
for how the build keeps itself from going stale.
