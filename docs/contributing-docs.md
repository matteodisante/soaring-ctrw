# Maintaining the docs (anti-rot contract)

The guiding rule of this documentation is: **the substance lives in the code,
not in parallel prose.** Outdated documentation is worse than none, so the docs
are generated from the source of truth and gated in CI. This page is the
contract that keeps it that way.

## What is generated automatically (never edit by hand)

| Content | Source of truth | Mechanism |
| --- | --- | --- |
| API reference ({doc}`api`) | docstrings in `src/soaring_ctrw` | `autodoc` + **recursive** `autosummary` — discovers new modules/classes/functions automatically |
| Per-script pages ({doc}`scripts`) | each script's module docstring | `automodule` + a `:glob:` toctree — a new script gets a page automatically |
| Script CLI options | the live `argparse` parser | `program-output` runs the real `--help` at build time |
| Dependency graphs ({doc}`architecture`) | the actual `import` statements | `docs/_ext/soaring_autodocs.py` re-parses the source tree on every build |
| Project version | installed package metadata | `conf.py` reads it via `importlib.metadata` |

Because each of these is re-derived on every build, renaming or removing a
module, class, attribute, or script is reflected in the docs with **zero**
manual edits — and the strict build (below) *fails* if a doc references
something that no longer exists.

## What is hand-written prose (keep it thin)

- {doc}`index`, {doc}`installation`, {doc}`model` — conceptual narrative that
  links **into** the generated API rather than restating it.
- {doc}`pipeline` — the intended run order, an editorial choice the code does
  not encode. Update it when you add/reorder a pipeline step.

When you write prose, link to the API object (e.g.
`` {class}`~soaring_ctrw.model.SoaringConfig` ``) instead of copying its
signature or behaviour. That way the prose points at the live definition and
cannot contradict it.

## The two-part freshness gate

1. **Warnings-as-errors build.** CI runs
   ```bash
   sphinx-build -b html -W --keep-going docs docs/_build/html
   ```
   With `-W`, a cross-reference or `automodule` directive that points at a
   removed/renamed object, a dead toctree entry, or a malformed docstring turns
   into a build **failure**. You cannot merge docs that reference code that is
   no longer there.

2. **Docstring-coverage gate.** CI runs `interrogate` (configured in
   `pyproject.toml`, `fail-under = 100` for the package). Adding a public
   function/class/method without a docstring fails the build. Single-underscore
   internal helpers and `@property` accessors are exempt (the latter are
   documented in their class docstrings).

Both run in `.github/workflows/docs.yml` on every push/PR.

## Adding things

- **New library module/class/function** → just write its docstring. It appears
  in {doc}`api` and in the dependency graph on the next build.
- **New script** → give it a module docstring (the "what / why") and a standard
  `argparse` parser. It gets a {doc}`scripts` page, an import map, and a
  rendered `--help` automatically.

## Building locally

```bash
pip install -e ".[docs]" --config-settings editable_mode=compat
sphinx-build -b html docs docs/_build/html          # fast iteration
sphinx-build -b html -W --keep-going docs docs/_build/html   # what CI runs
interrogate -c pyproject.toml src                    # the coverage gate
```

The generated artefacts (`docs/_build/`, `docs/_generated/`) are git-ignored
and rebuilt from scratch; never commit or hand-edit them.
