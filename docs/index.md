# soaring-ctrw

Companion code for *"How directional persistence shapes the Hurst-exponent crossover
of thermal soaring flights"* (Di Sante, 2026), extending the empirical
analysis of Vilpellet, Darmon & Benzaquen
([arXiv:2601.01293](https://arxiv.org/abs/2601.01293)).

A cross-country soaring flight is modelled as a renewal sequence of
**cycles**, each made of three phases — *transition* (T), *search* (S),
*climb* (C). Heading is carried from cycle to cycle by a Gaussian random
walk on the circle with dispersion $\sigma_\theta$, the single
phenomenological parameter, calibrated per aircraft against the universal
empirical Hurst exponent $H \approx 0.88$.

This site is generated **from the source code**: every API page comes from
the docstrings, every script page from the script's own module docstring and
its live `--help`, and every dependency graph from the actual imports. It is
rebuilt — and gated — in CI, so it cannot drift out of date. If you change a
module, class, or script, its documentation changes with it. See
{doc}`contributing-docs` for the maintenance contract.

## Where to start

- {doc}`installation` — set up the environment (note the Python 3.14 editable-install caveat).
- {doc}`model` — the model in one page, linking into the API.
- {doc}`pipeline` — the end-to-end run order that reproduces the figures.
- {doc}`architecture` — how the modules and scripts depend on each other.
- {doc}`scripts` — one reference page per script (purpose, import map, CLI).
- {doc}`api` — the full library API reference.

```{toctree}
:hidden:
:maxdepth: 2

installation
model
pipeline
architecture
scripts
api
contributing-docs
```
