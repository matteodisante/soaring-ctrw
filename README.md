# soaring-ctrw

A cycle-based continuous-time random walk (CTRW) model with angular
persistence for cross-country soaring flights. Companion code for the
manuscript *"A cycle-based model for the universal Hurst exponent in
thermal soaring flights"* (Di Sante, 2026), which extends the empirical
analysis of Vilpellet, Darmon & Benzaquen
([arXiv:2601.01293](https://arxiv.org/abs/2601.01293), VDB hereafter).

[![docs](https://github.com/matteodisante/soaring-ctrw/actions/workflows/docs.yml/badge.svg)](https://github.com/matteodisante/soaring-ctrw/actions/workflows/docs.yml)

**Documentation:** <https://matteodisante.github.io/soaring-ctrw/> — full API
reference, a page per script, and dependency graphs, all generated from the
source. This README is a quickstart; see the site for everything else.

## Model in one screen

A flight is a renewal sequence of soaring **cycles**. Each cycle has
three phases — *transition* T, *search* S, *climb* C — with i.i.d.
phase-duration scheduler.

- Transition and search: Lomax,
  ``P(τ > τ) = (1 + τ / τ_0)^{-μ}``.
- Climb: exponential,
  ``P(τ > τ) = exp(-τ / μ_C^eff)``.

The transition phase carries the heading from cycle to cycle through a
Gaussian random walk on the circle,
``θ_n = θ_{n-1} + η_n``, ``η_n ~ 𝒩(0, σ_θ²)``. The initial
``θ_0`` of each independent trajectory is uniform on ``[0, 2π)``.

Intra-phase dynamics:

- **Search** is a local CTRW with physical-duration stopping: ballistic
  legs of speed ``u_S`` and exponential duration ``τ_b ~ Exp(τ_b^S)``,
  interleaved with Mittag-Leffler turning waits
  ``τ_turn ~ ML(α_S, τ_turn^S)``. The direction is updated, after a
  fully completed wait, by
  ``ψ_{j+1} = ψ_j + ε_j · Ω_S · τ_turn_j``, ``ε_j = ±1``. The local
  CTRW stops when the cumulative *physical* time (legs + waits)
  reaches the Lomax-sampled search duration ``τ_S^n``; whichever
  component straddles ``τ_S^n`` is truncated, so the search physical
  duration is ``T_phys^S = Σ τ_b + Σ τ_turn = τ_S^n`` exactly.
- **Climb** is circular motion at radius ``r_0`` with per-cycle turn
  period ``T_turn_n ~ 𝒩(T_turn_mean, T_turn_std²)`` (clipped at
  ``0.2 · T_turn_mean``) plus a slow orographic drift ``v_drift`` with
  an independent uniform direction per cycle.

The single phenomenological parameter introduced beyond what the data
fix is the cycle-to-cycle heading dispersion ``σ_θ``, calibrated
per aircraft against the empirical ``H ≈ 0.88``.

## Repository layout

```
soaring-ctrw/
├── src/
│   └── soaring_ctrw/
│       ├── __init__.py
│       ├── distributions.py   # Pareto, Lomax, Exponential, Mittag-Leffler
│       ├── model.py           # SoaringConfig, SearchMotionConfig, ClimbMotionConfig, ...
│       ├── simulation.py      # simulate_single, simulate_ensemble, interpolate_trajectory
│       ├── observables.py     # time-averaged MSD (FFT), Hurst-exponent fit
│       ├── calibration.py     # read/write outputs/data/calibration/<aircraft>.yaml; apply_calibration
│       ├── cache.py           # script-side NPZ + manifest cache for Monte-Carlo runs
│       └── paths.py           # repo-relative output paths
├── configs/
│   ├── paragliders.yaml   # Table 1 of the manuscript
│   ├── hang_gliders.yaml
│   └── sailplanes.yaml
├── scripts/
│   ├── compute_ml_median_and_tau_turn.py  # m_{1/2}(α) and τ_turn^S = (π/2)/(Ω_S·m_{1/2})
│   ├── estimate_sigma_theta.py            # 1-D scan H_eff(σ_θ); finds σ_θ⋆ at H=0.88
│   ├── compute_critical_time.py           # crossover time t_c = 2⟨T⟩/σ_θ² from calibrated σ_θ
│   ├── plot_per_phase_msd.py              # 3-panel per-phase EA-MSD with log-log fits
│   ├── plot_msd_all_aircraft.py           # MSD overlay + local-slope panel + rescaled MSD
│   ├── compare_msd_to_theory.py           # simulated MSD vs paper Eqs. 10, 11, 23, 26
│   ├── plot_trajectory.py                 # 2×2 sample-trajectory figure (uniform square axes)
│   └── plot_phase_durations.py            # analytic CCDF panels (transition / search / climb)
├── tests/                       # pytest suite
├── docs/                        # Sphinx documentation (auto-generated from source)
├── outputs/                     # all script outputs land here
│   ├── figures/
│   └── data/
├── paper/                       # manuscript and figures used by the .tex
└── pyproject.toml
```

Every script writes its artefacts under ``outputs/`` (figures as PDF
or PNG under ``outputs/figures/``, numerical arrays as NPZ under
``outputs/data/``). The default paths can be overridden with
``--figures-dir`` and ``--data-dir``.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" --config-settings editable_mode=compat
```

The package is installed as `soaring_ctrw`; modules are imported with
the usual namespace, e.g. ``from soaring_ctrw.model import
SoaringConfig``.

The `editable_mode=compat` flag is required on Python 3.14, which
ignores hidden `.pth` files (those starting with `_`) for security
reasons. Without it, setuptools' default editable install writes a
`__editable__.<project>.pth` file that is silently skipped and the
package fails to import. The compat mode falls back to the legacy
`easy-install.pth` scheme that Python 3.14 still loads. Requires
`setuptools >= 64`.

## Documentation

The full documentation lives at
**<https://matteodisante.github.io/soaring-ctrw/>**. It is built with Sphinx
from the docstrings, the scripts, and the actual imports — so it stays in sync
with the code by construction — and contains:

- the complete **API reference** (every module/class/function from its docstring);
- **one page per script** (purpose, import map, and the live `--help`);
- auto-derived **dependency graphs** ([architecture](https://matteodisante.github.io/soaring-ctrw/architecture.html));
- the **reproduction pipeline** ([pipeline](https://matteodisante.github.io/soaring-ctrw/pipeline.html)).

A GitHub Actions workflow (`.github/workflows/docs.yml`) rebuilds the site on
every push to `main` and publishes it to GitHub Pages. The same workflow gates
every push/PR with a warnings-as-errors build and a docstring-coverage check,
so documentation that references removed or renamed code fails CI. See
[`docs/contributing-docs.md`](docs/contributing-docs.md) for the maintenance
contract.

To build it locally:

```bash
pip install -e ".[docs]" --config-settings editable_mode=compat
# graphviz is needed for the dependency graphs:
#   macOS: brew install graphviz   •   Debian/Ubuntu: sudo apt-get install graphviz
sphinx-build -b html docs docs/_build/html   # open docs/_build/html/index.html
```

## Reproducing the figures

For per-script details (every option, the import map, the live `--help`) see
the [scripts reference](https://matteodisante.github.io/soaring-ctrw/scripts.html)
and the [pipeline page](https://matteodisante.github.io/soaring-ctrw/pipeline.html).

### Recommended run order

All scripts can be invoked independently, but several depend on
artefacts produced by others. The end-to-end pipeline that reproduces
every figure of the manuscript is:

1. **Search-phase calibration** — compute `m_{1/2}(α_S)` and the
   calibrated `τ_turn^S`. With `--write` they are merged into
   `outputs/data/calibration/<aircraft>.yaml` under the
   `mittag_leffler` section:
   ```bash
   python scripts/compute_ml_median_and_tau_turn.py --write
   ```

2. **Per-phase MSD** — 3-panel EA-MSD figure with log-log fits and
   per-aircraft MSD `.npz` files used by step 5 below:
   ```bash
   python scripts/plot_per_phase_msd.py
   ```

3. **σ_θ calibration** — 1-D scan against the empirical
   `H_eff = 0.88`; with `--write` the full-cycle `σ_θ⋆` is merged
   into the `sigma_theta` section of the calibration YAML:
   ```bash
   python scripts/estimate_sigma_theta.py \
       --n-sigma 16 --n-trajectories 1000 --total-time 15000 \
       --fit-min 10 --fit-max 7000 --write
   ```

4. **Pre-asymptotic crossover time** — reads the calibrated
   `σ_θ` (full-cycle) from step 3 and the mean cycle duration
   `⟨T⟩ = ⟨τ_T⟩ + ⟨τ_S⟩ + ⟨τ_C⟩` from `configs/<aircraft>.yaml`,
   prints the breakdown, and (with `--write`) merges the result into
   the `critical_time` section:
   ```bash
   python scripts/compute_critical_time.py --write
   ```

5. **Final figures** — simulation MSDs vs theory, MSD overlay,
   trajectories, phase-duration CCDFs:
   ```bash
   python scripts/plot_msd_all_aircraft.py
   python scripts/compare_msd_to_theory.py
   python scripts/plot_trajectory.py
   python scripts/plot_phase_durations.py
   ```

The `configs/<aircraft>.yaml` files keep `angular.sigma_theta: null`; any
script that needs `σ_θ` reads it from `outputs/data/calibration/<aircraft>.yaml`
via `apply_calibration`, and all derived per-aircraft quantities live in that
single YAML (sections merged in place, so calibrators never overwrite each
other). Full details are in the
[`calibration` API docs](https://matteodisante.github.io/soaring-ctrw/_generated/api/soaring_ctrw.calibration.html).

## Tests

```bash
pytest -q
```

The suite checks the configuration loading, the search-phase
invariants (``T_phys^S = Σ τ_b + Σ τ_turn = τ_S^n`` in full mode and
``T_phys^S = τ_S^n`` in bare mode), the ``σ_θ = 0`` straight-line
limit of the transition, the ensemble-averaged MSD on synthetic
ballistic and diffusive trajectories, and the recovery of analytic
Hurst exponents on synthetic power-law MSDs.

## References

- J. Vilpellet, A. Darmon, M. Benzaquen, *From Random Walks to Thermal
  Rides: Universal Anomalous Transport in Soaring Flights*,
  [arXiv:2601.01293](https://arxiv.org/abs/2601.01293) (2026).
- R. Metzler, J. Klafter, Phys. Rep. 339, 1 (2000).
- V. Zaburdaev, S. Denisov, J. Klafter, Rev. Mod. Phys. 87, 483 (2015).
- E. A. Codling, M. J. Plank, S. Benhamou, J. R. Soc. Interface 5, 813
  (2008).
- O. Bénichou, C. Loverdo, M. Moreau, R. Voituriez, Rev. Mod. Phys. 83,
  81 (2011).
