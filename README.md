# soaring-ctrw

A cycle-based continuous-time random walk (CTRW) model with angular
persistence for cross-country soaring flights. Companion code for the
manuscript *"A cycle-based model for the universal Hurst exponent in
thermal soaring flights"* (Di Sante, 2026), which extends the empirical
analysis of Vilpellet, Darmon & Benzaquen
([arXiv:2601.01293](https://arxiv.org/abs/2601.01293), VDB hereafter).

## Model in one screen

A flight is a renewal sequence of soaring **cycles**. Each cycle has
three phases — *transition* T, *search* S, *climb* C — with i.i.d.
phase-duration scheduler

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
fix is the cycle-to-cycle heading dispersion ``σ_θ``. The manuscript
shows that the iso-contour ``H_eff = 0.88`` is pinned at a common
``σ_θ`` essentially independently of the aircraft-specific
``μ_T`` — this is the analytical origin of the empirical universality
``H ≈ 0.88``.

## Repository layout

```
soaring-ctrw/
├── src/
│   ├── distributions.py   # Pareto, Lomax, Exponential, Mittag-Leffler
│   ├── model.py           # SoaringConfig, SearchMotionConfig, ClimbMotionConfig, ...
│   ├── simulation.py      # simulate_single, simulate_ensemble, interpolate_trajectory
│   ├── observables.py     # time-averaged MSD (FFT), Hurst-exponent fit
│   ├── calibration.py     # read/write outputs/data/calibration/<aircraft>.yaml; apply_calibration
│   ├── cache.py           # script-side NPZ + manifest cache for Monte-Carlo runs
│   └── paths.py           # repo-relative output paths
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
pip install -e ".[dev]"
```

The package exposes its modules at the top level (``model``,
``simulation``, ``distributions``, ``observables``, ``paths``), so
``from model import SoaringConfig`` works out of the box once
installed in editable mode.

## Reproducing the figures

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

The `configs/<aircraft>.yaml` files keep `angular.sigma_theta: null`;
any script that needs `σ_θ` reads it from
`outputs/data/calibration/<aircraft>.yaml` via
`src/calibration.py::apply_calibration` (no manual promotion step is
required). All per-aircraft derived quantities (`σ_θ⋆`,
`tau_turn_calibrated`, `m_{1/2}(α)`, `t_c`, …) live in that single
YAML; sections are merged in place by `write_calibration_section`, so
successive runs of different calibrators never overwrite each other.

### Individual commands

Calibrate the cycle-to-cycle heading dispersion σ_θ (1-D scan against
the empirical H_eff = 0.88):

```bash
python scripts/estimate_sigma_theta.py \
    --n-sigma 16 --n-trajectories 1000 --total-time 15000 \
    --fit-min 10 --fit-max 7000
```

This populates ``outputs/figures/estimate_sigma_theta_*_overlay.pdf``
(one overlay per aircraft with the bare- and full-cycle
``H_eff(σ_θ)`` curves) and caches the scan in
``outputs/data/estimate_sigma_theta/*.npz``. Add ``--cache reuse`` to
reuse a cached scan instead of re-running the Monte Carlo. Without
``--write`` only the figures + cache are produced; ``σ_θ⋆`` is
printed to stdout/log but not persisted.

Quick stand-alone table of `m_{1/2}(α)` and the calibrated
`τ_turn^S = (π/2) / (Ω_S · m_{1/2}(α_S))`:

```bash
python scripts/compute_ml_median_and_tau_turn.py
```

Pre-asymptotic crossover time `t_c = 2⟨T⟩/σ_θ²` (requires step 3
to have been run with `--write`):

```bash
python scripts/compute_critical_time.py --write
```

This also prints the per-phase mean durations
`⟨τ_T⟩, ⟨τ_S⟩, ⟨τ_C⟩`, the mean cycle duration `⟨T⟩`, and the
number-of-cycles crossover `n_c = 2/σ_θ²`, all of which are stored
in the `critical_time` section of the calibration YAML.

The per-phase MSD figures (3-panel, one column per aircraft) are
reproduced with

```bash
python scripts/plot_per_phase_msd.py
```

The combined MSD comparison across aircraft (overlay + local-slope
sub-panel + rescaled-by-v_xy² figure with power-law fits over
``[10, 7000]`` s) is reproduced with

```bash
python scripts/plot_msd_all_aircraft.py
```

The four side-by-side comparisons between simulated MSDs and the
closed-form expressions of the manuscript (Eq. 10 for the search
MSD, Eq. 11 for the climb MSD, Eq. 23 for the cycle-counted MSD,
and Eq. 26 for the local Hurst exponent `H_eff(N)`) are produced by

```bash
python scripts/compare_msd_to_theory.py
```

It reads the per-phase MSD cache (step 2) and the calibrated
`σ_θ` (step 3) and runs a small live cycle-counted simulation for
the last two panels.

The sample-trajectory figure (2×2 panel with all four sub-plots
rendered as identical squares) is reproduced with

```bash
python scripts/plot_trajectory.py
```

The analytic phase-duration CCDFs of Fig. 2 of the manuscript come from

```bash
python scripts/plot_phase_durations.py
```

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
