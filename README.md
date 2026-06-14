# soaring-ctrw

A cycle-based continuous-time random walk (CTRW) model with angular
persistence for cross-country soaring flights. Companion code for the
manuscript *"How directional persistence shapes the Hurst-exponent crossover
of thermal soaring flights"* (Di Sante, 2026), which extends the empirical
analysis of Vilpellet, Darmon & Benzaquen
([arXiv:2601.01293](https://arxiv.org/abs/2601.01293), VDB hereafter).

[![docs](https://github.com/matteodisante/soaring-ctrw/actions/workflows/docs.yml/badge.svg)](https://github.com/matteodisante/soaring-ctrw/actions/workflows/docs.yml)

**Documentation:** <https://matteodisante.github.io/soaring-ctrw/> — full API
reference, a page per script, and dependency graphs, all generated from the
source.

---

## Model in one screen

A flight is a renewal sequence of soaring **cycles**. Each cycle has three
phases — *transition* T, *search* S, *climb* C — with i.i.d. phase-duration
scheduler.

- Transition and search: Lomax, `P(τ > t) = (1 + t/τ_0)^{-μ}`.
- Climb: exponential, `P(τ > t) = exp(-t/μ_C^eff)`.

The transition phase carries the heading from cycle to cycle through a
Gaussian random walk on the circle, `θ_n = θ_{n-1} + η_n`,
`η_n ~ 𝒩(0, σ_θ²)`. The initial `θ_0` of each independent trajectory
is uniform on `[0, 2π)`.

Intra-phase dynamics:

- **Search** is a local CTRW with physical-duration stopping: ballistic
  legs of speed `u_S` and exponential duration `τ_b ~ Exp(τ_b^S)`,
  interleaved with Mittag-Leffler turning waits `τ_turn ~ ML(α_S, τ_turn^S)`.
  The search stops when the cumulative physical time (legs + waits) reaches
  the Lomax-sampled duration `τ_S^n` exactly.
- **Climb** is circular motion at radius `r_0` with per-cycle turn period
  `T_turn_n ~ 𝒩(T_turn_mean, T_turn_std²)` (clipped at `0.2·T_turn_mean`)
  plus a slow orographic drift `v_drift` with an independent uniform direction
  per cycle.

The only free parameter is the cycle-to-cycle heading dispersion `σ_θ`,
calibrated per aircraft against the empirical `H ≈ 0.88`. Calibrated values
(full-cycle variant, with sub-ensemble standard errors):

| Class         | σ_θ* (rad)       | n_c  | t_c (s)  |
|---------------|------------------|------|----------|
| Paragliders   | 0.412 ± 0.009    | 11.8 | ~4 900   |
| Hang gliders  | 0.242 ± 0.005    | 34.2 | ~15 100  |
| Sailplanes    | 0.556 ± 0.014    |  6.5 |  ~3 100  |

---

## Repository layout

```
soaring-ctrw/
├── src/
│   └── soaring_ctrw/
│       ├── __init__.py
│       ├── distributions.py   # Pareto, Lomax, Exponential, Mittag-Leffler samplers
│       ├── model.py           # SoaringConfig, PhaseConfig, SearchMotionConfig, ClimbMotionConfig
│       ├── simulation.py      # simulate_single, simulate_ensemble, interpolate_trajectory
│       ├── observables.py     # EA-MSD (msd_ensemble, msd_ensemble_percentiles) + HurstFit / fit_hurst
│       ├── theory.py          # ALL analytical closed-form expressions (single source of truth)
│       ├── calibration.py     # read/write outputs/data/calibration/<aircraft>.yaml
│       ├── cache.py           # script-side NPZ + manifest cache for Monte Carlo runs
│       └── paths.py           # repo-relative output paths (CONFIGS_DIR, DATA_DIR, FIGURES_DIR)
├── configs/
│   ├── paragliders.yaml       # Table 1 of the manuscript
│   ├── hang_gliders.yaml
│   └── sailplanes.yaml
├── scripts/
│   ├── compute_ml_median_and_tau_turn.py   # m_{1/2}(α) and τ_turn^S calibration
│   ├── estimate_sigma_theta.py             # 1-D scan H_eff(σ_θ); finds σ_θ⋆ at H=0.88
│   ├── compute_critical_time.py            # crossover time t_c = 2⟨T⟩/σ_θ²
│   ├── plot_per_phase_msd.py               # 3-panel per-phase EA-MSD with log-log fits
│   ├── plot_msd_all_aircraft.py            # MSD overlay + local slope + rescaled collapse
│   ├── compare_msd_to_theory.py            # simulated MSD vs closed-form expressions
│   ├── plot_variance_convergence.py        # Appendix D: convergence of Var(τ_T) estimator
│   ├── plot_trajectory.py                  # sample 2-D cycle trajectory
│   ├── plot_phase_durations.py             # analytic CCDF panels (T / S / C)
│   └── scan_parameter_sensitivity.py       # ±20% sensitivity of H_fit to eye-read parameters
├── tests/                     # pytest suite (56 tests)
├── docs/                      # Sphinx documentation
├── outputs/
│   ├── figures/               # all PDF/PNG figures land here
│   └── data/                  # NPZ caches + calibration YAMLs + sensitivity CSV
└── pyproject.toml
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" --config-settings editable_mode=compat
```

What each line does, and why it is necessary:

- **`python -m venv .venv && source .venv/bin/activate`** — creates and
  activates a *virtual environment*, an isolated Python with its own `pip`
  and packages, separate from the system Python. This avoids version
  conflicts with whatever else is installed on the machine and is the
  standard workflow for any scientific Python project. To use the project
  in a new shell, just re-run `source .venv/bin/activate`. To wipe
  everything and start over: `rm -rf .venv`.

- **`pip install -e ".[dev]"`** — the library code lives in
  `src/soaring_ctrw/`, but the scripts in `scripts/` `import soaring_ctrw`.
  Without this command, Python would not find the package and every
  script would fail with `ModuleNotFoundError`. The `-e` (*editable*) flag
  points the installer at the source tree instead of copying the files,
  so any edit to `src/soaring_ctrw/` is picked up immediately on the next
  `python` invocation — no re-install needed. The `[dev]` extra also
  pulls in test/lint/docs dependencies declared in `pyproject.toml`.

- **`--config-settings editable_mode=compat`** — needed only on **Python
  3.14** (the version this project targets). The default editable install
  drops a hidden `__editable__.soaring_ctrw-*.pth` file in
  `site-packages`; Python 3.14's import loader silently ignores `.pth`
  files whose name starts with `__`, so `import soaring_ctrw` then fails
  with no clear error. The `compat` mode uses the legacy
  `easy-install.pth` scheme that 3.14 still loads. Requires
  `setuptools ≥ 64`. On Python ≤ 3.13 the flag is harmless and can be
  dropped.

Verify the install:

```bash
python -m pytest -q          # should print "56 passed"
python -c "from soaring_ctrw.theory import G_N; print('OK')"
```

---

## Reproducing all figures — full pipeline

Run the steps **in order**. Each step writes its outputs under `outputs/` and
later steps read those outputs. Total wall time on a laptop: roughly 10–20 min
(dominated by the large sailplane cycle-counted ensemble in step 6).

### Step 1 — Mittag-Leffler calibration of τ_turn^S

Computes the dimensionless median `m_{1/2}(α_S)` of the ML distribution and
derives `τ_turn^S = (π/2) / (Ω_S · m_{1/2})` for each aircraft. Prints a
table and (with `--write`) merges results into the calibration YAMLs.

```bash
python scripts/compute_ml_median_and_tau_turn.py --write
```

Expected output (printed table, columns: aircraft, α_S, m_{1/2}, Ω_S, τ_turn^S, YAML):

```
aircraft       alpha_S   m_{1/2}       Omega_S   tau_turn^S       YAML
paragliders     0.6000   0.605685     0.2700     9.6053    9.605
hang_gliders    0.6000   0.605685     0.2700     9.6053    9.605
sailplanes      0.4000   0.580269     0.2900     9.3345    9.334
```

A `*** MISMATCH` flag and a `UserWarning` are raised if the calibrated value
differs from the YAML by more than 5%.

### Step 2 — Per-phase EA-MSD

Simulates `2 000` trajectories of `50` cycles each per aircraft. Writes one
NPZ per aircraft to `outputs/data/per_phase_msd/` (needed by step 6).

```bash
python scripts/plot_per_phase_msd.py --n-trajectories 2000
```

**Output figure:** `outputs/figures/per_phase_msd_combined.pdf`
(→ Fig. 4 of the manuscript)

Runtime: ~2 min.

### Step 3 — σ_θ calibration

Scans `σ_θ` on a 16-point grid in `[0.05, 1.5]` rad and interpolates to
find `σ_θ⋆` at `H_eff = 0.88`. With `--write`, writes `σ_θ⋆` into the
calibration YAMLs (required by all subsequent steps).

```bash
python scripts/estimate_sigma_theta.py \
    --n-sigma 16 \
    --n-trajectories 2000 \
    --total-time 15000 \
    --fit-min 10 \
    --fit-max 7000 \
    --fit-lag-spacing linear \
    --n-log-lags 40 \
    --n-groups 10 \
    --write
```

**Key new options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--fit-lag-spacing` | `linear` | `linear` or `log`: lag spacing for the OLS fit. `log` uses ~`--n-log-lags` lags uniformly spaced in log(Δ), which avoids overweighting large lags. The two conventions differ by ±0.01–0.02 in H. |
| `--n-log-lags` | `40` | Number of log-spaced lags when `--fit-lag-spacing log`. |
| `--n-subensembles` | `10` | Number of disjoint sub-ensembles used to estimate the replica-level uncertainty on `σ_θ⋆`. |

**Output figure:** `outputs/figures/estimate_sigma_theta_{aircraft}_overlay.pdf`
(→ Fig. 4 of the manuscript; one panel per aircraft)

Calibrated values written to YAML:

```
paragliders:  σ_θ⋆ = 0.412 ± 0.009 rad
hang_gliders: σ_θ⋆ = 0.242 ± 0.005 rad
sailplanes:   σ_θ⋆ = 0.556 ± 0.014 rad
```

(Uncertainties are sub-ensemble standard errors, not OLS standard errors.)

Runtime: ~3 min.

### Step 4 — Crossover time

Reads the calibrated `σ_θ⋆` and computes `n_c = 2/σ_θ²` and
`t_c = n_c · ⟨T⟩`. With `--write`, merges into the calibration YAMLs.

```bash
python scripts/compute_critical_time.py --write
```

### Step 5 — Total MSD and rescaled collapse

Simulates `2 000` full-model trajectories per aircraft. Produces the total
EA-MSD with local-slope panel and the rescaled `δ²/v_xy²` collapse.
Prints both `H_fit (linear-lags)` and `H_fit (log-lags)` after each aircraft
so the lag-spacing sensitivity is immediately visible.

Pass `--write-hfit` to persist H_fit and its replica s.e. to each aircraft's
calibration YAML (section `h_fit`). This is required before running Step 5a.

```bash
python scripts/plot_msd_all_aircraft.py \
    --n-trajectories 2000 \
    --fit-lag-spacing linear \
    --n-log-lags 40 \
    --n-groups 10 \
    --write-hfit
```

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--fit-lag-spacing` | `linear` | Lag spacing for `H_fit`. Passed to `fit_hurst()`. |
| `--n-log-lags` | `40` | Log-spaced lag count for `--fit-lag-spacing log`. |
| `--n-groups` | `10` | Disjoint sub-ensembles for the replica-level H uncertainty. |
| `--write-hfit` | off | Write H_fit + replica s.e. to calibration YAMLs. |

**Output figures:**
- `outputs/figures/msd_all_aircraft_raw.pdf` → Fig. 5 (MSD + local slope)
- `outputs/figures/msd_all_aircraft_rescaled.pdf` → Fig. 6 (rescaled collapse)

Runtime: ~2 min.

### Step 5a — Generate LaTeX paper macros

Reads calibration YAMLs (σ_θ⋆ from Step 3, H_fit from Step 5) and writes
`outputs/data/paper_macros.tex` with `\newcommand` definitions for every
numeric quantity in the paper. The paper includes this file via
`\InputIfFileExists`.

```bash
python scripts/write_paper_macros.py \
    --out ../12_06_26_soaring-ctrw-paper_slides/paper_macros.tex
```

(Adjust `--out` to wherever your paper `.tex` file lives.)
After this step, compiling `soaring_ctrw.tex` will use the code-computed values
for H_fit and σ_θ⋆ rather than any value typed by hand.

### Step 6 — Comparison with closed-form theory

Four comparison figures: search MSD, climb MSD (with exact reference curve),
cycle-counted MSD, and effective exponent H_eff(N). The sailplane cycle-counted
ensemble is enlarged to 120 000 trajectories because `μ_T = 2.62` makes the
variance estimator very slow to converge.

```bash
python scripts/compare_msd_to_theory.py \
    --n-traj-cycles 10000 \
    --n-traj-cycles-by-aircraft sailplanes=120000 \
    --n-cycles 60
```

**Output figures:**
- `outputs/figures/compare_theory_msd_search.pdf` → Fig. 7 (search MSD)
- `outputs/figures/compare_theory_msd_climb.pdf` → Fig. 8 (climb MSD; includes exact reference dotted curve)
- `outputs/figures/compare_theory_msd_cycles.pdf` → Fig. 9 (cycle-counted MSD)
- `outputs/figures/compare_theory_Heff.pdf` → Fig. 10 (H_eff(N) crossover)
- `outputs/figures/compare_theory_components.pdf` → Fig. 2 (G_N, s_G, w, H_eff components)

Runtime: ~5–15 min (dominated by sailplanes at 120 000 trajectories).

### Step 7 — Appendix D: variance convergence

```bash
python scripts/plot_variance_convergence.py
```

**Output figure:** `outputs/figures/variance_convergence.pdf` → App. D

Runtime: ~1 min.

### Step 8 — Sample trajectory and phase-duration CCDFs

```bash
python scripts/plot_trajectory.py
python scripts/plot_phase_durations.py
```

**Output figures:**
- `outputs/figures/example_trajectories.pdf` → Fig. 1
- `outputs/figures/phase_durations_ccdf.pdf` → Fig. 3

Runtime: <1 min each.

### Step 9 (optional) — Parameter sensitivity

Scans ±20% perturbations of the eye-read parameters (τ_0^T, μ_T, v_xy,
T_turn_mean, T_turn_std, v_drift, σ_θ) and reports how much each shift moves
H_fit. The result is printed to stdout and saved as
`outputs/data/sensitivity_scan.csv`.

```bash
python scripts/scan_parameter_sensitivity.py \
    --n-cycles 300 \
    --n-traj 500 \
    --perturb 0.20
```

Runtime: ~5–10 min (7 parameters × 2 directions × 3 aircraft).

---

## Figure → script map

| Figure | Script | Key options |
|--------|--------|-------------|
| Fig. 1: sample cycle | `plot_trajectory.py` | `--seed` |
| Fig. 2: G_N components | `compare_msd_to_theory.py` | `--n-traj-cycles 10000` |
| Fig. 3: phase CCDFs | `plot_phase_durations.py` | — |
| Fig. 4: phase-cond. MSD | `plot_per_phase_msd.py` | `--n-trajectories 2000` |
| Fig. 5: σ_θ calibration | `estimate_sigma_theta.py` | `--n-sigma 16 --n-trajectories 2000` |
| Fig. 6: total MSD | `plot_msd_all_aircraft.py` | `--n-trajectories 2000` |
| Fig. 7: rescaled collapse | `plot_msd_all_aircraft.py` | same run as Fig. 6 |
| Fig. 8: search MSD | `compare_msd_to_theory.py` | — |
| Fig. 9: climb MSD | `compare_msd_to_theory.py` | — |
| Fig. 10: cycle MSD | `compare_msd_to_theory.py` | `--n-traj-cycles-by-aircraft sailplanes=120000` |
| Fig. 11: H_eff(N) | `compare_msd_to_theory.py` | same run as Fig. 10 |
| App. D: var. conv. | `plot_variance_convergence.py` | — |

---

## Methodological notes

### Lag spacing in the OLS fit

The default `--fit-lag-spacing linear` passes all integer lags in `[fit_min,
fit_max]` to the OLS regression. On a uniform time grid this overweights large
lags by ∝ Δ per unit log Δ, pulling the slope toward the large-lag local
slope. `--fit-lag-spacing log` subsamples at `--n-log-lags` lags uniformly
spaced in log Δ, which is the standard convention for power-law MSD fitting.
On crossover-shaped MSDs the two conventions differ by ±0.01–0.02 in H with a
class-dependent sign; this is declared as a systematic and reported alongside
the point estimate.

### H_fit uncertainty: replica SE vs OLS SE

The OLS standard error on H_fit (printed in legends when `lag_spacing=linear`)
assumes the log(MSD) points are independent. Adjacent MSD lags of the same
ensemble are strongly correlated, so the OLS SE (~10⁻⁴–10⁻³) underestimates
the true fit uncertainty by roughly an order of magnitude. The honest estimate
is the standard error of the mean of H_fit re-extracted on ten disjoint
sub-ensembles, `std(H_sub, ddof=1) / √n_groups` (replica SE), which is ~0.01
for all three aircraft classes. The same convention is used for the σ_θ⋆
replica SE in `estimate_sigma_theta.py`. All scripts report replica SE when
`--n-groups > 1`.

### Float precision

All position arrays are accumulated in `float64`. Phase durations (Lomax,
exponential) and the cycle-counted `|X_N|²` can reach ~10¹² m² for
heavy-tailed sailplane ensembles; `float32` would lose ≈ 8 significant digits
and is not used anywhere in the production pipeline.

---

## Tests

```bash
pytest -q      # 56 tests, ~1 s
```

The suite checks: configuration loading, search-phase invariants
(`T_phys^S = Σ τ_b + Σ τ_turn = τ_S^n`), the `σ_θ = 0` straight-line limit,
the EA-MSD on synthetic ballistic/diffusive trajectories, and the recovery of
analytic Hurst exponents on synthetic power-law MSDs.

---

## References

- J. Vilpellet, A. Darmon, M. Benzaquen, *From Random Walks to Thermal Rides:
  Universal Anomalous Transport in Soaring Flights*,
  [arXiv:2601.01293](https://arxiv.org/abs/2601.01293) (2026).
- R. Metzler, J. Klafter, Phys. Rep. 339, 1 (2000).
- V. Zaburdaev, S. Denisov, J. Klafter, Rev. Mod. Phys. 87, 483 (2015).
- E. A. Codling, M. J. Plank, S. Benhamou, J. R. Soc. Interface 5, 813 (2008).
- O. Bénichou, C. Loverdo, M. Moreau, R. Voituriez, Rev. Mod. Phys. 83, 81 (2011).
