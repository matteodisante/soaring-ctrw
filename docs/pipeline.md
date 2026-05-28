# Reproduction pipeline

All scripts can be run independently, but several consume artefacts produced by
others. The end-to-end order that reproduces every figure of the manuscript is
below. The per-script details (options, import map) are on each script's page
under {doc}`scripts`.

Every script writes under `outputs/` (`figures/` for PDF/PNG, `data/` for NPZ),
overridable with `--figures-dir` / `--data-dir`.

1. **Search-phase calibration** — $m_{1/2}(\alpha_S)$ and the calibrated
   $\tau_{\mathrm{turn}}^S$; with `--write`, merged into the `mittag_leffler`
   section of `outputs/data/calibration/<aircraft>.yaml`:
   ```bash
   python scripts/compute_ml_median_and_tau_turn.py --write
   ```

2. **Per-phase MSD** — the 3-panel EA-MSD figure and the per-aircraft MSD `.npz`
   caches used in step 5:
   ```bash
   python scripts/plot_per_phase_msd.py
   ```

3. **$\sigma_\theta$ calibration** — 1-D scan of $H_{\mathrm{eff}}(\sigma_\theta)$
   against the empirical $0.88$; with `--write`, the full-cycle
   $\sigma_\theta^\star$ is merged into the `sigma_theta` section:
   ```bash
   python scripts/estimate_sigma_theta.py \
       --n-sigma 16 --n-trajectories 1000 --total-time 15000 \
       --fit-min 10 --fit-max 7000 --write
   ```

4. **Pre-asymptotic crossover time** — reads the calibrated $\sigma_\theta$ from
   step 3 and the mean cycle duration from the config, writes the
   `critical_time` section:
   ```bash
   python scripts/compute_critical_time.py --write
   ```

5. **Final figures** — simulation MSDs vs theory, MSD overlay, trajectories,
   phase-duration CCDFs:
   ```bash
   python scripts/plot_msd_all_aircraft.py
   python scripts/compare_msd_to_theory.py
   python scripts/plot_trajectory.py
   python scripts/plot_phase_durations.py
   ```

The input `configs/<aircraft>.yaml` files keep `angular.sigma_theta: null`; any
script that needs $\sigma_\theta$ reads it from the calibration YAML through
{func}`~soaring_ctrw.calibration.apply_calibration`. All derived per-aircraft
quantities live in that single YAML, merged section-by-section by
{func}`~soaring_ctrw.calibration.write_calibration_section`, so successive
calibrators never overwrite each other.

```{note}
This ordering is one of the few pieces of documentation that is **hand-written
prose** (the dependency *graph* in {doc}`architecture` is auto-derived, but the
intended *run sequence* is an editorial choice). If you add or reorder a
pipeline step, update this page. See {doc}`contributing-docs`.
```
