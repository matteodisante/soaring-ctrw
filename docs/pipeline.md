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
   $\sigma_\theta^\star$ (with its sub-ensemble replica standard error) is
   merged into the `sigma_theta` section:
   ```bash
   python scripts/estimate_sigma_theta.py \
       --n-sigma 16 --n-trajectories 2000 --total-time 15000 \
       --fit-min 10 --fit-max 7000 --fit-lag-spacing linear \
       --n-groups 10 --write
   ```

4. **Pre-asymptotic crossover time** — reads the calibrated $\sigma_\theta$ from
   step 3 and the mean cycle duration from the config, writes the
   `critical_time` section:
   ```bash
   python scripts/compute_critical_time.py --write
   ```

5. **Total MSD and rescaled collapse** — the MSD overlay with local-slope
   panel and the rescaled collapse. With `--write-hfit`, the fitted $H$ and
   its replica standard error are merged into the `h_fit` section (required
   by step 5a):
   ```bash
   python scripts/plot_msd_all_aircraft.py \
       --n-trajectories 2000 --n-groups 10 --write-hfit
   ```

6. **LaTeX paper macros** — reads $\sigma_\theta^\star$ (step 3) and $H$
   (step 5) from the calibration YAMLs and writes a `\newcommand` file the
   manuscript includes via `\InputIfFileExists`:
   ```bash
   python scripts/write_paper_macros.py --out <paper-dir>/paper_macros.tex
   ```

7. **Comparison with closed-form theory** — simulation MSDs (search, climb,
   cycle-counted) and $H_{\mathrm{eff}}(N)$ against the analytical
   expressions in {mod}`soaring_ctrw.theory`:
   ```bash
   python scripts/compare_msd_to_theory.py \
       --n-traj-cycles 10000 \
       --n-traj-cycles-by-aircraft sailplanes=120000 --n-cycles 60
   ```

8. **Appendix figures and diagnostics** — variance convergence, a sample
   trajectory, and the phase-duration CCDFs:
   ```bash
   python scripts/plot_variance_convergence.py
   python scripts/plot_trajectory.py
   python scripts/plot_phase_durations.py
   ```

9. **(optional) Parameter sensitivity** — $\pm20\%$ perturbations of the
   eye-read parameters and their effect on $H_{\mathrm{fit}}$, printed and
   saved to `outputs/data/sensitivity_scan.csv`:
   ```bash
   python scripts/scan_parameter_sensitivity.py \
       --n-cycles 300 --n-traj 500 --perturb 0.20
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
