"""Combined MSD comparison for the three aircraft.

For each aircraft (paragliders, hang gliders, sailplanes) the script
simulates a Monte Carlo ensemble and computes the pure
ensemble-averaged MSD ``⟨|r(Δ)-r(0)|²⟩`` from the M trajectory
endpoints (one pair per trajectory per lag).
Two figures are produced.

1. ``msd_all_aircraft_raw.pdf`` -- two stacked panels in log-log:

   - Top: the three MSD curves superimposed on the same axes, over the
     lag range ``[dt, --lag-max]`` s (default ``[1, 7000]`` s, matching
     the calibration window of estimate_sigma_theta.py). A power-law
     fit over ``[--fit-min, --lag-max]`` s (default ``[10, 7000]`` s)
     is overlaid for each aircraft and the slope / effective Hurst
     exponent are reported in the legend.
   - Bottom: the local log-log slope
     ``d log MSD / d log Delta`` of each curve, computed by a local
     linear fit over a +/- ``--slope-window`` factor around each lag
     (default 1.5, i.e. about half a decade). Reference horizontal
     lines mark the ballistic (2), the empirical universal
     (1.76 = 2*0.88) and the diffusive (1) slopes.

2. ``msd_all_aircraft_rescaled.pdf`` -- one panel in log-log, showing
   each MSD divided by the aircraft-specific horizontal speed squared
   ``v_xy^2`` (the universal rescaling of VDB Fig. 1 inset). Only the
   simulated data points are plotted: no fit lines are overlaid.

Cache: per-aircraft data is stored under
``outputs/data/msd_all_aircraft/<aircraft>.npz`` with the standard
manifest. Cache modes follow the same convention as the rest of the
repo (see ``cache.py``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from soaring_ctrw.cache import (
    add_cache_args,
    build_manifest,
    decide_action,
    derived_seed,
    load_dataset,
    save_dataset,
    slot_paths,
)
from soaring_ctrw.calibration import calibration_path, load_calibrated_config
from soaring_ctrw.model import SoaringConfig
from soaring_ctrw.observables import (
    fit_hurst,
    msd_ensemble,
    msd_ensemble_percentiles,
)
from soaring_ctrw.paths import CONFIGS_DIR, DATA_DIR, FIGURES_DIR
from soaring_ctrw.simulation import simulate_ensemble

SCRIPT_SLUG = "msd_all_aircraft"
AIRCRAFT_ORDER = ("paragliders", "hang_gliders", "sailplanes")
COLORS = {
    "paragliders":  "tab:orange",
    "hang_gliders": "tab:blue",
    "sailplanes":   "tab:purple",
}
LABELS = {
    "paragliders":  "Paragliders",
    "hang_gliders": "Hang gliders",
    "sailplanes":   "Sailplanes",
}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _compute_msd(
    config: SoaringConfig,
    n_trajectories: int,
    total_time: float,
    dt: float,
    seed: int,
    q_lo: float = 5.0,
    q_hi: float = 95.0,
    fit_min: float = 10.0,
    fit_max: float = 7000.0,
    fit_lag_spacing: str = "linear",
    n_log_lags: int = 40,
    n_groups: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lags, msd, msd_lo, msd_hi, H_sub)``.

    ``msd_lo``/``msd_hi`` are the direct ``q_lo``--``q_hi`` percentiles
    (default 5--95) of the per-flight squared displacements at each lag
    — the same M samples averaged into the EA-MSD. No bootstrap.

    ``H_sub`` are the Hurst exponents refit on ``n_groups`` disjoint
    sub-ensembles over ``[fit_min, fit_max]``: ``std(H_sub)/sqrt(n)``
    estimates the replica-level standard error of the full-ensemble
    ``H`` — the honest uncertainty, an order of magnitude larger than
    the OLS standard error of the (correlated-point) log-log fit.
    """
    rng = np.random.default_rng(
        derived_seed(seed, config.name, SCRIPT_SLUG)
    )
    ens = simulate_ensemble(
        config=config, n_trajectories=n_trajectories,
        total_time=total_time, dt=dt, rng=rng,
    )
    msd, lo, hi = msd_ensemble_percentiles(ens, q_lo=q_lo, q_hi=q_hi)
    lags = np.arange(len(msd)) * dt

    grp = max(2, int(n_groups))
    gsize = n_trajectories // grp
    H_sub = np.full(grp, np.nan)
    if gsize >= 2:
        for j in range(grp):
            sub = ens[j * gsize:(j + 1) * gsize]
            try:
                H_sub[j] = fit_hurst(
                    lags[1:], msd_ensemble(sub)[1:], (fit_min, fit_max),
                    lag_spacing=fit_lag_spacing, n_log_lags=n_log_lags,
                ).hurst
            except ValueError:
                pass
    return lags, msd, lo, hi, H_sub


def _local_slope(
    lags: np.ndarray,
    msd: np.ndarray,
    win_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local log-log slope via per-point window regression.

    For each positive lag ``t`` with ``msd > 0``, fit a line in log
    space using lags within a multiplicative factor ``win_factor`` from
    ``t`` on each side. Returns the (lags, slope) arrays restricted to
    the valid subset.
    """
    L = np.asarray(lags); M = np.asarray(msd)
    mask = (L > 0) & np.isfinite(M) & (M > 0)
    L = L[mask]; M = M[mask]
    if len(L) < 3:
        return L, np.full_like(L, np.nan)
    logL = np.log(L); logM = np.log(M)
    slope = np.full(len(L), np.nan)
    for i, t in enumerate(L):
        window = (L > t / win_factor) & (L < t * win_factor)
        if window.sum() < 3:
            continue
        s, _ = np.polyfit(logL[window], logM[window], 1)
        slope[i] = float(s)
    return L, slope


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_raw_with_slope(
    curves: dict[str, dict[str, np.ndarray]],
    slope_window: float,
    lag_min: float,
    lag_max: float,
    fit_min: float,
    output_path: Path,
    fit_lag_spacing: str = "linear",
    n_log_lags: int = 40,
) -> None:
    """Figure 1: MSD log-log (top, with per-aircraft power-law fits
    overlaid over ``[fit_min, lag_max]``) + local log-log slope
    (bottom, one curve per aircraft).

    The fit window matches the empirical VDB window
    ``[fit_min, lag_max] = [10, 7000]`` s by default. The fitted slope
    and the corresponding effective Hurst exponent ``H = slope/2`` (with
    the standard error of the log-log regression) are reported in the
    legend; the shaded band spans the 5th-95th percentile of the
    per-flight squared displacements at each lag (sample spread, no
    bootstrap).
    """
    fig, (ax_msd, ax_slope) = plt.subplots(
        2, 1, figsize=(8.0, 8.5), constrained_layout=True, sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )

    for aircraft in AIRCRAFT_ORDER:
        if aircraft not in curves:
            continue
        lags = curves[aircraft]["lags"]
        msd = curves[aircraft]["msd"]
        lo = curves[aircraft].get("lo")
        hi = curves[aircraft].get("hi")
        m = (lags >= lag_min) & (lags <= lag_max) & np.isfinite(msd) & (msd > 0)
        # --- 5-95% sample-percentile band (if available) -------------
        if lo is not None and hi is not None:
            mb = m & np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
            ax_msd.fill_between(
                lags[mb], lo[mb], hi[mb],
                color=COLORS[aircraft], alpha=0.20, lw=0,
            )
        # --- MSD curve ------------------------------------------------
        ax_msd.loglog(
            lags[m], msd[m],
            "-", color=COLORS[aircraft], lw=2.0,
            label=LABELS[aircraft],
        )
        # --- Power-law fit over [fit_min, lag_max] --------------------
        # The quoted uncertainty is the replica-level standard error
        # (std over disjoint sub-ensembles / sqrt(n)); the OLS standard
        # error of the correlated-point log-log fit underestimates it
        # by roughly an order of magnitude and is not displayed.
        try:
            fit = fit_hurst(
                lags[1:], msd[1:], (fit_min, lag_max),
                lag_spacing=fit_lag_spacing, n_log_lags=n_log_lags,
            )
        except ValueError:
            fit = None
        if fit is not None:
            grid = np.geomspace(fit_min, lag_max, 80)
            H_sub = curves[aircraft].get("H_sub")
            if H_sub is not None:
                H_sub = np.asarray(H_sub, dtype=float)
                H_sub = H_sub[np.isfinite(H_sub)]
            if H_sub is not None and H_sub.size >= 2:
                h_err = float(H_sub.std(ddof=1) / np.sqrt(H_sub.size))
                err_tag = "replica s.e."
            else:
                h_err = fit.hurst_err
                err_tag = "OLS s.e. (underest.)"
            ax_msd.loglog(
                grid, np.exp(fit.intercept) * grid ** fit.slope,
                "--", color=COLORS[aircraft], lw=1.4, alpha=0.95,
                label=(
                    rf"  fit {LABELS[aircraft]}: slope $={fit.slope:.3f}$  "
                    rf"($H={fit.hurst:.3f}\pm{h_err:.2g}$, {err_tag})"
                ),
            )
        # --- Local log-log slope -------------------------------------
        L_s, s = _local_slope(lags[m], msd[m], slope_window)
        ax_slope.semilogx(
            L_s, s, "-", color=COLORS[aircraft], lw=1.8,
            label=LABELS[aircraft],
        )

    ax_msd.set_ylabel(r"$\langle\delta^2(\Delta)\rangle$  (m$^2$)")
    ax_msd.grid(True, which="both", ls=":", alpha=0.4)
    ax_msd.legend(loc="lower right", fontsize=9)
    ax_msd.set_title(
        rf"MSD vs lag (log-log) — power-law fit over "
        rf"$\Delta\in[{fit_min:g},{lag_max:g}]$ s",
        fontsize=11,
    )

    ax_slope.axhline(2.0, color="k", ls="--", lw=0.9, alpha=0.6,
                     label="slope 2 (ballistic, $H=1$)")
    ax_slope.axhline(1.76, color="gray", ls="-.", lw=0.9, alpha=0.7,
                     label=r"slope 1.76 ($H=0.88$, VDB)")
    ax_slope.axhline(1.0, color="k", ls=":", lw=0.9, alpha=0.6,
                     label="slope 1 (diffusive, $H=1/2$)")
    ax_slope.set_xlabel(r"$\Delta$  (s)")
    ax_slope.set_ylabel(r"$d\log\,\delta^2 / d\log\,\Delta$")
    ax_slope.grid(True, which="both", ls=":", alpha=0.4)
    ax_slope.set_ylim(0.5, 2.2)
    ax_slope.set_title(
        rf"Local log-log slope (smoothing window: "
        rf"$\Delta\in[\Delta/{slope_window:g}, \Delta\cdot{slope_window:g}]$)",
        fontsize=10,
    )
    ax_slope.legend(loc="lower left", fontsize=8)
    ax_slope.set_xlim(lag_min, lag_max)

    fig.savefig(output_path)
    plt.close(fig)


def _plot_rescaled(
    curves: dict[str, dict[str, np.ndarray]],
    configs: dict[str, SoaringConfig],
    lag_min: float,
    lag_max: float,
    output_path: Path,
) -> None:
    """Figure 2: rescaled MSD (data points only, no fit overlay)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)

    for aircraft in AIRCRAFT_ORDER:
        if aircraft not in curves:
            continue
        v_xy = float(configs[aircraft].v_xy)
        lags = curves[aircraft]["lags"]
        msd = curves[aircraft]["msd"]
        lo = curves[aircraft].get("lo")
        hi = curves[aircraft].get("hi")
        msd_resc = msd / (v_xy ** 2)
        m = (lags >= lag_min) & (lags <= lag_max) & np.isfinite(msd_resc) & (msd_resc > 0)

        if lo is not None and hi is not None:
            mb = m & np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
            ax.fill_between(
                lags[mb], lo[mb] / v_xy ** 2, hi[mb] / v_xy ** 2,
                color=COLORS[aircraft], alpha=0.18, lw=0,
            )
        ax.loglog(
            lags[m], msd_resc[m],
            "o", color=COLORS[aircraft], ms=3.0, alpha=0.7, mew=0,
            label=rf"{LABELS[aircraft]}  ($v_{{xy}}={v_xy:.2f}$ m/s)",
        )

    ax.set_xlabel(r"$\Delta$  (s)")
    ax.set_ylabel(r"$\langle\delta^2(\Delta)\rangle\,/\,v_{xy}^2$  (s$^2$)")
    ax.set_title("Rescaled MSD — universal collapse", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(lag_min, lag_max)

    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft", nargs="+", default=list(AIRCRAFT_ORDER),
        choices=list(AIRCRAFT_ORDER),
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=2000,
        help="Ensemble size M. Heavy-tailed transition durations "
             "(mu_T < 4 for paragliders and sailplanes) make the MSD "
             "amplitude converge slowly; prefer large M (see Appendix D "
             "of the manuscript and plot_variance_convergence.py).",
    )
    parser.add_argument(
        "--total-time", type=float, default=15_000.0,
        help="Trajectory length (s). Must be >= lag_max so each "
             "trajectory contributes a sample at the largest lag.",
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--lag-min", type=float, default=1.0,
        help="Lower edge of the lag axis (s). Cannot be 0 in log-log.",
    )
    parser.add_argument(
        "--lag-max", type=float, default=7_000.0,
        help="Upper edge of the lag axis (s) — matches the calibration "
             "window used by estimate_sigma_theta.py.",
    )
    parser.add_argument(
        "--slope-window", type=float, default=1.5,
        help="Smoothing factor for the local log-log slope panel: at "
             "each lag t, regress over lags in [t/win, t*win].",
    )
    parser.add_argument(
        "--fit-min", type=float, default=10.0,
        help="Lower edge of the power-law fit on the rescaled MSD (s). "
             "Upper edge = --lag-max.",
    )
    parser.add_argument(
        "--q-lo", type=float, default=5.0,
        help="Lower percentile of the sample band on the MSD (default 5).",
    )
    parser.add_argument(
        "--q-hi", type=float, default=95.0,
        help="Upper percentile of the sample band on the MSD (default 95).",
    )
    parser.add_argument(
        "--fit-lag-spacing", choices=("linear", "log"), default="linear",
        help="Lag-spacing convention of the log-log H fit: 'linear' "
             "uses every lag of the uniform grid (manuscript "
             "convention); 'log' subsamples --n-log-lags lags uniformly "
             "in log(lag). Both fits are printed for comparison.",
    )
    parser.add_argument(
        "--n-log-lags", type=int, default=40,
        help="Number of log-spaced lags when --fit-lag-spacing=log.",
    )
    parser.add_argument(
        "--n-groups", type=int, default=10,
        help="Disjoint sub-ensembles for the replica-level standard "
             "error of the fitted H (quoted in the legend).",
    )
    parser.add_argument(
        "--write-hfit",
        action="store_true",
        default=False,
        help="After computing the MSD, write the fitted H and its replica "
             "s.e. to outputs/data/calibration/<aircraft>.yaml (section "
             "'h_fit'). These values are then read by "
             "scripts/write_paper_macros.py to generate LaTeX \\newcommand "
             "definitions — so the paper never contains a hardcoded H_fit "
             "uncertainty chosen by hand.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    add_cache_args(parser)
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    curves: dict[str, dict[str, np.ndarray]] = {}
    configs: dict[str, SoaringConfig] = {}

    for aircraft in args.aircraft:
        print(f"=== {aircraft} ===")
        config_path = CONFIGS_DIR / f"{aircraft}.yaml"
        # Loads configs/<aircraft>.yaml and overrides sigma_theta with
        # the value calibrated by estimate_sigma_theta.py (stored in
        # outputs/data/calibration/<aircraft>.yaml). Raises if the
        # calibration is missing — required for the H_eff ≈ 0.88 check.
        config = load_calibrated_config(aircraft)
        configs[aircraft] = config
        print(f"  sigma_theta (calibrated) = {config.angular.sigma_theta:.6f}")

        npz_path, manifest_path = slot_paths(args.data_dir, SCRIPT_SLUG, aircraft)
        requested = build_manifest(
            script=SCRIPT_SLUG,
            params={
                "n_trajectories": args.n_trajectories,
                "total_time": args.total_time,
                "dt": args.dt,
                "seed": args.seed,
                "band": f"sample-percentile-{args.q_lo:g}-{args.q_hi:g}",
                "msd_estimator": "ea",
                "fit_window": [args.fit_min, args.lag_max],
                "fit_lag_spacing": args.fit_lag_spacing,
                "n_log_lags": args.n_log_lags,
                "n_groups": args.n_groups,
                "h_sub": True,
            },
            # Hash both the input YAML and the calibration YAML so the
            # cache is invalidated whenever either changes (in particular
            # when sigma_theta is re-calibrated).
            config_paths={
                "aircraft": config_path,
                "calibration": calibration_path(aircraft),
            },
        )
        decision = decide_action(
            npz_path=npz_path,
            manifest_path=manifest_path,
            requested_manifest=requested,
            mode=args.cache,
            slot_label=f"{SCRIPT_SLUG}/{aircraft}",
        )
        if decision.diff:
            print("  cache params differ:")
            for line in decision.diff:
                print(f"    - {line}")
        print(f"  cache decision: {decision.action}  ({decision.reason})")

        if decision.action == "reuse":
            arrays, _ = load_dataset(npz_path, manifest_path)
            lags = arrays["lags"]
            msd = arrays["msd"]
            # Older caches predate the percentile band; fall back to None.
            msd_lo = arrays["msd_lo"] if "msd_lo" in arrays else None
            msd_hi = arrays["msd_hi"] if "msd_hi" in arrays else None
            H_sub = arrays["H_sub"] if "H_sub" in arrays else None
        else:
            print(
                f"  simulating {args.n_trajectories} traj "
                f"× {args.total_time:.0f} s ..."
            )
            t0 = time.time()
            lags, msd, msd_lo, msd_hi, H_sub = _compute_msd(
                config, args.n_trajectories, args.total_time, args.dt,
                args.seed, q_lo=args.q_lo, q_hi=args.q_hi,
                fit_min=args.fit_min, fit_max=args.lag_max,
                fit_lag_spacing=args.fit_lag_spacing,
                n_log_lags=args.n_log_lags,
                n_groups=args.n_groups,
            )
            print(f"  done in {time.time() - t0:.1f} s")
            if args.cache != "off":
                save_dataset(
                    npz_path=npz_path,
                    manifest_path=manifest_path,
                    manifest=requested,
                    arrays={
                        "lags": lags, "msd": msd,
                        "msd_lo": msd_lo, "msd_hi": msd_hi,
                        "H_sub": H_sub,
                    },
                )
                print(f"  saved data: {npz_path}")

        curves[aircraft] = {
            "lags": lags, "msd": msd, "lo": msd_lo, "hi": msd_hi,
            "H_sub": H_sub,
        }

        # Convention-sensitivity printout: same cached MSD, two fits.
        # The difference is a pure lag-weighting effect (declared
        # systematic; see Sec. V.A of the manuscript).
        try:
            h_lin = fit_hurst(
                lags[1:], msd[1:], (args.fit_min, args.lag_max),
                lag_spacing="linear",
            ).hurst
            h_log = fit_hurst(
                lags[1:], msd[1:], (args.fit_min, args.lag_max),
                lag_spacing="log", n_log_lags=args.n_log_lags,
            ).hurst
            print(
                f"  H_fit: linear-lags = {h_lin:.4f} | "
                f"log-lags = {h_log:.4f} | diff = {h_lin - h_log:+.4f}"
            )
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Optionally persist H_fit + replica SE to the per-aircraft
    # calibration YAMLs so that write_paper_macros.py can generate
    # paper_macros.tex without any hardcoded values.
    # ------------------------------------------------------------------
    if args.write_hfit:
        from soaring_ctrw.calibration import write_calibration_section
        print("\n=== Writing H_fit to calibration YAMLs (--write-hfit) ===")
        for aircraft in args.aircraft:
            if aircraft not in curves:
                continue
            lags_ac = curves[aircraft]["lags"]
            msd_ac  = curves[aircraft]["msd"]
            H_sub_ac = curves[aircraft].get("H_sub")
            try:
                fit_ac = fit_hurst(
                    lags_ac[1:], msd_ac[1:], (args.fit_min, args.lag_max),
                    lag_spacing=args.fit_lag_spacing,
                    n_log_lags=args.n_log_lags,
                )
                h_val = float(fit_ac.hurst)
            except ValueError:
                h_val = float("nan")
            # Replica SE from sub-ensembles (honest uncertainty)
            if H_sub_ac is not None:
                H_sub_arr = np.asarray(H_sub_ac, dtype=float)
                H_sub_arr = H_sub_arr[np.isfinite(H_sub_arr)]
                if H_sub_arr.size >= 2:
                    replica_se = float(H_sub_arr.std(ddof=1) / np.sqrt(H_sub_arr.size))
                    n_groups_eff = int(H_sub_arr.size)
                else:
                    replica_se = float("nan")
                    n_groups_eff = 0
            else:
                replica_se = float("nan")
                n_groups_eff = 0
            payload = {
                "source_script": "plot_msd_all_aircraft",
                "h_fit": h_val,
                "h_fit_replica_se": replica_se,
                "n_groups": n_groups_eff,
                "fit_lag_spacing": args.fit_lag_spacing,
                "fit_window": [float(args.fit_min), float(args.lag_max)],
            }
            cal_path = write_calibration_section(aircraft, "h_fit", payload)
            print(
                f"  [{aircraft}]  H_fit = {h_val:.4f} ± {replica_se:.4f} "
                f"(replica s.e., n_groups={n_groups_eff})  →  {cal_path}"
            )

    raw_path = args.figures_dir / f"{SCRIPT_SLUG}_raw.pdf"
    _plot_raw_with_slope(
        curves, slope_window=args.slope_window,
        lag_min=args.lag_min, lag_max=args.lag_max,
        fit_min=args.fit_min,
        output_path=raw_path,
        fit_lag_spacing=args.fit_lag_spacing,
        n_log_lags=args.n_log_lags,
    )
    print(f"\nSaved raw figure: {raw_path}")

    rescaled_path = args.figures_dir / f"{SCRIPT_SLUG}_rescaled.pdf"
    _plot_rescaled(
        curves, configs,
        lag_min=args.lag_min, lag_max=args.lag_max,
        output_path=rescaled_path,
    )
    print(f"Saved rescaled figure: {rescaled_path}")


if __name__ == "__main__":
    main()
