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
from soaring_ctrw.observables import msd_ensemble_ci
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
    n_boot: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lags, msd, msd_lo, msd_hi)`` with a bootstrap 95% CI band."""
    rng = np.random.default_rng(
        derived_seed(seed, config.name, SCRIPT_SLUG)
    )
    ens = simulate_ensemble(
        config=config, n_trajectories=n_trajectories,
        total_time=total_time, dt=dt, rng=rng,
    )
    boot_rng = np.random.default_rng(
        derived_seed(seed, config.name, SCRIPT_SLUG + "_boot")
    )
    msd, lo, hi = msd_ensemble_ci(ens, n_boot=n_boot, ci=0.95, rng=boot_rng)
    lags = np.arange(len(msd)) * dt
    return lags, msd, lo, hi


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


def _fit_powerlaw(
    lags: np.ndarray,
    msd: np.ndarray,
    fit_min: float,
    fit_max: float,
) -> tuple[float, float, float] | None:
    """Linear fit of log MSD vs log lag over ``[fit_min, fit_max]``.

    Returns ``(slope, intercept, slope_err)`` or ``None`` if fewer than
    2 points are in range. ``slope_err`` is the OLS standard error of
    the slope (``NaN`` for fewer than 3 points).
    """
    mask = (
        (lags >= fit_min) & (lags <= fit_max)
        & (msd > 0) & np.isfinite(msd)
    )
    if mask.sum() < 2:
        return None
    x = np.log(lags[mask])
    y = np.log(msd[mask])
    slope, intercept = np.polyfit(x, y, 1)
    n = x.size
    if n > 2:
        resid = y - (slope * x + intercept)
        sxx = float(np.sum((x - x.mean()) ** 2))
        slope_err = (
            float(np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx))
            if sxx > 0.0 else float("nan")
        )
    else:
        slope_err = float("nan")
    return float(slope), float(intercept), slope_err


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
) -> None:
    """Figure 1: MSD log-log (top, with per-aircraft power-law fits
    overlaid over ``[fit_min, lag_max]``) + local log-log slope
    (bottom, one curve per aircraft).

    The fit window matches the empirical VDB window
    ``[fit_min, lag_max] = [10, 7000]`` s by default. The fitted slope
    and the corresponding effective Hurst exponent ``H = slope/2`` (with
    the standard error of the log-log regression) are reported in the
    legend; a shaded band shows the bootstrap 95% CI of the MSD.
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
        # --- Bootstrap 95% CI band (if available) --------------------
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
        fit = _fit_powerlaw(lags, msd, fit_min, lag_max)
        if fit is not None:
            slope, intercept, slope_err = fit
            grid = np.geomspace(fit_min, lag_max, 80)
            h_err = slope_err / 2.0
            ax_msd.loglog(
                grid, np.exp(intercept) * grid ** slope,
                "--", color=COLORS[aircraft], lw=1.4, alpha=0.95,
                label=(
                    rf"  fit {LABELS[aircraft]}: slope $={slope:.3f}$  "
                    rf"($H={slope/2:.3f}\pm{h_err:.3f}$)"
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
    parser.add_argument("--n-trajectories", type=int, default=500)
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
        "--n-boot", type=int, default=1000,
        help="Bootstrap resamples for the 95%% CI band on the MSD.",
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
                "n_boot": args.n_boot,
                "msd_estimator": "ea",
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
            # Older caches predate the bootstrap CI; fall back to None.
            msd_lo = arrays["msd_lo"] if "msd_lo" in arrays else None
            msd_hi = arrays["msd_hi"] if "msd_hi" in arrays else None
        else:
            print(
                f"  simulating {args.n_trajectories} traj "
                f"× {args.total_time:.0f} s ..."
            )
            t0 = time.time()
            lags, msd, msd_lo, msd_hi = _compute_msd(
                config, args.n_trajectories, args.total_time, args.dt,
                args.seed, n_boot=args.n_boot,
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
                    },
                )
                print(f"  saved data: {npz_path}")

        curves[aircraft] = {
            "lags": lags, "msd": msd, "lo": msd_lo, "hi": msd_hi,
        }

    raw_path = args.figures_dir / f"{SCRIPT_SLUG}_raw.pdf"
    _plot_raw_with_slope(
        curves, slope_window=args.slope_window,
        lag_min=args.lag_min, lag_max=args.lag_max,
        fit_min=args.fit_min,
        output_path=raw_path,
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
