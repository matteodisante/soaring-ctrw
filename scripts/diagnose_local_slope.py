"""Diagnostic: local log-log slope of the simulated MSD.

Goal: test whether the step-based CTRW MSD actually exhibits a clean
power-law regime in the observational window of Vilpellet et al.
(2026), rather than a smooth crossover that a global power-law fit
would average into a single effective exponent.

The local slope is computed as a centred log-derivative:
    d log MSD / d log Delta,
estimated at each lag as the linear regression slope over a small
symmetric log-window.  A true power law shows a flat local slope over
the relevant Delta range; a crossover shows a monotonic variation.

Outputs:
  - outputs/local_slope_diagnostic.pdf (2-panel figure: MSD + local slope)
  - outputs/local_slope_diagnostic.npz (raw arrays)

Assumption (explicit): the MSD is averaged over a finite ensemble of
finite trajectories, so statistical noise grows at large lags where
the time-average has few contributing pairs. We therefore bound the
local-slope evaluation to Delta <= total_time / 2, beyond which the
noise dominates the signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soaring_ctrw import SoaringConfig, simulate_ensemble
from soaring_ctrw.observables import msd_ensemble


def local_slope(
    lags: np.ndarray,
    msd: np.ndarray,
    window_decades: float = 0.3,
) -> np.ndarray:
    """Local log-log slope of msd(lags), computed by a sliding linear fit.

    Parameters
    ----------
    lags, msd : ndarray
        Positive arrays of matched length (lag 0 should be excluded).
    window_decades : float
        Half-width (in log10 units) of the sliding window. Default 0.3
        covers roughly half a decade and is a good compromise between
        locality and statistical stability.

    Returns
    -------
    slope : ndarray
        Same shape as lags; NaN where the window does not contain at
        least 4 points (edges).
    """
    log_lags = np.log(lags)
    log_msd = np.log(msd)
    half = window_decades * np.log(10.0)

    slope = np.full_like(log_lags, np.nan, dtype=float)
    for i, x in enumerate(log_lags):
        mask = (log_lags >= x - half) & (log_lags <= x + half)
        if mask.sum() < 4:
            continue
        xs = log_lags[mask]
        ys = log_msd[mask]
        m, _ = np.polyfit(xs, ys, 1)
        slope[i] = m
    return slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trajectories", type=int, default=400)
    parser.add_argument("--total-time", type=float, default=10_000.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-decades", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aircraft = ["paragliders", "hang_gliders", "sailplanes"]
    colors = {
        "paragliders": "#d95f02",
        "hang_gliders": "#1f78b4",
        "sailplanes": "#6a3d9a",
    }

    rng = np.random.default_rng(args.seed)
    results = {}

    # Bound lags to total_time / 2 to keep time-average statistics reliable
    lag_cutoff = args.total_time / 2

    for name in aircraft:
        cfg = SoaringConfig.from_yaml(f"configs/{name}.yaml")
        print(
            f"Simulating {name}: {args.n_trajectories} trajectories "
            f"of {args.total_time:.0f} s each..."
        )
        ens = simulate_ensemble(
            config=cfg,
            n_trajectories=args.n_trajectories,
            total_time=args.total_time,
            dt=args.dt,
            rng=rng,
        )
        lags = np.arange(ens.shape[1]) * args.dt
        msd = msd_ensemble(ens)

        # drop lag 0 and lags beyond the statistical horizon
        valid = (lags > 0) & (lags <= lag_cutoff)
        lags_v = lags[valid]
        msd_v = msd[valid]
        slope = local_slope(lags_v, msd_v, window_decades=args.window_decades)

        results[name] = dict(lags=lags_v, msd=msd_v, slope=slope)

    # ---- Figure ----
    fig, (ax_msd, ax_slope) = plt.subplots(
        2, 1, figsize=(6.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
        constrained_layout=True,
    )

    # Top: MSD
    for name in aircraft:
        r = results[name]
        ax_msd.loglog(
            r["lags"], r["msd"], ".", ms=3, color=colors[name],
            label=name.replace("_", " "),
        )
    # reference slopes anchored visually
    lag_ref = np.geomspace(3, 3e3, 100)
    ax_msd.loglog(lag_ref, 1e2 * lag_ref**2, "k--", lw=0.8, alpha=0.5,
                  label=r"$\Delta^{2}$ (ballistic)")
    ax_msd.loglog(lag_ref, 3e3 * lag_ref**1, "k:", lw=0.8, alpha=0.5,
                  label=r"$\Delta^{1}$ (diffusive)")
    ax_msd.set_ylabel(r"MSD $\delta^2(\Delta)$ (m$^2$)")
    # Read sigma_theta from the paragliders config (identical across yamls
    # by design of the universality argument).
    sigma_theta_display = SoaringConfig.from_yaml(
        "configs/paragliders.yaml"
    ).angular.sigma_theta
    ax_msd.set_title(
        "Step-based CTRW: time-lag MSD and local scaling exponent\n"
        rf"$\sigma_\theta = {sigma_theta_display:.2f}$ rad, "
        rf"{args.n_trajectories} trajectories, "
        rf"$T_{{\mathrm{{sim}}}} = {args.total_time:.0f}$ s"
    )
    ax_msd.legend(loc="upper left", fontsize=9, ncol=2)
    ax_msd.grid(True, which="both", alpha=0.3)

    # Bottom: local slope = d log MSD / d log Delta = 2H_local
    for name in aircraft:
        r = results[name]
        ax_slope.semilogx(
            r["lags"], r["slope"], "-", lw=1.5, color=colors[name],
            label=name.replace("_", " "),
        )
    ax_slope.axhline(2.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax_slope.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    ax_slope.axhline(1.76, color="r", ls="-.", lw=1.0, alpha=0.7,
                     label=r"$\Delta^{1.76}$ (Vilpellet et al.)")
    ax_slope.set_xlabel(r"$\Delta$ (s)")
    ax_slope.set_ylabel(r"local slope $d\log\delta^2 / d\log\Delta$")
    ax_slope.set_ylim(0.4, 2.1)
    ax_slope.legend(loc="lower left", fontsize=9)
    ax_slope.grid(True, which="both", alpha=0.3)

    # Annotate regimes on the slope panel
    ax_slope.text(
        0.02, 0.92,
        "ballistic: slope = 2\ndiffusive: slope = 1\nanomalous: 1 < slope < 2",
        transform=ax_slope.transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
    )

    out_pdf = args.output_dir / "local_slope_diagnostic.pdf"
    fig.savefig(out_pdf)
    print(f"\nSaved: {out_pdf}")

    # Save raw data
    out_npz = args.output_dir / "local_slope_diagnostic.npz"
    np.savez(
        out_npz,
        **{f"{n}_{k}": results[n][k] for n in aircraft for k in ("lags", "msd", "slope")},
    )
    print(f"Saved: {out_npz}")

    # ---- Numerical summary ----
    print("\n=== Summary: local-slope statistics across Delta decades ===")
    print(f"{'aircraft':<15s} " + "  ".join(f"Δ~{d:5d}s" for d in [10, 30, 100, 300, 1000, 3000]))
    for name in aircraft:
        r = results[name]
        row = [name]
        for d in [10, 30, 100, 300, 1000, 3000]:
            idx = np.argmin(np.abs(r["lags"] - d))
            row.append(f"{r['slope'][idx]:6.3f}")
        print(f"{row[0]:<15s} " + "  ".join(f"{v:>7s}" for v in row[1:]))


if __name__ == "__main__":
    main()
