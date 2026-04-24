"""Final comparison: simulated MSD overlaid with reference slope lines
matching Fig. 1 of Vilpellet et al. (2026).

Produces a PRL-quality two-panel figure:
  (top)    MSD(Delta) for the three aircraft, with Delta^2 and Delta^1.75
           reference slopes from the empirical paper.
  (bottom) local slope dlogMSD/dlogDelta vs Delta, with the target
           slope 1.75 marked.

Intended as Fig. 2 of our paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soaring_ctrw import SoaringConfig, simulate_ensemble
from soaring_ctrw.observables import msd_ensemble, fit_hurst


def local_slope(lags: np.ndarray, msd: np.ndarray, window_decades: float = 0.3) -> np.ndarray:
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
    p = argparse.ArgumentParser()
    p.add_argument("--n-trajectories", type=int, default=300)
    p.add_argument("--total-time", type=float, default=10_000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aircraft = ["paragliders", "hang_gliders", "sailplanes"]
    colors = {
        "paragliders": "#d95f02",
        "hang_gliders": "#1f78b4",
        "sailplanes": "#6a3d9a",
    }
    markers = {"paragliders": "o", "hang_gliders": "*", "sailplanes": "D"}

    rng = np.random.default_rng(args.seed)
    results = {}
    for name in aircraft:
        cfg = SoaringConfig.from_yaml(f"configs/{name}.yaml")
        print(f"Simulating {name}...")
        ens = simulate_ensemble(
            config=cfg, n_trajectories=args.n_trajectories,
            total_time=args.total_time, dt=1.0, rng=rng,
        )
        lags = np.arange(ens.shape[1]) * 1.0
        msd = msd_ensemble(ens)
        valid = (lags > 0) & (lags <= args.total_time / 2)
        lags_v, msd_v = lags[valid], msd[valid]
        slope = local_slope(lags_v, msd_v)
        fit = fit_hurst(lags_v, msd_v, lag_range=(10.0, 5000.0))
        results[name] = dict(lags=lags_v, msd=msd_v, slope=slope, fit=fit)
        print(f"  H_fit (10-5000 s) = {fit.hurst:.3f}")

    fig, (ax_msd, ax_slope) = plt.subplots(
        2, 1, figsize=(6.5, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0]},
        constrained_layout=True,
    )

    # Top: MSD
    for name in aircraft:
        r = results[name]
        ax_msd.loglog(
            r["lags"], r["msd"],
            marker=markers[name], ms=3, lw=0.5, color=colors[name],
            label=f"{name.replace('_', ' ')} ($H$ = {r['fit'].hurst:.2f})",
        )
    lag_ref = np.geomspace(1.0, 1e4, 100)
    ax_msd.loglog(lag_ref, 1e2 * lag_ref**2, "k--", lw=0.8, alpha=0.4, label=r"$\Delta^{2}$ (ballistic)")
    ax_msd.loglog(lag_ref, 3e2 * lag_ref**1.75, "k:", lw=0.8, alpha=0.4, label=r"$\Delta^{1.75}$ (Vilpellet)")
    ax_msd.set_ylabel(r"MSD $\delta^2(\Delta)$ (m$^2$)")
    ax_msd.set_title(
        "Step-based CTRW with velocity-limited intra-phase legs\n"
        rf"$N_{{\mathrm{{traj}}}}={args.n_trajectories}$, $T_{{\mathrm{{sim}}}}={int(args.total_time)}$ s"
    )
    ax_msd.legend(loc="upper left", fontsize=8, ncol=2)
    ax_msd.grid(True, which="both", alpha=0.3)
    ax_msd.set_ylim(1e1, 1e10)

    # Bottom: local slope
    for name in aircraft:
        r = results[name]
        ax_slope.semilogx(
            r["lags"], r["slope"], "-", lw=1.5, color=colors[name],
            label=name.replace("_", " "),
        )
    ax_slope.axhline(2.0, color="k", ls="--", lw=0.7, alpha=0.4)
    ax_slope.axhline(1.0, color="k", ls=":", lw=0.7, alpha=0.4)
    ax_slope.axhline(1.75, color="r", ls="-.", lw=1.0, alpha=0.6, label=r"$\Delta^{1.75}$")
    ax_slope.set_xlabel(r"$\Delta$ (s)")
    ax_slope.set_ylabel(r"local slope $d\log\delta^2/d\log\Delta$")
    ax_slope.set_ylim(0.4, 2.15)
    ax_slope.grid(True, which="both", alpha=0.3)
    ax_slope.legend(loc="lower right", fontsize=8)

    ax_slope.text(
        0.02, 0.95,
        "ballistic: slope = 2\nsub-ballistic: 1 < slope < 2",
        transform=ax_slope.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )

    out_pdf = args.output_dir / "msd_final_calibrated.pdf"
    fig.savefig(out_pdf)
    print(f"\nSaved: {out_pdf}")


if __name__ == "__main__":
    main()
