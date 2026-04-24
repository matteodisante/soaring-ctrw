"""Diagnostic: compare model local slope against the empirical scaling.

Vilpellet et al. (2026) Fig. 1 reports a clean Δ^{1.76} scaling from
Δ ~ 3 s to Δ ~ 5×10³ s, spanning ~3 decades. This corresponds to a
flat local slope ≈ 1.76 over that range.

This script overlays, for each aircraft class, the local
log-log slope of the simulated MSD against the empirical target. It
serves as a quick visual check that the calibrated model
(σ_θ = 0.35 rad, refined intra-phase dynamics) tracks the empirical
scaling across the observation window.

Earlier versions of the model with horizontally-stationary search
and climb produced a marked dip at Δ ~ 50–100 s (slope falling to
~ 1.5) and a hump at Δ ~ 10³ s (slope rising to ~ 1.85); both have
been removed by adding the intra-phase contributions
(``SearchMotionConfig``, ``ClimbMotionConfig``).
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
    parser.add_argument("--seed", type=int, default=42)
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
    lag_cutoff = args.total_time / 2

    for name in aircraft:
        cfg = SoaringConfig.from_yaml(f"configs/{name}.yaml")
        ens = simulate_ensemble(
            config=cfg,
            n_trajectories=args.n_trajectories,
            total_time=args.total_time,
            dt=1.0,
            rng=rng,
        )
        lags = np.arange(ens.shape[1]) * 1.0
        msd = msd_ensemble(ens)
        valid = (lags > 0) & (lags <= lag_cutoff)
        lags_v = lags[valid]
        msd_v = msd[valid]
        slope = local_slope(lags_v, msd_v, window_decades=0.3)
        results[name] = dict(lags=lags_v, msd=msd_v, slope=slope)

    # ---- Figure: model vs empirical ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

    for name in aircraft:
        r = results[name]
        ax.semilogx(
            r["lags"], r["slope"], "-", lw=1.8, color=colors[name],
            label=f"model, {name.replace('_', ' ')}",
        )

    # Empirical: Vilpellet et al. report a clean Delta^1.76 scaling.
    ax.axhline(
        1.76, color="red", ls="--", lw=1.5, alpha=0.8,
        label=r"empirical [Vilpellet et al. Fig.~1]: slope $\simeq 1.76$",
    )

    # Reference regimes
    ax.axhline(2.0, color="k", ls=":", lw=0.7, alpha=0.5)
    ax.axhline(1.0, color="k", ls=":", lw=0.7, alpha=0.5)
    ax.text(1.2, 2.02, "ballistic", fontsize=8, color="gray")
    ax.text(1.2, 1.02, "diffusive", fontsize=8, color="gray")

    ax.set_xlabel(r"$\Delta$ (s)")
    ax.set_ylabel(r"local scaling slope  $d\log\delta^2 / d\log\Delta$")
    ax.set_ylim(0.4, 2.15)
    ax.set_title(
        "Calibrated CTRW model vs empirical scaling: local-slope diagnostic"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    out_pdf = args.output_dir / "model_vs_empirical_slope.pdf"
    fig.savefig(out_pdf)
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
