"""Run a single-aircraft Monte Carlo simulation and plot the MSD.

Usage
-----
    python scripts/run_simulation.py --config configs/paragliders.yaml

Outputs
-------
- MSD curve plotted to ``outputs/<name>_msd.pdf``
- Raw arrays (lags, msd, hurst fit) saved to ``outputs/<name>_msd.npz``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soaring_ctrw import (
    SoaringConfig,
    fit_hurst,
    simulate_ensemble,
)
from soaring_ctrw.observables import msd_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML aircraft configuration.",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=200,
        help="Ensemble size (default: 200).",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=3600.0,
        help="Physical duration per trajectory in seconds (default: 3600).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: ./outputs).",
    )
    args = parser.parse_args()

    config = SoaringConfig.from_yaml(args.config)
    rng = np.random.default_rng(args.seed)

    print(f"Simulating {config.name}:")
    print(f"  trajectories: {args.n_trajectories}")
    print(f"  duration:     {args.total_time:.0f} s")
    print(f"  dt:           {args.dt:.2f} s")
    print(f"  sigma_theta:  {config.angular.sigma_theta:.3f} rad")
    print(f"  N_p (cycles): {config.angular.persistence_cycles:.1f}")

    ensemble = simulate_ensemble(
        config=config,
        n_trajectories=args.n_trajectories,
        total_time=args.total_time,
        dt=args.dt,
        rng=rng,
    )

    # time lags and MSD
    n_steps = ensemble.shape[1]
    lags = np.arange(n_steps) * args.dt
    msd = msd_ensemble(ensemble)

    # fit H in a Vilpellet-style window
    fit = fit_hurst(lags[1:], msd[1:], lag_range=(10.0, 1000.0))
    print(f"\nFitted Hurst exponent (10 s < Δ < 1000 s): H = {fit.hurst:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # save numerical data
    np.savez(
        args.output_dir / f"{config.name}_msd.npz",
        lags=lags,
        msd=msd,
        hurst=fit.hurst,
        slope=fit.slope,
        intercept=fit.intercept,
    )

    # plot
    fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    ax.loglog(lags[1:], msd[1:], "o", ms=3, label=f"{config.name} (MC)")

    # show the fit over its range
    grid = np.geomspace(fit.fit_range[0], fit.fit_range[1], 50)
    ax.loglog(
        grid,
        np.exp(fit.intercept) * grid**fit.slope,
        "k--",
        label=rf"fit: $H = {fit.hurst:.2f}$",
    )
    # reference slopes
    ax.loglog(grid, 1e2 * grid**2, "k:", alpha=0.5, label=r"$\Delta^2$ (ballistic)")
    ax.loglog(
        grid, 1e3 * grid**1.76, "k-.", alpha=0.5, label=r"$\Delta^{1.76}$ (paper)"
    )

    ax.set_xlabel(r"$\Delta$ (s)")
    ax.set_ylabel(r"MSD (m$^2$)")
    ax.set_title(f"Step-based CTRW MSD — {config.name}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plot_path = args.output_dir / f"{config.name}_msd.pdf"
    fig.savefig(plot_path)
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {args.output_dir / (config.name + '_msd.npz')}")


if __name__ == "__main__":
    main()
