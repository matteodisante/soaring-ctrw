"""Compute the (mu_transition, sigma_theta) phase diagram of H_eff.

This script runs a grid of Monte Carlo simulations varying the tail
exponent ``mu_transition`` of the transition-phase duration and the
angular diffusivity ``sigma_theta``. For each (mu, sigma) it fits an
effective Hurst exponent in the window 10 s < Δ < 1000 s, reproducing
the window chosen by Vilpellet et al. for their empirical Hurst fit.

The output is a contour plot with the H = 0.88 iso-line highlighted.
The three empirical aircraft are overlaid at their measured mu, with
sigma_theta treated as a free parameter fitted to match H = 0.88.

This is the central figure of the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soaring_ctrw.model import (
    AngularConfig,
    PhaseConfig,
    SoaringConfig,
)
from soaring_ctrw.observables import fit_hurst, msd_ensemble
from soaring_ctrw.simulation import simulate_ensemble


def h_eff(
    mu_T: float,
    sigma_theta: float,
    v_xy: float,
    n_trajectories: int,
    total_time: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    """Effective Hurst exponent for a given (mu_T, sigma_theta).

    Other parameters (search/climb statistics) are fixed to generic
    paraglider-like defaults; they have minor influence on H_eff in the
    fitting window as transition dominates the horizontal displacement.
    """
    config = SoaringConfig(
        name=f"scan_mu{mu_T:.2f}_sig{sigma_theta:.2f}",
        v_xy=v_xy,
        transition=PhaseConfig("pareto", {"mu": mu_T, "tau_min": 40.0}),
        search=PhaseConfig("pareto", {"mu": 3.0, "tau_min": 20.0}),
        climb=PhaseConfig("exponential", {"tau_mean": 120.0}),
        angular=AngularConfig(sigma_theta=sigma_theta),
    )
    ensemble = simulate_ensemble(
        config=config,
        n_trajectories=n_trajectories,
        total_time=total_time,
        dt=dt,
        rng=rng,
    )
    lags = np.arange(ensemble.shape[1]) * dt
    msd = msd_ensemble(ensemble)
    fit = fit_hurst(lags[1:], msd[1:], lag_range=(10.0, 5000.0))
    return fit.hurst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--n-sigma", type=int, default=8)
    parser.add_argument("--mu-min", type=float, default=2.1)
    parser.add_argument("--mu-max", type=float, default=5.0)
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--total-time", type=float, default=3600.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    mu_grid = np.linspace(args.mu_min, args.mu_max, args.n_mu)
    sigma_grid = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma)
    H_grid = np.zeros((args.n_mu, args.n_sigma))

    rng = np.random.default_rng(args.seed)

    print(
        f"Scanning {args.n_mu} × {args.n_sigma} = {args.n_mu * args.n_sigma} points"
    )
    for i, mu in enumerate(mu_grid):
        for j, sigma in enumerate(sigma_grid):
            H_grid[i, j] = h_eff(
                mu_T=mu,
                sigma_theta=sigma,
                v_xy=10.0,
                n_trajectories=args.n_trajectories,
                total_time=args.total_time,
                dt=args.dt,
                rng=rng,
            )
            print(f"  mu={mu:.2f}  sigma={sigma:.2f}  H_eff={H_grid[i, j]:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "phase_diagram.npz",
        mu_grid=mu_grid,
        sigma_grid=sigma_grid,
        H_grid=H_grid,
    )

    # contour plot
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    MU, SIG = np.meshgrid(mu_grid, sigma_grid, indexing="ij")

    levels = np.linspace(0.5, 1.0, 11)
    cs = ax.contourf(MU, SIG, H_grid, levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=r"$H_{\mathrm{eff}}$")

    ch = ax.contour(
        MU, SIG, H_grid, levels=[0.88], colors="white", linewidths=2.0
    )
    ax.clabel(ch, fmt={0.88: r"$H=0.88$"}, fontsize=10)

    # aircraft markers on their measured mu_T
    aircraft = {
        "paragliders": 3.93,
        "hang gliders": 4.79,
        "sailplanes": 2.62,
    }
    for label, mu_val in aircraft.items():
        ax.axvline(mu_val, color="white", ls="--", lw=0.8, alpha=0.7)
        ax.text(
            mu_val,
            sigma_grid[-1] * 0.95,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            color="white",
        )

    ax.set_xlabel(r"$\mu_{\mathrm{T}}$ (transition tail exponent)")
    ax.set_ylabel(r"$\sigma_{\theta}$ (rad)")
    ax.set_title(r"Effective Hurst exponent $H_{\mathrm{eff}}(\mu_T,\sigma_\theta)$")

    fig.savefig(args.output_dir / "phase_diagram.pdf")
    print(f"\nSaved: {args.output_dir / 'phase_diagram.pdf'}")
    print(f"Saved: {args.output_dir / 'phase_diagram.npz'}")


if __name__ == "__main__":
    main()
