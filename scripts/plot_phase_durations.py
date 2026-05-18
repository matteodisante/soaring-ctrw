#!/usr/bin/env python3
"""Plot Lomax CCDF for transition and search phases across aircraft."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def lomax_ccdf(t: np.ndarray, mu: float, tau_0: float) -> np.ndarray:
    """CCDF for Lomax with survival-tail exponent mu and scale tau_0."""
    return (1.0 + t / tau_0) ** (-mu)


def mean_lomax(mu: float, tau_0: float) -> float:
    """Mean of Lomax distribution, valid for mu > 1."""
    return tau_0 / (mu - 1.0)


def plot_phase(
    ax,
    phase_data: dict,
    phase_name: str,
    t_min: float = 1e-1,
    t_max: float = 1e4,
) -> None:
    """Plot a single phase with three aircraft curves."""
    colors = {
        "paragliders": "tab:orange",
        "sailplanes": "tab:purple",
        "hang_gliders": "tab:blue",
    }
    aircraft_names = {
        "paragliders": "Paragliders",
        "sailplanes": "Sailplanes",
        "hang_gliders": "Hang gliders",
    }

    t = np.logspace(np.log10(t_min), np.log10(t_max), 800)

    for aircraft_key, (tau_0, mu) in phase_data.items():
        color = colors[aircraft_key]
        name = aircraft_names[aircraft_key]

        ccdf = lomax_ccdf(t, mu=mu, tau_0=tau_0)
        mean_val = mean_lomax(mu=mu, tau_0=tau_0)
        
        ax.loglog(t, ccdf, color=color, lw=2.2, label=f"{name}: μ={mu}, ⟨τ⟩={mean_val:.1f}s")

        ax.axvline(
            mean_val,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel(r"$\tau$ (s)", fontsize=13)
    ax.set_ylabel(rf"$P(\tau_{{{phase_name.lower()}}} > \tau)$", fontsize=13)
    ax.set_title(f"{phase_name} Phase", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10, loc="lower left")


def plot_climb_phase(
    ax,
    climb_data: dict,
    phase_name: str = "Climb",
    t_min: float = 1e0,
    t_max: float = 1e3,
) -> None:
    """Plot climb phase with exponential CCDF."""
    colors = {
        "paragliders": "tab:orange",
        "sailplanes": "tab:purple",
        "hang_gliders": "tab:blue",
    }
    aircraft_names = {
        "paragliders": "Paragliders",
        "sailplanes": "Sailplanes",
        "hang_gliders": "Hang gliders",
    }

    t = np.logspace(np.log10(t_min), np.log10(t_max), 800)

    for aircraft_key, mu_eff in climb_data.items():
        color = colors[aircraft_key]
        name = aircraft_names[aircraft_key]

        ccdf = np.exp(-t / mu_eff)
        mean_val = 1.0 / mu_eff
        
        ax.loglog(t, ccdf, color=color, lw=2.2, label=f"{name}: ⟨τ⟩=μ_eff={mu_eff:.0f} s")

        ax.axvline(
            mean_val,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel(r"$\tau$ (s)", fontsize=13)
    ax.set_ylabel(rf"$P(\tau_{{{phase_name.lower()}}} > \tau)$", fontsize=13)
    ax.set_title(f"{phase_name} Phase", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_xlim(t_min, t_max)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10, loc="lower left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Lomax CCDF for transition and search phases, "
            "and exponential CCDF for climb phase across aircraft."
        )
    )
    parser.add_argument(
        "--output",
        default="phase_durations_ccdf.png",
        help="Output figure path (default: phase_durations_ccdf.png).",
    )
    args = parser.parse_args()

    transition_data = {
        "paragliders": (450.0, 3.93),
        "sailplanes": (500.0, 2.62),
        "hang_gliders": (400.0, 4.79),
    }

    search_data = {
        "paragliders": (150.0, 3.88),
        "sailplanes": (50.0, 2.9),
        "hang_gliders": (80.0, 2.25),
    }

    climb_data = {
        "paragliders": 110.0,
        "sailplanes": 110.0,
        "hang_gliders": 110.0,
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    plot_phase(ax1, transition_data, "Transition", t_min=1e0, t_max=1e4)
    plot_phase(ax2, search_data, "Search", t_min=1e-1, t_max=1e3)
    plot_climb_phase(ax3, climb_data, "Climb", t_min=1e0, t_max=1e3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
