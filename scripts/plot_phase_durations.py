#!/usr/bin/env python3
"""Validate the phase-duration samplers against theoretical distributions.

Layout: 2 rows × 3 columns (transition | search | climb).

  Top row    — CCDF log-log:
      solid line   = theoretical S(τ) = P(T > τ)
      dashed line  = empirical ECDF S_N(τ) from Monte Carlo samples
  Bottom row — PDF log-log:
      step outline = log-spaced histogram (PDF normalised)
      solid line   = theoretical f(τ)

The two should visually coincide in every panel, which validates that
the ``LomaxTail`` and ``Exponential`` samplers in
``src/distributions.py`` implement the intended distributions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from soaring_ctrw.distributions import Exponential, LomaxTail  # noqa: E402
from soaring_ctrw.model import SoaringConfig  # noqa: E402
from soaring_ctrw.paths import CONFIGS_DIR, FIGURES_DIR  # noqa: E402

_COLORS: dict[str, str] = {
    "paragliders":  "tab:orange",
    "sailplanes":   "tab:purple",
    "hang_gliders": "tab:blue",
}
_NAMES: dict[str, str] = {
    "paragliders":  "Paragliders",
    "sailplanes":   "Sailplanes",
    "hang_gliders": "Hang gliders",
}


# ---------------------------------------------------------------------------
# Empirical helpers
# ---------------------------------------------------------------------------

def _empirical_survival(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_sorted, S) for the empirical survival function.

    Uses S(x_{(i)}) = (n − i) / n  (1-based rank i = 1 … n), so the
    largest sample maps to 1/n > 0 (safe on a log scale).
    """
    x = np.sort(samples)
    n = len(x)
    s = (n - np.arange(n)) / n
    return x, s


def _subsample_log(
    x: np.ndarray, y: np.ndarray, n_pts: int = 2_000
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ~n_pts rows uniformly spaced on a log scale."""
    n = len(x)
    if n <= n_pts:
        return x, y
    idx = np.unique(
        np.round(np.logspace(0, np.log10(n - 1), n_pts)).astype(int)
    )
    return x[idx], y[idx]


def _histogram_step(
    samples: np.ndarray, t_min: float, t_max: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a step-function (x, y) for a log-spaced histogram.

    Returns arrays suitable for ``ax.loglog`` that draw the histogram
    outline (horizontal segments per bin + vertical connectors). Empty
    bins are filled with ``nan`` so the line breaks there cleanly.
    """
    bins = np.logspace(np.log10(t_min), np.log10(t_max), n_bins + 1)
    counts, edges = np.histogram(samples, bins=bins)
    pdf = counts / (len(samples) * np.diff(edges))

    # Build step:  [e0, e1, e1, e2, e2, …, e_{n−1}, e_n]
    #              [p0, p0, p1, p1, …, p_{n−1}, p_{n−1}]
    # At repeated edge values (e.g. e1, e1), matplotlib draws a vertical
    # connector from p0 to p1 — giving the classic histogram staircase.
    x = np.concatenate([[edges[0]], np.repeat(edges[1:-1], 2), [edges[-1]]])
    y = np.repeat(pdf, 2).astype(float)
    y[y == 0] = np.nan  # break line at empty bins (safe for log scale)
    return x, y


# ---------------------------------------------------------------------------
# CCDF panels (top row)
# ---------------------------------------------------------------------------

def _ccdf_panel(
    ax,
    theory_curves: dict[str, tuple],   # key → (x_theory, y_theory, mean_val, label)
    samples_dict: dict[str, np.ndarray],
    phase_name: str,
    t_min: float,
    t_max: float,
) -> None:
    """Plot theoretical CCDF (solid) + empirical ECDF (dashed) on log-log."""
    for i, (key, (x_th, y_th, mean_val, lbl)) in enumerate(theory_curves.items()):
        color = _COLORS[key]
        ax.loglog(x_th, y_th, color=color, lw=2.2, label=lbl, zorder=4)
        ax.axvline(mean_val, color=color, ls="--", lw=1.0, alpha=0.45)

        x_ec, s_ec = _empirical_survival(samples_dict[key])
        x_sub, s_sub = _subsample_log(x_ec, s_ec, 2_000)
        ax.step(
            x_sub, s_sub, where="post",
            color=color, lw=1.0, ls="--", alpha=0.75, zorder=3,
            label="_",  # suppress from legend; colour ties it to theory
        )

    # Single legend proxy for all "simulated" lines
    ax.plot([], [], color="gray", ls="--", lw=1.0, label="simulated")

    ax.set_xlabel(r"$\tau$ (s)", fontsize=11)
    ax.set_ylabel(
        rf"$P(\tau_{{\mathrm{{{phase_name.lower()}}}}} > \tau)$", fontsize=11
    )
    ax.set_title(f"{phase_name} — CCDF (Survival Function)", fontsize=12, fontweight="bold")
    ax.set_xlim(t_min, t_max)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9, loc="lower left")


# ---------------------------------------------------------------------------
# PDF panels (bottom row)
# ---------------------------------------------------------------------------

def _pdf_panel(
    ax,
    theory_curves: dict[str, tuple],   # key → (x_theory, y_theory, ...)
    samples_dict: dict[str, np.ndarray],
    phase_name: str,
    t_min: float,
    t_max: float,
    n_bins: int,
) -> None:
    """Plot theoretical PDF (solid) + log-spaced histogram step (dashed)."""
    for key, (x_th, y_pdf, *_) in theory_curves.items():
        color = _COLORS[key]
        ax.loglog(x_th, y_pdf, color=color, lw=2.2, zorder=4)

        x_h, y_h = _histogram_step(samples_dict[key], t_min, t_max, n_bins)
        ax.loglog(x_h, y_h, color=color, lw=1.0, ls="--", alpha=0.75, zorder=3)

    ax.plot([], [], color="gray", ls="--", lw=1.0, label="simulated")
    ax.legend(fontsize=9, loc="lower left")

    ax.set_xlabel(r"$\tau$ (s)", fontsize=11)
    ax.set_ylabel(
        rf"$f(\tau_{{\mathrm{{{phase_name.lower()}}}}})$", fontsize=11
    )
    ax.set_title(f"{phase_name} — PDF", fontsize=12, fontweight="bold")
    ax.set_xlim(t_min, t_max)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.tick_params(axis="both", labelsize=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=str(FIGURES_DIR / "phase_durations_ccdf.pdf"),
        help="Output figure path (default: outputs/figures/phase_durations_ccdf.pdf).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50_000,
        help="Monte Carlo samples drawn per distribution.",
    )
    parser.add_argument(
        "--n-bins", type=int, default=40,
        help="Number of log-spaced histogram bins (PDF panels).",
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=CONFIGS_DIR,
        help="Directory containing <aircraft>.yaml config files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- Load Lomax (transition, search) and Exponential (climb) parameters
    # from the per-aircraft YAML configs (configs/<aircraft>.yaml). ---
    aircraft_keys = list(_NAMES.keys())
    configs: dict[str, SoaringConfig] = {
        k: SoaringConfig.from_yaml(args.configs_dir / f"{k}.yaml")
        for k in aircraft_keys
    }

    def _phase_params(phase: str, expected_dist: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k, cfg in configs.items():
            phase_cfg = getattr(cfg, phase)
            if phase_cfg.distribution.lower() != expected_dist:
                raise ValueError(
                    f"{k}.yaml: phase {phase!r} has distribution "
                    f"{phase_cfg.distribution!r}, expected {expected_dist!r}."
                )
            out[k] = dict(phase_cfg.params)
        return out

    lomax_params: dict[str, dict[str, tuple[float, float]]] = {
        "transition": {
            k: (float(p["tau_0"]), float(p["mu"]))
            for k, p in _phase_params("transition", "lomax").items()
        },
        "search": {
            k: (float(p["tau_0"]), float(p["mu"]))
            for k, p in _phase_params("search", "lomax").items()
        },
    }
    climb_params: dict[str, float] = {
        k: float(p["tau_mean"])
        for k, p in _phase_params("climb", "exponential").items()
    }

    # --- Draw all samples up-front (shared between CCDF and PDF panels) ---
    all_samples: dict[str, dict[str, np.ndarray]] = {}
    for phase, params in lomax_params.items():
        all_samples[phase] = {
            k: LomaxTail(mu=mu, tau_0=tau_0).sample(args.n_samples, rng)
            for k, (tau_0, mu) in params.items()
        }
    all_samples["climb"] = {
        k: Exponential(tau_mean=tm).sample(args.n_samples, rng)
        for k, tm in climb_params.items()
    }

    # --- Build theoretical curve dicts ---
    # Structure:  key → (x_grid, y_ccdf,  mean_val, label)   [CCDF row]
    #             key → (x_grid, y_pdf,   mean_val, label)   [PDF row]
    limits = {
        "transition": (1e0, 1e4),
        "search":     (1e-1, 1e3),
        "climb":      (1e0,  1e3),
    }

    def _lomax_curves(params, t_min, t_max):
        out = {}
        t = np.logspace(np.log10(t_min), np.log10(t_max), 800)
        for k, (tau_0, mu) in params.items():
            mean_v = LomaxTail(mu=mu, tau_0=tau_0).mean
            ccdf = (1.0 + t / tau_0) ** (-mu)
            pdf  = (mu / tau_0) * (1.0 + t / tau_0) ** (-(mu + 1))
            lbl  = (
                rf"{_NAMES[k]}: $\mu$={mu}, "
                rf"$\langle\tau\rangle$={mean_v:.1f} s"
            )
            out[k] = (t, ccdf, mean_v, lbl, t, pdf)
        return out  # key → (t, ccdf, mean, lbl, t, pdf)

    def _exp_curves(params, t_min, t_max):
        out = {}
        t = np.logspace(np.log10(t_min), np.log10(t_max), 800)
        for k, tm in params.items():
            ccdf = np.exp(-t / tm)
            pdf  = np.exp(-t / tm) / tm
            lbl  = rf"{_NAMES[k]}: $\langle\tau\rangle$={tm:.0f} s"
            out[k] = (t, ccdf, tm, lbl, t, pdf)
        return out

    tr_curves  = _lomax_curves(lomax_params["transition"], *limits["transition"])
    sr_curves  = _lomax_curves(lomax_params["search"],     *limits["search"])
    cl_curves  = _exp_curves(climb_params,                  *limits["climb"])

    # Unpack into separate CCDF and PDF dicts
    def _split(curves):
        ccdf_d = {k: v[:4] for k, v in curves.items()}   # (t, ccdf, mean, lbl)
        pdf_d  = {k: (v[4], v[5], v[2], v[3]) for k, v in curves.items()}
        return ccdf_d, pdf_d

    tr_ccdf, tr_pdf = _split(tr_curves)
    sr_ccdf, sr_pdf = _split(sr_curves)
    cl_ccdf, cl_pdf = _split(cl_curves)

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Phase-duration distributions: theory vs. simulated  "
        f"(N = {args.n_samples:,} samples per distribution)",
        fontsize=13, y=1.01,
    )

    # Top row — CCDF
    _ccdf_panel(axes[0, 0], tr_ccdf, all_samples["transition"],
                "Transition", *limits["transition"])
    _ccdf_panel(axes[0, 1], sr_ccdf, all_samples["search"],
                "Search",     *limits["search"])
    _ccdf_panel(axes[0, 2], cl_ccdf, all_samples["climb"],
                "Climb",      *limits["climb"])

    # Bottom row — PDF histograms
    _pdf_panel(axes[1, 0], tr_pdf, all_samples["transition"],
               "Transition", *limits["transition"], args.n_bins)
    _pdf_panel(axes[1, 1], sr_pdf, all_samples["search"],
               "Search",     *limits["search"],     args.n_bins)
    _pdf_panel(axes[1, 2], cl_pdf, all_samples["climb"],
               "Climb",      *limits["climb"],      args.n_bins)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
