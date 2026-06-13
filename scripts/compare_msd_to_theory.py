"""Compare simulated MSDs to the analytical formulas of ``soaring_ctrw.tex``.

Equation numbers in the comments below are indicative; the authoritative
references are the ``\\label`` names in the manuscript (e.g.
``eq:msd-search``, ``eq:msd-climb``, ``eq:msd-closed``,
``eq:Heff-convex``, ``eq:GN``, ``eq:SigmaS``, ``eq:SigmaC``).

Four comparison figures, one per analytical formula. Each figure has
three panels (one per aircraft):

  - ``compare_theory_msd_search.pdf`` — Eq. 10 of the paper vs the
    simulated per-phase search MSD.
  - ``compare_theory_msd_climb.pdf``  — Eq. 11 vs the simulated
    per-phase climb MSD.
  - ``compare_theory_msd_cycles.pdf`` — Eq. 23 vs the cycle-counted
    MSD ``⟨|X_N|²⟩`` simulated by recording the position at the end
    of every cycle. This is the cleanest test of the closed-form MSD:
    no physical-time-to-cycle conversion is involved.
  - ``compare_theory_Heff.pdf``       — Eq. 26 vs the numerical local
    log-log slope ``H_eff(N) = (1/2) d log ⟨|X_N|²⟩ / d log N`` of
    the same cycle-counted MSD, plotted vs cycle count ``N``. The
    simulated slope is fit on a multiplicative window in log N,
    selectable via ``--win-factor``.

The cycle-counted simulation is run live (no cache); it is small and
fast (``--n-cycles`` × ``--n-traj-cycles`` trajectories per aircraft).

Inputs:

  - ``outputs/data/per_phase_msd/<aircraft>.npz``     (msd_S, msd_C, lags)
  - ``outputs/data/calibration/<aircraft>.yaml``      (sigma_theta)
  - ``configs/<aircraft>.yaml``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from soaring_ctrw.calibration import load_calibrated_config, write_calibration_section
from soaring_ctrw.model import SoaringConfig
from soaring_ctrw.paths import CONFIGS_DIR, DATA_DIR, FIGURES_DIR
from soaring_ctrw.simulation import simulate_single
from soaring_ctrw.theory import (
    G_N,
    G_N_prime,
    Heff_theory_N,
    climb_msd_exact,
    climb_msd_theory,
    compute_AB,
    compute_Sigma_C,
    compute_Sigma_S,
    lomax_alpha_moment,
    lomax_mean,
    lomax_var,
    search_msd_long,
    search_msd_short,
)

AIRCRAFT_ORDER = ("paragliders", "hang_gliders", "sailplanes")
AIRCRAFT_LABELS = {
    "paragliders": "paragliders",
    "hang_gliders": "hang gliders",
    "sailplanes": "sailplanes",
}
PANEL_LETTERS = ("a", "b", "c")
COLOR_SIM = "tab:blue"
COLOR_THY = "tab:red"


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------
# All closed-form expressions live in ``soaring_ctrw.theory`` (single
# source of truth, unit-tested in ``tests/test_theory.py``) and are
# re-exported here so external users of this script keep working.


def local_loglog_slope(x: np.ndarray, y: np.ndarray,
                       win_factor: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Numerical ``d log y / d log x`` using a local multiplicative
    window. Returns ``(x_valid, slope)``."""
    mask = (x > 0) & np.isfinite(y) & (y > 0)
    X = x[mask]; Y = y[mask]
    logX = np.log(X); logY = np.log(Y)
    slope = np.full(len(X), np.nan)
    for i, xi in enumerate(X):
        win = (X > xi / win_factor) & (X < xi * win_factor)
        if win.sum() < 3:
            continue
        s, _ = np.polyfit(logX[win], logY[win], 1)
        slope[i] = float(s)
    return X, slope


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------


def _load_per_phase_msd(aircraft: str, data_dir: Path) -> dict:
    path = data_dir / "per_phase_msd" / f"{aircraft}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Per-phase MSD cache missing: {path}\n"
            f"  Run:  python scripts/plot_per_phase_msd.py --aircraft {aircraft}"
        )
    return dict(np.load(path))




# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _setup_axes(aircrafts: list[str]) -> tuple[plt.Figure, list]:
    n = len(aircrafts)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.6),
                              constrained_layout=True)
    if n == 1:
        axes = [axes]
    return fig, list(axes)


def plot_search(configs: dict[str, SoaringConfig],
                per_phase: dict[str, dict], output_path: Path,
                aircrafts: list[str]) -> None:
    fig, axes = _setup_axes(aircrafts)
    for ax, ac, letter in zip(axes, aircrafts, PANEL_LETTERS):
        cfg = configs[ac]; sm = cfg.search_motion
        lags = per_phase[ac]["lags"]; msd = per_phase[ac]["msd_S"]
        m = (lags > 0) & np.isfinite(msd) & (msd > 0)
        # Sample-percentile band (5--95%): direct percentiles of the
        # per-episode squared search displacements at each lag.
        blo = per_phase[ac].get("q05_S"); bhi = per_phase[ac].get("q95_S")
        if blo is not None and bhi is not None:
            mb = m & np.isfinite(blo) & np.isfinite(bhi) & (blo > 0)
            ax.fill_between(lags[mb], blo[mb], bhi[mb],
                            color=COLOR_SIM, alpha=0.18, lw=0,
                            label="5–95% of episodes")
        ax.loglog(lags[m], msd[m], "o", ms=3, color=COLOR_SIM,
                  alpha=0.7, label="simulated")
        grid = np.geomspace(lags[m].min(), lags[m].max(), 200)
        ball_grid = grid[grid <= 2.0 * sm.tau_b_S]
        sub_grid = grid[grid >= 0.5 * sm.tau_b_S]
        ax.loglog(ball_grid, search_msd_short(ball_grid, sm.u_S),
                  "-", lw=1.8, color=COLOR_THY,
                  label=r"Eq. 10 (short): $u_S^2\,\Delta^2$")
        ax.loglog(sub_grid, search_msd_long(sub_grid, sm.alpha_S, sm.tau_b_S,
                                              sm.tau_turn_S, sm.u_S, sm.Omega_S),
                  "--", lw=1.8, color=COLOR_THY,
                  label=rf"Eq. 10 (long): $\propto\Delta^{{\alpha_S={sm.alpha_S}}}$")
        ax.axvline(sm.tau_b_S, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel(r"$\Delta$  (s)")
        if ax is axes[0]:
            ax.set_ylabel(r"$\delta^2_S(\Delta)$  (m$^2$)")
        ax.set_title(f"({letter}) {AIRCRAFT_LABELS[ac]} — search",
                     fontsize=10, loc="left")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Search MSD: simulation vs paper Eq. 10", fontsize=11)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_climb(configs: dict[str, SoaringConfig],
               per_phase: dict[str, dict], output_path: Path,
               aircrafts: list[str]) -> None:
    fig, axes = _setup_axes(aircrafts)
    for ax, ac, letter in zip(axes, aircrafts, PANEL_LETTERS):
        cm = configs[ac].climb_motion
        lags = per_phase[ac]["lags"]; msd = per_phase[ac]["msd_C"]
        m = (lags > 0) & np.isfinite(msd) & (msd > 0)
        # Sample-percentile band (5--95%): direct percentiles of the
        # per-episode squared climb displacements at each lag.
        blo = per_phase[ac].get("q05_C"); bhi = per_phase[ac].get("q95_C")
        if blo is not None and bhi is not None:
            mb = m & np.isfinite(blo) & np.isfinite(bhi) & (blo > 0)
            ax.fill_between(lags[mb], blo[mb], bhi[mb],
                            color=COLOR_SIM, alpha=0.18, lw=0,
                            label="5–95% of episodes")
        ax.loglog(lags[m], msd[m], "o", ms=3, color=COLOR_SIM,
                  alpha=0.7, label="simulated")
        grid = np.geomspace(lags[m].min(), lags[m].max(), 400)
        theory = climb_msd_theory(grid, cm.r0, cm.T_turn_mean,
                                  cm.T_turn_std, cm.v_drift)
        ax.loglog(grid, theory, "-", lw=1.8, color=COLOR_THY,
                  label="Eq. 11 (closed form)")
        # Reference: exact numerical average over the clipped-Gaussian
        # turn period actually sampled by the simulator. Quantifies the
        # accuracy of the first-order T->omega expansion of Eq. 11,
        # whose error peaks around the turn lag (Appendix D).
        exact = climb_msd_exact(grid, cm.r0, cm.T_turn_mean,
                                cm.T_turn_std, cm.v_drift)
        ax.loglog(grid, exact, ":", lw=1.7, color="0.15",
                  label="period-averaged (exact)")
        ax.axhline(2.0 * cm.r0 ** 2, color="0.5", ls=":", lw=0.8,
                    label=rf"$2 r_0^2 = {2*cm.r0**2:.0f}\,\mathrm{{m}}^2$")
        ax.set_xlabel(r"$\Delta$  (s)")
        if ax is axes[0]:
            ax.set_ylabel(r"$\delta^2_C(\Delta)$  (m$^2$)")
        ax.set_title(f"({letter}) {AIRCRAFT_LABELS[ac]} — climb",
                     fontsize=10, loc="left")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Climb MSD: simulation vs paper Eq. 11", fontsize=11)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def simulate_cycle_msd(cfg: SoaringConfig, n_cycles: int, n_traj: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Simulate ``n_traj`` independent trajectories of exactly
    ``n_cycles`` cycles and return the full matrix ``r2`` of shape
    ``(n_traj, n_cycles + 1)`` with ``r2[m, N] = |X_N|^2`` of
    trajectory ``m`` at the end of cycle ``N``. No time-to-cycle
    conversion is involved.

    The ensemble mean ``r2.mean(axis=0)`` is the cycle-counted MSD the
    closed form predicts; direct percentiles of ``r2`` along axis 0
    give the sample band. The full matrix is kept because,
    for the near-critical classes (``mu_T < 4``: sailplanes and
    paragliders), the per-cycle squared displacement is heavy-tailed
    and both the spread and the sub-ensemble variability are of
    interest. Stored in ``float64``: the heavy-tailed ``|X_N|^2``
    reach ``~1e12 m^2`` and the running mean must not lose precision.
    """
    r2 = np.empty((n_traj, n_cycles + 1), dtype=np.float64)
    for m in range(n_traj):
        traj = simulate_single(cfg, n_cycles=n_cycles, rng=rng)
        r2[m] = (traj.positions ** 2).sum(axis=1)
    return r2


def cycle_band(r2: np.ndarray, q_lo: float = 5.0,
               q_hi: float = 95.0) -> tuple[np.ndarray, np.ndarray]:
    """Direct ``q_lo``–``q_hi`` percentiles (default 5–95) of the
    per-trajectory ``|X_N|^2`` at each ``N`` — the same sample whose
    mean is the cycle-counted MSD. No bootstrap."""
    lo, hi = np.percentile(r2, [q_lo, q_hi], axis=0)
    return lo, hi


def heff_subensemble_band(
    r2: np.ndarray,
    n_groups: int = 10,
    win_factor: float = 2.0,
    q_lo: float = 5.0,
    q_hi: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample-based variability band for the cycle-counted ``H_eff(N)``.

    ``H_eff(N)`` is a functional of the ensemble-*mean* MSD, so a
    per-trajectory percentile band is not defined for it. The
    sample-based analogue, with no resampling involved, splits the
    ``M`` trajectories into ``n_groups`` disjoint sub-ensembles,
    computes the mean MSD and its local log-log slope per group, and
    returns the ``q_lo``–``q_hi`` percentiles of the group curves at
    each ``N``. The spread visualises the finite-sample variability of
    the estimator at ensemble size ``M / n_groups``.

    Returns ``(N_valid, lo, hi)`` with ``H_eff`` values (slope / 2).
    """
    M = r2.shape[0]
    g = max(2, int(n_groups))
    size = M // g
    if size < 2:
        raise ValueError(f"too few trajectories ({M}) for {g} groups")
    n_arr = np.arange(r2.shape[1], dtype=float)
    curves = []
    for k in range(g):
        mean_k = r2[k * size:(k + 1) * size].mean(axis=0)
        Nv, slope = local_loglog_slope(n_arr, mean_k, win_factor=win_factor)
        curves.append(0.5 * slope)
    curves = np.vstack(curves)  # (g, n_valid) — same mask for all groups
    lo = np.nanpercentile(curves, q_lo, axis=0)
    hi = np.nanpercentile(curves, q_hi, axis=0)
    return Nv, lo, hi


def plot_cycles(configs: dict[str, SoaringConfig],
                cycle_msd: dict[str, np.ndarray], output_path: Path,
                aircrafts: list[str],
                cycle_bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                band_label: str = "5–95% of trajectories") -> None:
    fig, axes = _setup_axes(aircrafts)
    for ax, ac, letter in zip(axes, aircrafts, PANEL_LETTERS):
        cfg = configs[ac]
        sq = cycle_msd[ac]
        N_arr = np.arange(len(sq))
        m = N_arr > 0
        if cycle_bands is not None and ac in cycle_bands:
            blo, bhi = cycle_bands[ac]
            mb = m & np.isfinite(blo) & np.isfinite(bhi) & (blo > 0)
            ax.fill_between(N_arr[mb], blo[mb], bhi[mb],
                            color=COLOR_SIM, alpha=0.18, lw=0,
                            label=band_label)
        ax.loglog(N_arr[m], sq[m], "o", ms=4, color=COLOR_SIM,
                  alpha=0.8, label=r"simulated $\langle|\mathbf{X}_N|^2\rangle$")
        A, B, rho, mean_T = compute_AB(cfg)
        N_grid = np.geomspace(1.0, float(len(sq) - 1), 200)
        ax.loglog(N_grid, A * G_N(N_grid, rho) + B * N_grid,
                  "-", lw=1.8, color=COLOR_THY,
                  label=r"Eq. 23: $A\,G_N(\rho) + B\,N$")
        n_c = 2.0 / cfg.angular.sigma_theta ** 2
        ax.axvline(n_c, color="0.5", ls=":", lw=0.8,
                   label=rf"$n_c={n_c:.1f}$")
        ax.set_xlabel(r"$N$  (cycles)")
        if ax is axes[0]:
            ax.set_ylabel(r"$\langle|\mathbf{X}_N|^2\rangle$  (m$^2$)")
        ax.set_title(f"({letter}) {AIRCRAFT_LABELS[ac]} — cycle-counted",
                     fontsize=10, loc="left")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Cycle-counted MSD: simulation vs paper Eq. 23 "
                 "(direct test, no time→cycle conversion)", fontsize=11)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_Heff(configs: dict[str, SoaringConfig],
              cycle_msd: dict[str, np.ndarray], output_path: Path,
              aircrafts: list[str], win_factor: float = 2.0,
              heff_bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
              n_groups: int = 10) -> None:
    """1×N panel of the cycle-counted ``H_eff(N) = (1/2) d log
    ⟨|X_N|²⟩ / d log N``: the local log-log slope of the simulated
    cycle-counted MSD (no time→cycle conversion), compared to Eq. 26
    of the paper evaluated at the same cycle counts.

    The simulated slope is a local fit on a multiplicative window in
    ``log N``: at each N the line is fit over the points falling in
    ``[N / win_factor, N · win_factor]``. The default
    ``win_factor = 2`` gives a roughly half-decade window.

    ``heff_bands`` (optional): per-aircraft ``(N, lo, hi)`` from
    :func:`heff_subensemble_band` — the 5–95% spread of the same
    estimator over disjoint sub-ensembles, visualising its
    finite-sample variability (largest for the heavy-tailed classes).
    """
    fig, axes = _setup_axes(aircrafts)
    for ax, ac, letter in zip(axes, aircrafts, PANEL_LETTERS):
        cfg = configs[ac]
        sq = cycle_msd[ac]
        N_arr = np.arange(len(sq), dtype=float)
        N_sim, slope = local_loglog_slope(N_arr, sq, win_factor=win_factor)
        if heff_bands is not None and ac in heff_bands:
            Nb, blo, bhi = heff_bands[ac]
            ax.fill_between(Nb, blo, bhi, color=COLOR_SIM, alpha=0.18,
                            lw=0,
                            label=f"5–95% over {n_groups} sub-ensembles")
        ax.semilogx(N_sim, 0.5 * slope, "-", lw=1.6, color=COLOR_SIM,
                    label=rf"simulated (local slope, window $\times{win_factor:g}$)")
        N_grid = np.geomspace(max(1.0, N_sim.min()),
                               max(2.0, N_sim.max()), 400)
        ax.semilogx(N_grid, Heff_theory_N(N_grid, cfg),
                    "--", lw=1.8, color=COLOR_THY,
                    label="Eq. 26 (theory)")
        ax.axhline(0.88, color="0.4", ls=":", lw=0.9,
                   label=r"$H = 0.88$ (VDB)")
        ax.axhline(1.0, color="0.7", ls="--", lw=0.6)
        ax.axhline(0.5, color="0.7", ls="--", lw=0.6)
        n_c = 2.0 / cfg.angular.sigma_theta ** 2
        ax.axvline(n_c, color="0.5", ls=":", lw=0.8,
                   label=rf"$n_c={n_c:.1f}$")
        ax.set_xlabel(r"$N$  (cycles)")
        if ax is axes[0]:
            ax.set_ylabel(r"$H_{\mathrm{eff}}(N)$")
        ax.set_ylim(0.3, 1.1)
        ax.set_title(f"({letter}) {AIRCRAFT_LABELS[ac]} — $H_\\mathrm{{eff}}$",
                     fontsize=10, loc="left")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Cycle-counted effective Hurst exponent: "
                 "simulation vs paper Eq. 26", fontsize=11)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_theory_components(N: np.ndarray, cfg: SoaringConfig) -> dict[str, np.ndarray]:
    """Evaluate every analytical building block of Eqs. 23 / 26 at the
    cycle counts ``N`` and return them in a dict.

    Keys:
      - ``msd``      : Eq. 23 MSD, ``A G_N(rho) + B N``.
      - ``G``        : Eq. 21 ``G_N(rho)``.
      - ``Gp``       : ``dG_N/dN`` (used by Eq. 26).
      - ``s_G``      : local log-log slope of ``G_N``, ``N G'_N / G_N``.
      - ``w``        : crossover weight ``A G_N / (A G_N + B N)``.
      - ``one_minus_w``: ``1 - w``.
      - ``Heff``     : Eq. 26 ``H_eff(N) = 0.5 (1 + w (s_G - 1))``.
    """
    A, B, rho, _ = compute_AB(cfg)
    G = G_N(N, rho)
    Gp = G_N_prime(N, rho)
    s_G = N * Gp / G
    w = A * G / (A * G + B * N)
    Heff = 0.5 * (1.0 + w * (s_G - 1.0))
    return {
        "msd": A * G + B * N,
        "G": G,
        "Gp": Gp,
        "s_G": s_G,
        "w": w,
        "one_minus_w": 1.0 - w,
        "Heff": Heff,
    }


def plot_theory_components(configs: dict[str, SoaringConfig],
                           output_path: Path, aircrafts: list[str],
                           n_cycles: int = 60) -> None:
    """Two-row × one-column-per-aircraft figure of the analytical
    functions vs cycle count ``N`` (no simulation involved).

    Curves are grouped by magnitude:

      - top row (log-log): the extensive quantities sharing a large
        dynamic range — the MSD ``A G_N + B N`` (Eq. 23), ``G_N(rho)``
        (Eq. 21) and its derivative ``G'_N``.
      - bottom row (semilog-x, linear ``y``): the dimensionless ``O(1)``
        quantities — the local slope ``s_G``, the crossover weight
        ``w`` and ``1 - w``, and the effective Hurst exponent
        ``H_eff`` (Eq. 26).
    """
    n = len(aircrafts)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8.4),
                             constrained_layout=True, squeeze=False)
    N_grid = np.geomspace(1.0, float(max(2, n_cycles)), 400)
    for col, (ac, letter) in enumerate(zip(aircrafts, PANEL_LETTERS)):
        cfg = configs[ac]
        comp = compute_theory_components(N_grid, cfg)
        n_c = 2.0 / cfg.angular.sigma_theta ** 2

        # --- top row: extensive quantities (log-log) ---------------
        ax_top = axes[0][col]
        ax_top.loglog(N_grid, comp["msd"], "-", lw=1.8, color="tab:red",
                      label=r"MSD: $A\,G_N(\rho)+B\,N$ (Eq. 23)")
        ax_top.loglog(N_grid, comp["G"], "-", lw=1.6, color="tab:blue",
                      label=r"$G_N(\rho)$ (Eq. 21)")
        ax_top.loglog(N_grid, comp["Gp"], "--", lw=1.6, color="tab:green",
                      label=r"$G'_N = \mathrm{d}G_N/\mathrm{d}N$")
        ax_top.axvline(n_c, color="0.5", ls=":", lw=0.8,
                       label=rf"$n_c={n_c:.1f}$")
        ax_top.set_xlabel(r"$N$  (cycles)")
        if col == 0:
            ax_top.set_ylabel("extensive quantities")
        ax_top.set_title(f"({letter}) {AIRCRAFT_LABELS[ac]}",
                         fontsize=10, loc="left")
        ax_top.grid(True, which="both", ls=":", alpha=0.4)
        ax_top.legend(fontsize=8.5, loc="upper left")

        # --- bottom row: dimensionless O(1) quantities (semilog-x) --
        ax_bot = axes[1][col]
        ax_bot.semilogx(N_grid, comp["s_G"], "-", lw=1.6, color="tab:purple",
                        label=r"$s_G = N\,G'_N/G_N$")
        ax_bot.semilogx(N_grid, comp["w"], "-", lw=1.6, color="tab:orange",
                        label=r"$w = A\,G_N/(A\,G_N+B\,N)$")
        ax_bot.semilogx(N_grid, comp["one_minus_w"], "--", lw=1.4,
                        color="tab:brown", label=r"$1-w$")
        ax_bot.semilogx(N_grid, comp["Heff"], "-", lw=2.0, color="tab:red",
                        label=r"$H_{\mathrm{eff}}$ (Eq. 26)")
        ax_bot.axhline(1.0, color="0.7", ls="--", lw=0.6)
        ax_bot.axhline(0.5, color="0.7", ls="--", lw=0.6)
        ax_bot.axvline(n_c, color="0.5", ls=":", lw=0.8,
                       label=rf"$n_c={n_c:.1f}$")
        ax_bot.set_xlabel(r"$N$  (cycles)")
        if col == 0:
            ax_bot.set_ylabel("dimensionless quantities")
        ax_bot.grid(True, which="both", ls=":", alpha=0.4)
        ax_bot.legend(fontsize=8.5, loc="center left")
    fig.suptitle("Analytical components of Eqs. 21 / 23 / 26 vs cycle "
                 "count $N$", fontsize=11)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aircraft", nargs="+",
                        default=list(AIRCRAFT_ORDER),
                        choices=list(AIRCRAFT_ORDER))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--win-factor", type=float, default=2.0,
                        help="Multiplicative window for the local "
                             "log-log slope of the simulated MSD vs N.")
    parser.add_argument("--n-cycles", type=int, default=60,
                        help="Number of cycles per trajectory for the "
                             "cycle-counted MSD comparison.")
    parser.add_argument("--n-traj-cycles", type=int, default=10000,
                        help="Trajectories for the cycle-counted MSD "
                             "(default for all aircraft). Heavy-tailed "
                             "transition durations (mu_T < 4) make the "
                             "estimator of Var(tau_T) converge slowly, "
                             "so prefer large ensembles.")
    parser.add_argument(
        "--n-traj-cycles-by-aircraft", nargs="*", default=[],
        metavar="AIRCRAFT=N",
        help=(
            "Per-aircraft override of --n-traj-cycles, e.g. "
            "`--n-traj-cycles-by-aircraft sailplanes=120000`. Required "
            "in practice for sailplanes (mu_T=2.62 close to 2): the "
            "estimator error decays only as M^{-(1-2/mu_T)} ~ M^-0.24, "
            "so the recommended production value is sailplanes=120000. "
            "Aircraft not listed use --n-traj-cycles."
        ),
    )
    parser.add_argument(
        "--n-groups", type=int, default=10,
        help="Disjoint sub-ensembles used for the 5-95%% variability "
             "band of the cycle-counted H_eff(N).",
    )
    parser.add_argument(
        "--q-lo", type=float, default=5.0,
        help="Lower percentile of the sample bands (default 5).",
    )
    parser.add_argument(
        "--q-hi", type=float, default=95.0,
        help="Upper percentile of the sample bands (default 95).",
    )
    parser.add_argument("--seed-cycles", type=int, default=42,
                        help="RNG seed for the cycle-counted MSD.")
    parser.add_argument(
        "--write", action="store_true",
        help=(
            "Also persist the climb-circling pulsation omega_0 and its "
            "jitter sigma_omega (computed from T_turn_mean / T_turn_std "
            "of the YAML config) into the `climb_circling` section of "
            "outputs/data/calibration/<aircraft>.yaml."
        ),
    )
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Load every needed input upfront (fail-fast if any cache missing).
    configs: dict[str, SoaringConfig] = {}
    per_phase: dict[str, dict] = {}
    for ac in args.aircraft:
        configs[ac] = load_calibrated_config(ac)
        per_phase[ac] = _load_per_phase_msd(ac, args.data_dir)
        cfg = configs[ac]
        A, B, rho, mean_T = compute_AB(cfg)
        cm = cfg.climb_motion
        omega_0 = 2.0 * np.pi / cm.T_turn_mean
        sigma_omega = omega_0 * cm.T_turn_std / cm.T_turn_mean
        # Breakdown of A and B (Eq. 23) and their v_xy^2-rescaled forms.
        v_xy2 = cfg.v_xy ** 2
        var_T_phase = lomax_var(cfg.transition.params["mu"],
                                cfg.transition.params["tau_0"])
        B_transition = v_xy2 * var_T_phase
        Sigma_S = compute_Sigma_S(cfg)
        Sigma_C = compute_Sigma_C(cfg)
        print(f"\n=== {AIRCRAFT_LABELS.get(ac, ac)} ===")
        print( "  calibration:")
        print(f"    sigma_theta = {cfg.angular.sigma_theta:.4f}"
              f"      n_c = 2/sigma_theta^2 = {2/cfg.angular.sigma_theta**2:.2f}")
        print(f"    <T>         = {mean_T:.1f} s"
              f"      rho = {rho:.4f}")
        print( "  MSD coefficients (Eq. 23):")
        print(f"    A = {A:.4g} m^2        A_hat = A/v_xy^2 = {A / v_xy2:.4g} s^2")
        print(f"    B = {B:.4g} m^2        B_hat = B/v_xy^2 = {B / v_xy2:.4g} s^2")
        print( "  B breakdown                        [m^2]            [s^2 = /v_xy^2]")
        print(f"    v_xy^2 * Var(T_T)       = {B_transition:>12.4g}   {var_T_phase:>14.4g}")
        print(f"    Sigma_S                 = {Sigma_S:>12.4g}   {Sigma_S / v_xy2:>14.4g}")
        print(f"    Sigma_C                 = {Sigma_C:>12.4g}   {Sigma_C / v_xy2:>14.4g}")
        print( "  climb circling:")
        print(f"    T_turn = {cm.T_turn_mean:.3f} +/- {cm.T_turn_std:.3f} s")
        print(f"    omega_0 = 2*pi/T_turn_mean              = {omega_0:.6f} rad/s")
        print(f"    sigma_omega = omega_0*T_turn_std/T_turn_mean = {sigma_omega:.6f} rad/s")
        if args.write:
            payload = {
                "source_script": "compare_msd_to_theory",
                "T_turn_mean": float(cm.T_turn_mean),
                "T_turn_std": float(cm.T_turn_std),
                "omega_0": float(omega_0),
                "sigma_omega": float(sigma_omega),
                "formula_omega_0": "2*pi / T_turn_mean",
                "formula_sigma_omega":
                    "omega_0 * T_turn_std / T_turn_mean",
            }
            out = write_calibration_section(ac, "climb_circling", payload)
            print(
                f"  {ac}: wrote {out}  (section: climb_circling, "
                f"omega_0={omega_0:.6f} rad/s)"
            )

    plot_search(configs, per_phase,
                args.figures_dir / "compare_theory_msd_search.pdf",
                list(args.aircraft))
    plot_climb(configs, per_phase,
               args.figures_dir / "compare_theory_msd_climb.pdf",
               list(args.aircraft))

    # Cycle-counted MSD — direct test of Eq. 23 without time→cycle
    # conversion. Runs a fresh small simulation per aircraft. The same
    # data feeds the H_eff(N) plot below.
    n_traj_map: dict[str, int] = {ac: args.n_traj_cycles for ac in args.aircraft}
    for spec in args.n_traj_cycles_by_aircraft:
        if "=" not in spec:
            parser.error(
                f"--n-traj-cycles-by-aircraft expects AIRCRAFT=N, got {spec!r}"
            )
        key, val = spec.split("=", 1)
        if key not in AIRCRAFT_ORDER:
            parser.error(
                f"Unknown aircraft {key!r} in --n-traj-cycles-by-aircraft; "
                f"choose from {list(AIRCRAFT_ORDER)}."
            )
        n_traj_map[key] = int(val)
    print(
        "\nCycle-counted MSD ("
        + ", ".join(f"{ac}: {n_traj_map[ac]} traj" for ac in args.aircraft)
        + f" × {args.n_cycles} cycles)..."
    )
    cycle_msd: dict[str, np.ndarray] = {}
    cycle_bands: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    heff_bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, ac in enumerate(args.aircraft):
        # Deterministic per-aircraft offset (avoid Python's randomised
        # str hash, which makes runs non-reproducible across processes).
        rng = np.random.default_rng(args.seed_cycles + i)
        r2 = simulate_cycle_msd(
            configs[ac], n_cycles=args.n_cycles,
            n_traj=n_traj_map[ac], rng=rng,
        )
        cycle_msd[ac] = r2.mean(axis=0)
        cycle_bands[ac] = cycle_band(r2, q_lo=args.q_lo, q_hi=args.q_hi)
        heff_bands[ac] = heff_subensemble_band(
            r2, n_groups=args.n_groups, win_factor=args.win_factor,
            q_lo=args.q_lo, q_hi=args.q_hi,
        )
        del r2
        print(f"  {ac}: done")
    plot_cycles(configs, cycle_msd,
                args.figures_dir / "compare_theory_msd_cycles.pdf",
                list(args.aircraft), cycle_bands=cycle_bands,
                band_label=f"{args.q_lo:g}–{args.q_hi:g}% of trajectories")
    plot_Heff(configs, cycle_msd,
              args.figures_dir / "compare_theory_Heff.pdf",
              list(args.aircraft),
              win_factor=args.win_factor,
              heff_bands=heff_bands, n_groups=args.n_groups)

    # Pure-theory breakdown of Eqs. 21 / 23 / 26 vs N (no simulation).
    plot_theory_components(configs,
                           args.figures_dir / "compare_theory_components.pdf",
                           list(args.aircraft),
                           n_cycles=args.n_cycles)

    print("\nSaved figures:")
    for name in ("msd_search", "msd_climb", "msd_cycles", "Heff", "components"):
        print(f"  {args.figures_dir / f'compare_theory_{name}.pdf'}")


if __name__ == "__main__":
    main()
