"""Compare simulated MSDs to the analytical formulas of ``paper_prl.tex``.

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
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_fn

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration import load_calibrated_config
from model import SoaringConfig
from paths import CONFIGS_DIR, DATA_DIR, FIGURES_DIR
from simulation import simulate_single

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
# Analytical helpers (paper_prl.tex equation numbers in comments)
# ---------------------------------------------------------------------------


def lomax_mean(mu: float, tau_0: float) -> float:
    return tau_0 / (mu - 1.0)


def lomax_var(mu: float, tau_0: float) -> float:
    """Var of Lomax (finite for mu > 2)."""
    if mu <= 2:
        return float("inf")
    return mu * tau_0 ** 2 / ((mu - 1.0) ** 2 * (mu - 2.0))


def lomax_alpha_moment(alpha: float, mu: float, tau_0: float) -> float:
    """E[tau^alpha] for Lomax with parameters (mu, tau_0). Finite for mu > alpha."""
    return tau_0 ** alpha * gamma_fn(alpha + 1) * gamma_fn(mu - alpha) / gamma_fn(mu)


# --- Eq. 10: conditional search MSD (asymptotic branches) ----------------

def search_msd_short(lags: np.ndarray, u_S: float) -> np.ndarray:
    """Short-lag ballistic branch of Eq. 10: u_S^2 Δ^2."""
    return u_S ** 2 * lags ** 2


def search_msd_long(lags: np.ndarray, alpha_S: float, tau_b_S: float,
                    tau_turn_S: float, u_S: float) -> np.ndarray:
    """Long-lag subdiffusive branch of Eq. 10."""
    return (
        2.0 * u_S ** 2 * tau_b_S ** 2
        / (tau_turn_S ** alpha_S * gamma_fn(1.0 + alpha_S))
        * lags ** alpha_S
    )


# --- Eq. 11: climb MSD ----------------------------------------------------

def climb_msd_theory(lags: np.ndarray, r0: float, T_turn_mean: float,
                     T_turn_std: float, v_drift: float) -> np.ndarray:
    omega_bar = 2.0 * np.pi / T_turn_mean
    sigma_omega = omega_bar * T_turn_std / T_turn_mean
    return (
        2.0 * r0 ** 2
        * (1.0 - np.exp(-0.5 * sigma_omega ** 2 * lags ** 2)
                * np.cos(omega_bar * lags))
        + v_drift ** 2 * lags ** 2
    )


# --- Eq. 21 / 23 / 26: full MSD and H_eff ---------------------------------

def G_N(N: np.ndarray, rho: float) -> np.ndarray:
    """Eq. 21 of the paper, evaluated at continuous N."""
    return N * (1.0 + rho) / (1.0 - rho) - 2.0 * rho * (1.0 - rho ** N) / (1.0 - rho) ** 2


def G_N_prime(N: np.ndarray, rho: float) -> np.ndarray:
    """d G_N / dN, treating N as continuous (used by Eq. 26)."""
    return (1.0 + rho) / (1.0 - rho) + 2.0 * rho ** (N + 1.0) * np.log(rho) / (1.0 - rho) ** 2


def compute_Sigma_S(cfg: SoaringConfig) -> float:
    """Per-cycle search displacement variance, Eq. 36.

    Uses T_phys^S = tau_S^n (Lomax) — the new physical-duration
    stopping rule — so ⟨(T_phys^S)^alpha_S⟩ is the Lomax alpha-moment.
    """
    sm = cfg.search_motion
    if sm is None:
        return 0.0
    mu_S = cfg.search.params["mu"]
    tau_0_S = cfg.search.params["tau_0"]
    T_phys_alpha = lomax_alpha_moment(sm.alpha_S, mu_S, tau_0_S)
    return (
        2.0 * sm.u_S ** 2 * sm.tau_b_S ** 2
        / (sm.tau_turn_S ** sm.alpha_S * gamma_fn(1.0 + sm.alpha_S))
        * T_phys_alpha
    )


def compute_Sigma_C(cfg: SoaringConfig) -> float:
    """Per-cycle climb displacement variance, Eq. 37."""
    cm = cfg.climb_motion
    if cm is None:
        return 0.0
    mu_C = cfg.climb.params["tau_mean"]
    omega_bar = 2.0 * np.pi / cm.T_turn_mean
    sigma_omega = omega_bar * cm.T_turn_std / cm.T_turn_mean
    A = 1.0 + omega_bar * mu_C
    B = 0.5 * sigma_omega ** 2 * mu_C ** 2
    Re_phi = A / (A ** 2 + B ** 2)
    return 2.0 * cm.r0 ** 2 * (1.0 - Re_phi) + 2.0 * cm.v_drift ** 2 * mu_C ** 2


def compute_AB(cfg: SoaringConfig) -> tuple[float, float, float, float]:
    """Return ``(A, B, rho, mean_T)`` of Eq. 23."""
    v_xy = cfg.v_xy
    mu_T = cfg.transition.params["mu"]
    tau_0_T = cfg.transition.params["tau_0"]
    mean_T_phase = lomax_mean(mu_T, tau_0_T)
    var_T_phase = lomax_var(mu_T, tau_0_T)
    mean_S_phase = lomax_mean(cfg.search.params["mu"], cfg.search.params["tau_0"])
    mean_C_phase = cfg.climb.params["tau_mean"]
    A = (v_xy * mean_T_phase) ** 2
    B = v_xy ** 2 * var_T_phase + compute_Sigma_S(cfg) + compute_Sigma_C(cfg)
    rho = float(np.exp(-cfg.angular.sigma_theta ** 2 / 2.0))
    return A, B, rho, mean_T_phase + mean_S_phase + mean_C_phase


def Heff_theory_N(N: np.ndarray, cfg: SoaringConfig) -> np.ndarray:
    """Eq. 26 evaluated at cycle count ``N`` (returns ``H_eff``, not
    ``2*H_eff``)."""
    A, B, rho, _ = compute_AB(cfg)
    GN = G_N(N, rho)
    GpN = G_N_prime(N, rho)
    s_G = N * GpN / GN
    w = A * GN / (A * GN + B * N)
    return 0.5 * (1.0 + w * (s_G - 1.0))


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
        ax.loglog(lags[m], msd[m], "o", ms=3, color=COLOR_SIM,
                  alpha=0.7, label="simulated")
        grid = np.geomspace(lags[m].min(), lags[m].max(), 200)
        ball_grid = grid[grid <= 2.0 * sm.tau_b_S]
        sub_grid = grid[grid >= 0.5 * sm.tau_b_S]
        ax.loglog(ball_grid, search_msd_short(ball_grid, sm.u_S),
                  "-", lw=1.8, color=COLOR_THY,
                  label=r"Eq. 10 (short): $u_S^2\,\Delta^2$")
        ax.loglog(sub_grid, search_msd_long(sub_grid, sm.alpha_S, sm.tau_b_S,
                                              sm.tau_turn_S, sm.u_S),
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
        ax.loglog(lags[m], msd[m], "o", ms=3, color=COLOR_SIM,
                  alpha=0.7, label="simulated")
        grid = np.geomspace(lags[m].min(), lags[m].max(), 400)
        theory = climb_msd_theory(grid, cm.r0, cm.T_turn_mean,
                                  cm.T_turn_std, cm.v_drift)
        ax.loglog(grid, theory, "-", lw=1.8, color=COLOR_THY,
                  label="Eq. 11 (theory)")
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
    ``n_cycles`` cycles and return ``⟨|X_N|²⟩`` for ``N = 0, …,
    n_cycles`` — the position at the end of each cycle averaged over
    the ensemble. No time-to-cycle conversion is involved: this is the
    direct cycle-counted quantity the closed-form MSD predicts.
    """
    sq = np.zeros(n_cycles + 1, dtype=float)
    for _ in range(n_traj):
        traj = simulate_single(cfg, n_cycles=n_cycles, rng=rng)
        sq += (traj.positions ** 2).sum(axis=1)
    return sq / n_traj


def plot_cycles(configs: dict[str, SoaringConfig],
                cycle_msd: dict[str, np.ndarray], output_path: Path,
                aircrafts: list[str]) -> None:
    fig, axes = _setup_axes(aircrafts)
    for ax, ac, letter in zip(axes, aircrafts, PANEL_LETTERS):
        cfg = configs[ac]
        sq = cycle_msd[ac]
        N_arr = np.arange(len(sq))
        m = N_arr > 0
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
              aircrafts: list[str], win_factor: float = 2.0) -> None:
    """1×N panel of the cycle-counted ``H_eff(N) = (1/2) d log
    ⟨|X_N|²⟩ / d log N``: the local log-log slope of the simulated
    cycle-counted MSD (no time→cycle conversion), compared to Eq. 26
    of the paper evaluated at the same cycle counts.

    The simulated slope is a local fit on a multiplicative window in
    ``log N``: at each N the line is fit over the points falling in
    ``[N / win_factor, N · win_factor]``. The default
    ``win_factor = 2`` gives a roughly half-decade window.
    """
    fig, axes = _setup_axes(aircrafts)
    for ax, ac, letter in zip(axes, aircrafts, PANEL_LETTERS):
        cfg = configs[ac]
        sq = cycle_msd[ac]
        N_arr = np.arange(len(sq), dtype=float)
        N_sim, slope = local_loglog_slope(N_arr, sq, win_factor=win_factor)
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
    parser.add_argument("--n-traj-cycles", type=int, default=3000,
                        help="Trajectories for the cycle-counted MSD.")
    parser.add_argument("--seed-cycles", type=int, default=42,
                        help="RNG seed for the cycle-counted MSD.")
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
        print(
            f"  {ac}: sigma_theta={cfg.angular.sigma_theta:.4f}  "
            f"<T>={mean_T:.1f} s  A={A:.3g}  B={B:.3g}  "
            f"rho={rho:.4f}  n_c={2/cfg.angular.sigma_theta**2:.2f}"
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
    print(
        f"\nCycle-counted MSD ({args.n_traj_cycles} trajectories × "
        f"{args.n_cycles} cycles per aircraft)..."
    )
    cycle_msd: dict[str, np.ndarray] = {}
    for ac in args.aircraft:
        rng = np.random.default_rng(args.seed_cycles
                                     + abs(hash(ac)) % (2 ** 31))
        cycle_msd[ac] = simulate_cycle_msd(
            configs[ac], n_cycles=args.n_cycles,
            n_traj=args.n_traj_cycles, rng=rng,
        )
        print(f"  {ac}: done")
    plot_cycles(configs, cycle_msd,
                args.figures_dir / "compare_theory_msd_cycles.pdf",
                list(args.aircraft))
    plot_Heff(configs, cycle_msd,
              args.figures_dir / "compare_theory_Heff.pdf",
              list(args.aircraft),
              win_factor=args.win_factor)

    print("\nSaved figures:")
    for name in ("msd_search", "msd_climb", "msd_cycles", "Heff"):
        print(f"  {args.figures_dir / f'compare_theory_{name}.pdf'}")


if __name__ == "__main__":
    main()
