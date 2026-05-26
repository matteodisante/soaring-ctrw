"""Per-phase ensemble MSD plots.

For each aircraft the script computes the pure ensemble-averaged MSD
of the horizontal motion *restricted to each phase* (transition,
search, climb). Each phase episode contributes a single pair per lag:
``|r(k·dt) - r(0)|^2`` where ``r(0)`` is the position at the start of
the episode. The MSD at lag ``k`` is the mean of this quantity across
all episodes whose physical duration covers ``k·dt``. No
time-averaging inside an episode is performed.

A log-log linear fit is overlaid on each curve (fit window per phase
selectable via CLI) and the fitted exponent is annotated per panel.

Cache policy: identical to the rest of the repository (see
``cache.py``). The computation slot is per aircraft.

Outputs (under ``outputs/``):
    data/per_phase_msd/{aircraft}.npz
    data/per_phase_msd/{aircraft}.json
    figures/per_phase_msd_combined.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import logging
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
    load_dataset,
    save_dataset,
    slot_paths,
)
from soaring_ctrw.model import SoaringConfig
from soaring_ctrw.paths import CONFIGS_DIR, DATA_DIR, FIGURES_DIR, REPO_ROOT
from soaring_ctrw.simulation import (
    CycleTrajectory,
    _climb_position,
    _search_position,
    simulate_single,
)

SCRIPT_SLUG = "per_phase_msd"
PHASES = ("T", "S", "C")
PHASE_LABELS = {"T": "transition", "S": "search", "C": "climb"}
PHASE_COLORS = {"T": "#1f3bd0", "S": "#d6243a", "C": "#207a3e"}
AIRCRAFT_LABELS = {
    "paragliders": "paragliders",
    "hang_gliders": "hang gliders",
    "sailplanes": "sailplanes",
}
PANEL_LETTERS = ("a", "b", "c")


def _seed_from(*parts) -> int:
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:4], "big")


# ---------------------------------------------------------------------------
# Per-episode sampling
# ---------------------------------------------------------------------------


def _phase_positions(
    traj: CycleTrajectory,
    config: SoaringConfig,
    cycle: int,
    phase: str,
    dt: float,
    max_samples: int | None = None,
) -> np.ndarray:
    """Return positions sampled at ``dt`` within one phase episode.

    Positions are expressed *relative to the start of the episode*, so
    the array starts at the origin and only the within-phase motion
    contributes. Shape ``(n_samples, 2)`` with ``n_samples >= 1``.

    ``max_samples`` caps the number of samples per episode so the
    memory cost stays bounded for occasional long Lomax draws; only
    the first ``max_samples·dt`` seconds of the episode are sampled.
    The pooled MSD at lag Δ is unaffected as long as
    ``max_samples ≥ n_lags + safety``.
    """
    if phase == "T":
        tau_T = float(traj.phase_durations_active[cycle, 0])
        n = max(1, int(np.floor(tau_T / dt)) + 1)
        if max_samples is not None:
            n = min(n, max_samples)
        t_max = (n - 1) * dt
        t = np.minimum(np.arange(n) * dt, min(tau_T, t_max))
        theta = float(traj.headings[cycle])
        x = config.v_xy * t * np.cos(theta)
        y = config.v_xy * t * np.sin(theta)
        return np.column_stack([x, y])

    if phase == "S":
        ep = traj.search_episodes[cycle]
        T_phys = float(ep.T_phys)
        if T_phys <= 0 or config.search_motion is None:
            return np.zeros((1, 2))
        n = max(1, int(np.floor(T_phys / dt)) + 1)
        if max_samples is not None:
            n = min(n, max_samples)
        t_max = (n - 1) * dt
        t = np.minimum(np.arange(n) * dt, min(T_phys, t_max))
        return _search_position(t, ep, config.search_motion.u_S)

    if phase == "C":
        tau_C = float(traj.phase_durations_active[cycle, 2])
        if tau_C <= 0 or config.climb_motion is None:
            return np.zeros((1, 2))
        n = max(1, int(np.floor(tau_C / dt)) + 1)
        if max_samples is not None:
            n = min(n, max_samples)
        t_max = (n - 1) * dt
        t = np.minimum(np.arange(n) * dt, min(tau_C, t_max))
        omega = 2.0 * np.pi / float(traj.climb_turn_period[cycle])
        return _climb_position(
            t=t,
            r0=config.climb_motion.r0,
            omega=omega,
            phi0=float(traj.climb_initial_phase[cycle]),
            v_drift=config.climb_motion.v_drift,
            phi_drift=float(traj.climb_drift_angle[cycle]),
        )

    raise ValueError(f"unknown phase {phase!r}")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_per_phase_msd(
    config: SoaringConfig,
    n_trajectories: int,
    n_cycles: int,
    dt: float,
    n_lags: int,
    max_samples_per_episode: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    """Compute pure ensemble-averaged MSD per phase.

    For each phase episode the displacement ``r(k·dt) - r(0)`` is
    evaluated for every lag ``k`` that the episode covers, and pooled
    across all episodes of the same phase (one sample per episode per
    lag). The MSD at lag ``k`` is then the mean over the subset of
    episodes whose physical duration reaches ``k·dt``.

    Returns
    -------
    dict with keys ``lags``, ``msd_T``, ``msd_S``, ``msd_C``,
    ``count_T``, ``count_S``, ``count_C`` (= number of episodes
    contributing at each lag), ``n_episodes_T`` etc.
    """
    lags = np.arange(n_lags) * dt
    sq_sum = {p: np.zeros(n_lags, dtype=float) for p in PHASES}
    count = {p: np.zeros(n_lags, dtype=float) for p in PHASES}
    n_episodes = {p: 0 for p in PHASES}

    t0 = time.time()
    for i in range(n_trajectories):
        traj = simulate_single(config, n_cycles=n_cycles, rng=rng)
        for n in range(traj.n_cycles):
            for phase in PHASES:
                pos = _phase_positions(
                    traj, config, n, phase, dt,
                    max_samples=max_samples_per_episode,
                )
                m = len(pos)
                if m < 2:
                    continue
                k_max = min(m, n_lags)
                # EA pooling: one sample per episode per lag.
                # pos[0] is the episode origin (see _phase_positions),
                # so |pos[k] - pos[0]|^2 = |pos[k]|^2 only if origin
                # truly is zero — compute the difference explicitly to
                # be robust to any future change in _phase_positions.
                disp = pos[:k_max] - pos[0]
                sq_sum[phase][:k_max] += np.sum(disp * disp, axis=1)
                count[phase][:k_max] += 1.0
                n_episodes[phase] += 1

        if (i + 1) % max(1, n_trajectories // 10) == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1:>5}/{n_trajectories}] elapsed {elapsed:.1f}s"
            )
            logger.info("traj %d/%d elapsed=%.1fs", i + 1, n_trajectories, elapsed)

    msd = {p: sq_sum[p] / np.where(count[p] > 0, count[p], np.nan) for p in PHASES}
    return {
        "lags": lags,
        "msd_T": msd["T"], "msd_S": msd["S"], "msd_C": msd["C"],
        "count_T": count["T"], "count_S": count["S"], "count_C": count["C"],
        "n_episodes_T": np.array([n_episodes["T"]]),
        "n_episodes_S": np.array([n_episodes["S"]]),
        "n_episodes_C": np.array([n_episodes["C"]]),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_single(
    aircraft: str,
    data: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    lags = data["lags"]
    for p in PHASES:
        msd = data[f"msd_{p}"]
        m = (lags > 0) & np.isfinite(msd) & (msd > 0)
        ax.loglog(
            lags[m], msd[m],
            linestyle="none", marker="o", ms=4,
            color=PHASE_COLORS[p], label=PHASE_LABELS[p],
        )
    ax.set_xlabel(r"$\Delta$ (s)")
    ax.set_ylabel(r"MSD (m$^2$)")
    ax.set_title(f"Per-phase MSD — {AIRCRAFT_LABELS[aircraft]}", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, ls=":", alpha=0.4, which="both")
    fig.savefig(output_path)
    plt.close(fig)


def _fit_log_log(
    lags: np.ndarray,
    msd: np.ndarray,
    fit_min: float,
    fit_max: float,
) -> tuple[float, float] | None:
    """Linear fit of log(MSD) vs log(lag) over ``[fit_min, fit_max]``.

    Returns ``(slope, intercept)`` or ``None`` if fewer than 2 valid
    points are in range.
    """
    mask = (
        (lags >= fit_min) & (lags <= fit_max)
        & np.isfinite(msd) & (msd > 0)
    )
    if mask.sum() < 2:
        return None
    slope, intercept = np.polyfit(np.log(lags[mask]), np.log(msd[mask]), 1)
    return float(slope), float(intercept)


def _plot_combined(
    datasets: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    *,
    fit_ranges: dict[str, tuple[float, float]],
    fit_offset: float = 3.0,
) -> None:
    """Three-panel figure mirroring Fig. 3 of the empirical reference.

    Each phase MSD is overlaid with a log-log linear fit over the lag
    window specified in ``fit_ranges`` (one ``(fit_min, fit_max)`` per
    phase key in ``PHASES``). The fit line is multiplied by
    ``fit_offset`` so it sits *above* the data points (visual aid) and
    the fitted exponent is reported in a per-panel annotation — each
    aircraft has its own three exponents.
    """
    aircraft_order = [a for a in ("paragliders", "hang_gliders", "sailplanes") if a in datasets]
    n = len(aircraft_order)
    fig, axes = plt.subplots(
        1, n, figsize=(4.5 * n, 4.0), constrained_layout=True, sharey=True,
    )
    if n == 1:
        axes = [axes]

    for ax, aircraft, letter in zip(axes, aircraft_order, PANEL_LETTERS):
        data = datasets[aircraft]
        lags = data["lags"]
        slope_text: list[str] = []
        for p in PHASES:
            msd = data[f"msd_{p}"]
            m = (lags > 0) & np.isfinite(msd) & (msd > 0)
            base_label = PHASE_LABELS[p]

            # Marker series: keep the legend uniform across panels
            # (only the leftmost panel populates the shared bottom
            # legend); the per-aircraft exponents go into an inset
            # text annotation instead.
            ax.loglog(
                lags[m], msd[m],
                linestyle="none", marker="o", ms=4,
                color=PHASE_COLORS[p],
                label=base_label if ax is axes[0] else None,
            )

            fit_min, fit_max = fit_ranges[p]
            fit = _fit_log_log(lags, msd, fit_min, fit_max)
            if fit is not None:
                slope, intercept = fit
                grid = np.geomspace(fit_min, fit_max, 60)
                ax.loglog(
                    grid, fit_offset * np.exp(intercept) * grid ** slope,
                    "--", color=PHASE_COLORS[p], lw=1.5, alpha=0.95,
                )
                slope_text.append(rf"{base_label}: $\alpha={slope:.3f}$")
            else:
                slope_text.append(rf"{base_label}: $\alpha=$N/A")

        ax.set_xlabel(r"$\Delta$ (s)")
        ax.set_title(f"({letter}) {AIRCRAFT_LABELS[aircraft]}", fontsize=11)
        ax.grid(True, ls=":", alpha=0.4, which="both")
        # Per-panel inset: the three fitted exponents for this aircraft.
        ax.text(
            0.03, 0.97, "\n".join(slope_text),
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white", edgecolor="0.6", alpha=0.85,
            ),
        )

    axes[0].set_ylabel(r"MSD (m$^2$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(PHASES),
        frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--aircraft", nargs="+",
        default=["paragliders", "hang_gliders", "sailplanes"],
        choices=["paragliders", "hang_gliders", "sailplanes"],
    )
    parser.add_argument("--n-trajectories", type=int, default=1000)
    parser.add_argument(
        "--n-cycles", type=int, default=50,
        help="Cycles per trajectory. Each cycle yields one T, one S and "
             "one C episode.",
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--max-lag", type=float, default=500.0,
        help="Maximum lag in seconds reported on the plot.",
    )
    parser.add_argument(
        "--max-samples-per-episode", type=int, default=None,
        help="Cap samples per phase episode. Default: 4·n_lags. Needed "
             "to keep memory bounded when Mittag-Leffler turning waits "
             "(α<1) make a single search episode extremely long.",
    )
    parser.add_argument(
        "--fit-min-T", type=float, default=None,
        help="Lower edge of the log-log fit on the transition MSD (s). "
             "Default: dt (full span).",
    )
    parser.add_argument(
        "--fit-min-S", type=float, default=None,
        help="Lower edge of the log-log fit on the search MSD (s). "
             "Default: 100 s (asymptotic alpha regime, up to max_lag).",
    )
    parser.add_argument(
        "--fit-min-C", type=float, default=None,
        help="Lower edge of the log-log fit on the climb MSD (s). "
             "Default: 100 s (ballistic drift regime, up to max_lag).",
    )
    parser.add_argument(
        "--fit-offset", type=float, default=3.0,
        help="Multiplicative offset applied to the fit line so it sits "
             "above the MSD points (visual aid). Use 1.0 to put the "
             "fit on the data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    add_cache_args(parser)
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.logs_dir / f"{SCRIPT_SLUG}_{int(time.time())}.log"
    logger = logging.getLogger(SCRIPT_SLUG)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    print(f"Logging to: {log_path}")
    logger.info("start run; args=%s", vars(args))

    n_lags = int(np.ceil(args.max_lag / args.dt)) + 1
    max_samples = args.max_samples_per_episode
    if max_samples is None:
        max_samples = max(4 * n_lags, 1000)
    print(
        f"Per-phase MSD scan.\n"
        f"Aircraft: {', '.join(args.aircraft)}.\n"
        f"{args.n_trajectories} trajectories × {args.n_cycles} cycles each, "
        f"dt={args.dt}s, max_lag={args.max_lag:g}s ({n_lags} lag points).\n"
        f"Per-episode sample cap: {max_samples}.\n"
    )

    datasets: dict[str, dict[str, np.ndarray]] = {}

    for aircraft in args.aircraft:
        print(f"=== {aircraft} ===")
        logger.info("aircraft=%s", aircraft)
        base = SoaringConfig.from_yaml(CONFIGS_DIR / f"{aircraft}.yaml")

        rng = np.random.default_rng(_seed_from(args.seed, aircraft, SCRIPT_SLUG))

        npz_path, manifest_path = slot_paths(args.data_dir, SCRIPT_SLUG, aircraft)
        requested = build_manifest(
            script=SCRIPT_SLUG,
            params={
                "n_trajectories": args.n_trajectories,
                "n_cycles": args.n_cycles,
                "dt": args.dt,
                "max_lag": args.max_lag,
                "max_samples_per_episode": max_samples,
                "seed": args.seed,
                "msd_estimator": "ea",
            },
            config_paths={"aircraft": CONFIGS_DIR / f"{aircraft}.yaml"},
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
        logger.info("decision=%s (%s)", decision.action, decision.reason)

        if decision.action == "reuse":
            arrays, _ = load_dataset(npz_path, manifest_path)
            data = dict(arrays)
        else:
            data = _compute_per_phase_msd(
                config=base,
                n_trajectories=args.n_trajectories,
                n_cycles=args.n_cycles,
                dt=args.dt,
                n_lags=n_lags,
                max_samples_per_episode=max_samples,
                rng=rng,
                logger=logger,
            )
            if args.cache != "off":
                save_dataset(
                    npz_path=npz_path,
                    manifest_path=manifest_path,
                    manifest=requested,
                    arrays=data,
                )
                print(f"  saved data: {npz_path}")
                logger.info("saved data: %s", npz_path)

        datasets[aircraft] = data

    # Fit ranges per phase. Transition spans the whole range; search
    # and climb are fit from 100 s up to ``max_lag``, where the curves
    # are genuinely linear on log-log (asymptotic α-regime for search,
    # ballistic drift regime for climb).
    fit_min_T = args.fit_min_T if args.fit_min_T is not None else args.dt
    tail_default = 100.0
    fit_min_S = args.fit_min_S if args.fit_min_S is not None else tail_default
    fit_min_C = args.fit_min_C if args.fit_min_C is not None else tail_default
    fit_ranges = {
        "T": (fit_min_T, args.max_lag),
        "S": (fit_min_S, args.max_lag),
        "C": (fit_min_C, args.max_lag),
    }
    print(
        "Power-law fit windows (s):\n"
        f"  transition: [{fit_min_T:g}, {args.max_lag:g}]\n"
        f"  search:     [{fit_min_S:g}, {args.max_lag:g}]\n"
        f"  climb:      [{fit_min_C:g}, {args.max_lag:g}]"
    )

    if len(datasets) >= 1:
        combined_path = args.figures_dir / f"{SCRIPT_SLUG}_combined.pdf"
        _plot_combined(
            datasets=datasets,
            output_path=combined_path,
            fit_ranges=fit_ranges,
            fit_offset=args.fit_offset,
        )
        print(f"\nCombined figure saved: {combined_path}")
        logger.info("saved combined: %s", combined_path)

    logger.info("end run")


if __name__ == "__main__":
    main()
