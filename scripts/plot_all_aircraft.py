"""Generate the four figures of the paper, reproducing Figs. 1-4 of
Vilpellet, Darmon & Benzaquen (2026).

Outputs
-------
  paper/figures/msd_all_aircraft.pdf   -- Fig. 1-equivalent
  paper/figures/example_trajectory.pdf -- Fig. 2-equivalent
  paper/figures/msd_per_phase.pdf      -- Fig. 3-equivalent
  paper/figures/phase_statistics.pdf   -- Fig. 4-equivalent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soaring_ctrw import SoaringConfig, simulate_ensemble, simulate_single
from soaring_ctrw.observables import fit_hurst, msd_ensemble
from soaring_ctrw.simulation import _interpolate_physical

AIRCRAFT = ["paragliders", "hang_gliders", "sailplanes"]
COLORS = {
    "paragliders": "#d95f02",
    "hang_gliders": "#1f78b4",
    "sailplanes": "#6a3d9a",
}
MARKERS = {"paragliders": "o", "hang_gliders": "*", "sailplanes": "D"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_all(n_trajectories: int, total_time: float, dt: float, seed: int):
    """Simulate an ensemble for each aircraft and return a dict of results.

    Each result contains the ensemble-averaged MSD, its standard error
    (std across trajectories / sqrt(N)), the fitted Hurst exponent and
    the config.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for name in AIRCRAFT:
        cfg = SoaringConfig.from_yaml(f"configs/{name}.yaml")
        print(f"  Simulating {name}: {n_trajectories} trajectories of "
              f"{total_time:.0f} s each...")
        ens = simulate_ensemble(
            config=cfg,
            n_trajectories=n_trajectories,
            total_time=total_time,
            dt=dt,
            rng=rng,
        )
        lags = np.arange(ens.shape[1]) * dt
        # Compute per-trajectory MSD (time-averaged), then ensemble
        # mean and standard error of the mean.
        from soaring_ctrw.observables import msd_time_averaged
        per_traj_msd = np.array(
            [msd_time_averaged(ens[i]) for i in range(ens.shape[0])]
        )
        msd = per_traj_msd.mean(axis=0)
        sem = per_traj_msd.std(axis=0, ddof=1) / np.sqrt(ens.shape[0])
        fit = fit_hurst(lags[1:], msd[1:], lag_range=(10.0, 5000.0))
        results[name] = dict(lags=lags, msd=msd, sem=sem, fit=fit, cfg=cfg)
    return results


def _msd_conditional(xy_segment: np.ndarray, max_lag_steps: int) -> np.ndarray:
    """Time-averaged MSD of a single-phase segment."""
    Ns = len(xy_segment)
    k_max = min(max_lag_steps, Ns - 1)
    if k_max < 1:
        return np.zeros(1)
    msd = np.zeros(k_max + 1)
    for k in range(1, k_max + 1):
        diffs = xy_segment[k:] - xy_segment[:-k]
        msd[k] = np.mean(np.sum(diffs ** 2, axis=1))
    return msd


# ---------------------------------------------------------------------------
# Fig 1: total MSD
# ---------------------------------------------------------------------------


def plot_fig1_msd_all(results: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)

    for name in AIRCRAFT:
        r = results[name]
        lags = r["lags"][1:]
        msd = r["msd"][1:]
        sem = r["sem"][1:]
        # Shaded band for standard error of the mean
        lower = np.maximum(msd - sem, 1e-10)
        upper = msd + sem
        ax.fill_between(lags, lower, upper, color=COLORS[name], alpha=0.25)
        ax.loglog(
            lags, msd,
            marker=MARKERS[name], linestyle="-",
            ms=3, lw=0.4, color=COLORS[name],
            label=rf"{name.replace('_', ' ')} ($H = {r['fit'].hurst:.2f}$)",
        )

    lag_ref = np.geomspace(1.0, 1e4, 100)
    ax.loglog(lag_ref, 1e2 * lag_ref ** 2,
              "k--", lw=0.9, alpha=0.5, label=r"$\Delta^{2}$")
    ax.loglog(lag_ref, 3e2 * lag_ref ** 1.75,
              "k:", lw=0.9, alpha=0.5, label=r"$\Delta^{1.75}$")

    ax.set_xlabel(r"$\Delta$ (s)")
    ax.set_ylabel(r"MSD $\delta^2(\Delta)$ (m$^2$)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e1, 1e10)
    ax.set_xlim(1, 1e4)

    fig.savefig(output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 2: example trajectory with phase-coloured dotted plot
# ---------------------------------------------------------------------------


def plot_fig2_trajectory(output_path: Path) -> None:
    cfg = SoaringConfig.from_yaml("configs/paragliders.yaml")
    rng = np.random.default_rng(7)
    # Simulate 6 cycles for a richer visual (~30 minutes of flight)
    traj = simulate_single(cfg, n_cycles=6, rng=rng)

    # Dense time grid
    dt_fine = 0.5
    total = traj.total_time
    t_grid = np.arange(0.0, total, dt_fine)
    xy = _interpolate_physical(traj, cfg, t_grid)

    # Classify each time-step by phase
    cycle_end_times = traj.cycle_end_times
    tau_T = traj.phase_durations[:, 0]
    tau_S = traj.phase_durations[:, 1]

    phase_labels = np.full(len(t_grid), "T", dtype=object)
    for n in range(traj.n_cycles):
        t_start = cycle_end_times[n]
        t_end = cycle_end_times[n + 1]
        in_cycle = (t_grid >= t_start) & (t_grid < t_end)
        dt_in = t_grid[in_cycle] - t_start
        seg_labels = np.where(dt_in < tau_T[n], "T",
                              np.where(dt_in < tau_T[n] + tau_S[n], "S", "C"))
        phase_labels[in_cycle] = seg_labels

    is_T = phase_labels == "T"
    is_S = phase_labels == "S"
    is_C = phase_labels == "C"

    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    ax.plot(xy[is_T, 0], xy[is_T, 1], ".",
            ms=2.0, color="#1f78b4", label="transition", alpha=0.7)
    ax.plot(xy[is_S, 0], xy[is_S, 1], ".",
            ms=2.0, color="#d62728", label="search", alpha=0.85)
    ax.plot(xy[is_C, 0], xy[is_C, 1], ".",
            ms=2.0, color="#2ca02c", label="climb", alpha=0.85)
    ax.plot(0, 0, "k*", ms=10)
    ax.annotate("$t = 0$", (0, 0), xytext=(30, 30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 3: conditional MSD per phase (one subplot per aircraft)
# ---------------------------------------------------------------------------


def _collect_phase_segments(
    cfg: SoaringConfig,
    n_trajectories: int,
    n_cycles_per_traj: int,
    dt: float,
    rng: np.random.Generator,
    max_seg_len: int = 400,
    max_lag_steps: int = 300,
):
    """For each phase of each cycle, extract the (x,y) path and its MSD."""
    phase_msds = {"T": [], "S": [], "C": []}

    for _ in range(n_trajectories):
        traj = simulate_single(cfg, n_cycles=n_cycles_per_traj, rng=rng)
        for n in range(traj.n_cycles):
            t_start = traj.cycle_end_times[n]
            tau_T = traj.phase_durations[n, 0]
            tau_S = traj.phase_durations[n, 1]
            tau_C = traj.phase_durations[n, 2]

            for phase_label, t0_in, dur in [
                ("T", 0.0, tau_T),
                ("S", tau_T, tau_S),
                ("C", tau_T + tau_S, tau_C),
            ]:
                n_samples = min(max_seg_len, int(dur / dt))
                if n_samples < 2:
                    continue
                local_t = np.linspace(0, dur, n_samples, endpoint=False)
                target_t = t_start + t0_in + local_t
                xy = _interpolate_physical(traj, cfg, target_t)
                msd = _msd_conditional(xy, max_lag_steps=max_lag_steps)
                phase_msds[phase_label].append((local_t, msd))
    return phase_msds


def _average_msd_at_lags(segments: list, target_lags: np.ndarray):
    """Average per-segment MSDs onto a common logarithmic lag grid.

    Returns
    -------
    mean : ndarray of shape (N,)
        Mean MSD at each target lag (NaN where no segment reaches it).
    sem : ndarray of shape (N,)
        Standard error of the mean (NaN where no segment reaches it).
    count : ndarray of shape (N,)
        Number of segments contributing to each lag.
    """
    per_lag_values = [[] for _ in range(len(target_lags))]
    for local_t, msd in segments:
        k_max = len(msd) - 1
        if k_max < 1:
            continue
        dt_seg = local_t[1] - local_t[0] if len(local_t) > 1 else 1.0
        seg_lags = np.arange(1, k_max + 1) * dt_seg
        seg_msd = msd[1:k_max + 1]
        mask = (target_lags >= seg_lags[0]) & (target_lags <= seg_lags[-1])
        if not np.any(mask):
            continue
        interp = np.interp(target_lags[mask], seg_lags, seg_msd)
        idx_mask = np.where(mask)[0]
        for j, v in zip(idx_mask, interp):
            per_lag_values[j].append(v)

    n_lags = len(target_lags)
    mean = np.full(n_lags, np.nan)
    sem = np.full(n_lags, np.nan)
    count = np.zeros(n_lags, dtype=int)
    for j in range(n_lags):
        if len(per_lag_values[j]) > 1:
            arr = np.array(per_lag_values[j])
            mean[j] = arr.mean()
            sem[j] = arr.std(ddof=1) / np.sqrt(len(arr))
            count[j] = len(arr)
        elif len(per_lag_values[j]) == 1:
            mean[j] = per_lag_values[j][0]
            sem[j] = 0.0
            count[j] = 1
    return mean, sem, count


def _power_law_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of log y = a*log x + b, returning (slope, intercept).

    Both arrays must be strictly positive and mask out non-finite values
    upstream. Intercept is in log space, so the power-law prefactor is
    exp(intercept).
    """
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
    return slope, intercept


def plot_fig3_msd_per_phase(output_path: Path, n_trajectories: int,
                             n_cycles_per_traj: int, dt: float,
                             seed: int) -> None:
    """Per-phase conditional MSD with fit lines overlaid in each regime.

    Like Fig. 3 of Vilpellet et al., we overlay power-law reference
    lines *only in the asymptotic region* where each phase shows its
    characteristic scaling, so that the visual comparison is direct:

    - transition: slope-2 ballistic line over its full range;
    - search: fit in the sub-diffusive plateau Δ > 10 s, slope expected
      near α_S;
    - climb: short-Δ ballistic tangent (before the first oscillation),
      and long-Δ drift-dominated slope (slope 2, drift-tail).

    Error bars are the standard error of the per-segment mean (std /
    sqrt(count)) shown as shaded regions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    phase_colors = {"T": "#1f78b4", "S": "#d62728", "C": "#2ca02c"}
    phase_names = {"T": "transition", "S": "search", "C": "climb"}
    panel_labels = {"paragliders": "(a) paragliders",
                    "hang_gliders": "(b) hang gliders",
                    "sailplanes":   "(c) sailplanes"}

    target_lags = np.geomspace(1.0, 300.0, 50)
    rng = np.random.default_rng(seed)

    for ax, name in zip(axes, AIRCRAFT):
        print(f"  Phase MSD for {name}...")
        cfg = SoaringConfig.from_yaml(f"configs/{name}.yaml")
        alpha_S = cfg.search_motion.alpha_S if cfg.search_motion else 0.6

        phase_msds = _collect_phase_segments(
            cfg,
            n_trajectories=n_trajectories,
            n_cycles_per_traj=n_cycles_per_traj,
            dt=dt,
            rng=rng,
            max_seg_len=600,
            max_lag_steps=400,
        )

        phase_results = {}
        for ph in ["T", "S", "C"]:
            mean, sem, count = _average_msd_at_lags(phase_msds[ph], target_lags)
            phase_results[ph] = (mean, sem, count)
            valid = ~np.isnan(mean)
            # shaded SEM band
            lower = np.maximum(mean[valid] - sem[valid], 1e-6)
            upper = mean[valid] + sem[valid]
            ax.fill_between(target_lags[valid], lower, upper,
                            color=phase_colors[ph], alpha=0.22)
            ax.loglog(target_lags[valid], mean[valid],
                      marker="o", ms=3, lw=0.5,
                      color=phase_colors[ph],
                      label=phase_names[ph] if ax is axes[0] else None)

        # Overlay regime fits
        mean_T, _, _ = phase_results["T"]
        mean_S, _, _ = phase_results["S"]
        mean_C, _, _ = phase_results["C"]
        
        slope_T = slope_S = slope_Cs = None

        # (i) Transition: slope 2 line through the middle of its range
        fit_window_T = (target_lags >= 1) & (target_lags <= 300) & ~np.isnan(mean_T)
        if fit_window_T.sum() >= 2:
            slope_T, inter_T = _power_law_fit(target_lags[fit_window_T], mean_T[fit_window_T])
            lag_line = np.array([target_lags[fit_window_T][0], target_lags[fit_window_T][-1]])
            y_line = np.exp(inter_T) * lag_line ** slope_T
            # Offset upward for visibility
            ax.loglog(lag_line, y_line * 2.5, color="black", lw=1.3, ls="-",
                      label=rf"$\sim\Delta^{{{slope_T:.2f}}}$ (T)" if ax is axes[0] else None)

        # (ii) Search: sub-diffusive regime of the subordinated Lévy walk.
        # We fit in [5, 50] s where the curve is well-resolved and shows
        # the approach to the asymptotic Δ^α_S scaling, before
        # finite-phase statistical saturation at Δ > 100 s.
        fit_window_S = (target_lags >= 10) & (target_lags <= 40) & ~np.isnan(mean_S)
        if fit_window_S.sum() >= 2:
            slope_S, inter_S = _power_law_fit(target_lags[fit_window_S], mean_S[fit_window_S])
            lag_line = np.array([target_lags[fit_window_S][0], target_lags[fit_window_S][-1]])
            y_line = np.exp(inter_S) * lag_line ** slope_S
            ax.loglog(lag_line, y_line * 2.5, color="black", lw=1.3, ls="--",
                      label=rf"$\sim\Delta^{{{slope_S:.2f}}}$ (S)" if ax is axes[0] else None)

        # (iii) Climb: long-Δ drift-dominated slope (Δ > 50 s)
        fit_window_C_short = (target_lags >= 100) & (target_lags <= 400) & ~np.isnan(mean_C)
        if fit_window_C_short.sum() >= 2:
            slope_Cs, inter_Cs = _power_law_fit(target_lags[fit_window_C_short],
                                                 mean_C[fit_window_C_short])
            lag_line = np.array([target_lags[fit_window_C_short][0], target_lags[fit_window_C_short][-1]])
            y_line = np.exp(inter_Cs) * lag_line ** slope_Cs
            ax.loglog(lag_line, y_line * 2.5, color="black", lw=1.3, ls=":",
                      label=rf"$\sim\Delta^{{{slope_Cs:.2f}}}$ (C short)"
                            if ax is axes[0] else None)

        # Build title with fit values
        title = panel_labels[name]
        if slope_T is not None and slope_S is not None and slope_Cs is not None:
            title += f"\n$\\alpha_T={slope_T:.2f}$, $\\alpha_S={slope_S:.2f}$, $\\alpha_C={slope_Cs:.2f}$"
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel(r"$\Delta$ (s)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(1, 400)
        ax.set_ylim(1e1, 1e8)

    axes[0].set_ylabel(r"MSD (m$^2$)")
    axes[0].legend(loc="upper left", fontsize=7.5, ncol=1,
                   handlelength=1.8)

    fig.savefig(output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 6: zoom on a single cycle (T + S + C)
# ---------------------------------------------------------------------------


def plot_fig6_single_cycle(output_path: Path) -> None:
    """Plot one complete cycle (transition + search + climb) of a
    paraglider, with high temporal resolution, so the three phase
    morphologies are visible on their intrinsic scales.

    Because the transition step (~1.5--2 km) dwarfs the search cloud
    (~tens of m) and climb loops (~40 m), we add a zoomed inset on the
    end of the transition / search / climb region so that all three
    phases are legible on their native scales."""
    cfg = SoaringConfig.from_yaml("configs/paragliders.yaml")
    # Loop over seeds until we find a cycle with a visually clear search
    # cloud (tau^S long enough to host several ML jumps) and a distinct
    # climb (tau^C long enough to host several thermalling loops).
    found = False
    for seed in range(1, 5000):
        rng = np.random.default_rng(seed)
        traj = simulate_single(cfg, n_cycles=1, rng=rng)
        tau_T = traj.phase_durations[0, 0]
        tau_S = traj.phase_durations[0, 1]
        tau_C = traj.phase_durations[0, 2]
        if 80 < tau_T < 300 and 50 < tau_S < 250 and 90 < tau_C < 250:
            # Also require enough search jumps for visibility
            waits, lens, _ = traj.search_legs[0][1], traj.search_legs[0][0], traj.search_legs[0][2]
            # Count ballistic legs with leg_length = v_c_S * leg_duration >= 5 m
            leg_lengths = cfg.search_motion.v_c_S * traj.search_legs[0][0]
            n_jumps = int((leg_lengths > 5).sum())  # legs above 5m
            if n_jumps >= 8:
                found = True
                break
    if not found:
        # Fallback: take the last sampled cycle
        pass

    dt_fine = 0.5
    t_grid = np.arange(0.0, traj.total_time, dt_fine)
    xy = _interpolate_physical(traj, cfg, t_grid)

    phase_labels = np.full(len(t_grid), "T", dtype=object)
    t_end_T = tau_T
    t_end_S = tau_T + tau_S
    phase_labels[t_grid >= t_end_T] = "S"
    phase_labels[t_grid >= t_end_S] = "C"

    is_T = phase_labels == "T"
    is_S = phase_labels == "S"
    is_C = phase_labels == "C"

    x_end_T, y_end_T = xy[np.argmin(np.abs(t_grid - t_end_T))]
    x_end_S, y_end_S = xy[np.argmin(np.abs(t_grid - t_end_S))]

    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=(11.5, 5.5),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
        constrained_layout=True,
    )

    # --- Main panel: whole cycle ---
    ax_main.plot(xy[is_T, 0], xy[is_T, 1], "-", lw=1.6,
                 color="#1f78b4",
                 label=f"transition ($\\tau^T$={tau_T:.0f} s)",
                 alpha=0.85)
    ax_main.plot(xy[is_S, 0], xy[is_S, 1], ".", ms=3.5,
                 color="#d62728",
                 label=f"search ($\\tau^S$={tau_S:.0f} s)",
                 alpha=0.85)
    ax_main.plot(xy[is_C, 0], xy[is_C, 1], "-", lw=1.0,
                 color="#2ca02c",
                 label=f"climb ($\\tau^C$={tau_C:.0f} s)",
                 alpha=0.85)

    ax_main.plot(0, 0, "k*", ms=12, zorder=5)
    ax_main.annotate(r"$t=0$", (0, 0), xytext=(15, 15),
                     textcoords="offset points", fontsize=9)

    ax_main.set_xlabel(r"$x$ (m)")
    ax_main.set_ylabel(r"$y$ (m)")
    ax_main.set_aspect("equal")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(fontsize=9, loc="best")
    ax_main.set_title("(a) whole cycle", fontsize=10, loc="left")

    # Draw zoom box on main panel
    zoom_pad = max(6 * cfg.climb_motion.thermal_radius, 80.0)
    xc = 0.5 * (x_end_T + x_end_S)
    yc = 0.5 * (y_end_T + y_end_S)
    # include all S+C points in the zoom
    xy_SC = xy[is_S | is_C]
    x_lo = xy_SC[:, 0].min() - 20
    x_hi = xy_SC[:, 0].max() + 20
    y_lo = xy_SC[:, 1].min() - 20
    y_hi = xy_SC[:, 1].max() + 20
    # Square out the box
    dx = x_hi - x_lo
    dy = y_hi - y_lo
    if dx > dy:
        pad = (dx - dy) / 2
        y_lo -= pad
        y_hi += pad
    else:
        pad = (dy - dx) / 2
        x_lo -= pad
        x_hi += pad

    from matplotlib.patches import Rectangle
    rect = Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                     fill=False, edgecolor="gray", lw=1.2, ls="--")
    ax_main.add_patch(rect)

    # --- Zoom panel ---
    ax_zoom.plot(xy[is_T, 0], xy[is_T, 1], "-", lw=1.6,
                 color="#1f78b4", alpha=0.85)
    ax_zoom.plot(xy[is_S, 0], xy[is_S, 1], ".", ms=5.0,
                 color="#d62728", alpha=0.9)
    ax_zoom.plot(xy[is_C, 0], xy[is_C, 1], "-", lw=1.2,
                 color="#2ca02c", alpha=0.9)

    ax_zoom.plot(x_end_T, y_end_T, "o", mfc="white", mec="black",
                 ms=10, mew=1.2, zorder=5)
    ax_zoom.annotate("T $\\to$ S", (x_end_T, y_end_T), xytext=(10, 10),
                     textcoords="offset points", fontsize=9)
    ax_zoom.plot(x_end_S, y_end_S, "s", mfc="white", mec="black",
                 ms=10, mew=1.2, zorder=5)
    ax_zoom.annotate("S $\\to$ C", (x_end_S, y_end_S), xytext=(10, -15),
                     textcoords="offset points", fontsize=9)

    ax_zoom.set_xlim(x_lo, x_hi)
    ax_zoom.set_ylim(y_lo, y_hi)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel(r"$x$ (m)")
    ax_zoom.set_ylabel(r"$y$ (m)")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(
        f"(b) zoom on search + climb "
        f"($T_{{\\mathrm{{turn}}}}$={traj.climb_turn_periods[0]:.1f} s, "
        f"$r_0$={cfg.climb_motion.thermal_radius:.0f} m)",
        fontsize=10, loc="left",
    )

    fig.savefig(output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 5: MSD rescaled by v_xy^2 (universal-collapse plot)
# ---------------------------------------------------------------------------


def plot_fig5_rescaled(results: dict, output_path: Path) -> None:
    """Reproduce the inset of Vilpellet et al. Fig. 1: MSD divided by
    the squared mean instantaneous horizontal velocity v_xy^2, showing
    the near-perfect universal collapse of the three aircraft curves
    onto a single scaling function of the time lag Δ."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)

    for name in AIRCRAFT:
        r = results[name]
        v_xy = r["cfg"].v_xy
        rescaled_msd = r["msd"][1:] / v_xy ** 2
        ax.loglog(
            r["lags"][1:], rescaled_msd,
            marker=MARKERS[name], linestyle="-",
            ms=3, lw=0.4, color=COLORS[name],
            label=f"{name.replace('_', ' ')}  "
                  rf"($v_{{xy}} = {v_xy:.2f}$ m/s)",
        )

    lag_ref = np.geomspace(1.0, 1e4, 100)
    ax.loglog(lag_ref, 1.0 * lag_ref ** 2,
              "k--", lw=0.9, alpha=0.5, label=r"$\Delta^{2}$")
    ax.loglog(lag_ref, 3.0 * lag_ref ** 1.75,
              "k:", lw=0.9, alpha=0.5, label=r"$\Delta^{1.75}$")

    ax.set_xlabel(r"$\Delta$ (s)")
    ax.set_ylabel(r"$\delta^2(\Delta) / v_{xy}^{2}$  (s$^2$)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, 1e4)

    fig.savefig(output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Fig 4: phase statistics (survival functions + velocity)
# ---------------------------------------------------------------------------


def plot_fig4_phase_statistics(output_path: Path,
                                n_trajectories: int,
                                n_cycles_per_traj: int,
                                seed: int) -> None:
    rng = np.random.default_rng(seed)

    duration_data = {p: {} for p in ["T", "S", "C"]}
    for name in AIRCRAFT:
        cfg = SoaringConfig.from_yaml(f"configs/{name}.yaml")
        durs_T, durs_S, durs_C = [], [], []
        for _ in range(n_trajectories):
            traj = simulate_single(cfg, n_cycles=n_cycles_per_traj, rng=rng)
            durs_T.append(traj.phase_durations[:, 0])
            durs_S.append(traj.phase_durations[:, 1])
            durs_C.append(traj.phase_durations[:, 2])
        duration_data["T"][name] = np.concatenate(durs_T)
        duration_data["S"][name] = np.concatenate(durs_S)
        duration_data["C"][name] = np.concatenate(durs_C)

    # Three panels: survival functions of tau_T, tau_S, tau_C (no velocity bar
    # chart, which was redundant with Table 1).
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8),
                             constrained_layout=True)

    panel_info = [
        ("T", r"$P(\tau_\mathrm{T} > \tau)$", "(a) transition times"),
        ("S", r"$P(\tau_\mathrm{S} > \tau)$", "(b) search times"),
        ("C", r"$P(\tau_\mathrm{C} > \tau)$", "(c) climb times"),
    ]
    # A fixed lower bound on tau makes the small-tau plateau (surv = 1)
    # visible on every panel, matching the layout of Fig. 4 of [Vilpellet et
    # al. 2026] where the curves are clearly flat at unity for tau below the
    # distribution scale before the heavy-tailed (T, S) or exponential (C)
    # decay sets in.
    TAU_FLOOR = 1e-1  # s
    for ax_i, (phase, ylabel, title) in enumerate(panel_info):
        ax = axes[ax_i]
        for name in AIRCRAFT:
            d = duration_data[phase][name]
            d_sorted = np.sort(d)
            tau_lo = TAU_FLOOR
            tau_hi = d_sorted[-1]
            # 120 log-spaced tau values, of which a non-trivial fraction lie
            # below min(d_sorted) so that the plateau at surv=1 is resolved.
            tau_eval = np.geomspace(tau_lo, tau_hi, 120)
            ranks = np.searchsorted(d_sorted, tau_eval, side="right")
            surv_eval = 1.0 - ranks / len(d_sorted)
            mask = surv_eval > 0  # drop the saturated last point
            ax.loglog(tau_eval[mask], surv_eval[mask],
                      marker=MARKERS[name], ms=3, lw=0,
                      color=COLORS[name],
                      label=name.replace("_", " "))
        ax.set_xlabel(r"$\tau$ (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(TAU_FLOOR, None)
        ax.set_ylim(1e-4, 1.5)
        if ax_i == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.savefig(output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--total-time", type=float, default=10_000.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("paper/figures"))
    parser.add_argument("--skip", default="",
                        help="comma-separated list of figures to skip")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    results_fig1 = None
    if "fig1" not in skip:
        print("[Fig 1] Simulating full ensemble for total MSD...")
        results_fig1 = _simulate_all(args.n_trajectories, args.total_time, args.dt,
                                     args.seed)
        plot_fig1_msd_all(results_fig1, args.output_dir / "msd_all_aircraft.pdf")

        print("\n  H values:")
        for name in AIRCRAFT:
            r = results_fig1[name]
            print(f"    {name:15s}  H = {r['fit'].hurst:.3f}  "
                  f"MSD(100s) = {r['msd'][100]:.2e} m^2")

    if "fig2" not in skip:
        print("\n[Fig 2] Example trajectory...")
        plot_fig2_trajectory(args.output_dir / "example_trajectory.pdf")

    if "fig3" not in skip:
        print("\n[Fig 3] Conditional MSD per phase "
              "(500 trajectories * 30 cycles each: heavy run)...")
        plot_fig3_msd_per_phase(
            args.output_dir / "msd_per_phase.pdf",
            n_trajectories=500, n_cycles_per_traj=30, dt=0.5, seed=args.seed + 1,
        )

    if "fig4" not in skip:
        print("\n[Fig 4] Single-cycle zoom (transition + search + climb)...")
        plot_fig6_single_cycle(args.output_dir / "single_cycle.pdf")

    if "fig5" not in skip:
        print("\n[Fig 5] Rescaled MSD for universal collapse...")
        # Reuse results from Fig 1 if available; otherwise re-simulate
        if results_fig1 is None:
            results_fig1 = _simulate_all(args.n_trajectories, args.total_time,
                                          args.dt, args.seed)
        plot_fig5_rescaled(results_fig1, args.output_dir / "msd_rescaled.pdf")

    if "fig7" not in skip:
        print("\n[Fig 7] Phase statistics "
              "(v_xy bar + survival of tau_T, tau_S, tau_C)...")
        plot_fig4_phase_statistics(
            args.output_dir / "phase_statistics.pdf",
            n_trajectories=args.n_trajectories,
            n_cycles_per_traj=30,
            seed=args.seed + 2,
        )


if __name__ == "__main__":
    main()
