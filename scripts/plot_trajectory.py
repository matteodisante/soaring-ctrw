"""Sample-trajectory plot for the cycle-based CTRW model.

Produces a single combined figure with one panel per aircraft (1×N
layout): each panel shows one representative trajectory of a complete
soaring cycle (transition + search + climb). The seed picked for each
aircraft is the smallest one (in increasing order) that matches the
requested ``--tau-T-range``, ``--tau-S-range``, ``--tau-C-range`` and
``--min-search-legs`` criteria.

Each panel is colour-coded by phase: transition (blue), search (red
dots, one per ballistic-leg sample), climb (green). The starting point
``(0, 0)`` and the two phase transitions ``T -> S`` and ``S -> C`` are
annotated. A zoom inset on the search + climb portion is added in the
bottom-right corner.

Outputs (under ``outputs/``)
---------------------------
- ``figures/example_trajectories.pdf``  -- the combined 1×N figure
- ``data/trajectory_panel/<aircraft>.npz``  -- the trajectory per aircraft
- ``data/trajectory_panel/<aircraft>.json`` -- the run manifest

The script honours the standard ``--cache`` policy (auto / reuse /
rerun / require / off): existing data is reused (after prompt, in
auto mode) when the manifest matches the requested parameters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cache import (
    add_cache_args,
    build_manifest,
    decide_action,
    load_dataset,
    save_dataset,
    slot_paths,
)
from model import SoaringConfig
from paths import CONFIGS_DIR, DATA_DIR, FIGURES_DIR
from simulation import interpolate_trajectory, simulate_single

SCRIPT_SLUG = "trajectory_panel"
N_PANELS = 1


def _find_typical_seeds(
    config: SoaringConfig,
    seed_max: int,
    tau_T_range: tuple[float, float],
    tau_S_range: tuple[float, float],
    tau_C_range: tuple[float, float],
    min_search_legs: int,
    n_panels: int,
    dt_fine: float,
) -> list[dict]:
    """Scan seeds 1..seed_max, keep the first ``n_panels`` whose first
    cycle satisfies the typicality ranges. Returns one dict per panel
    with the trajectory arrays needed for plotting."""
    panels: list[dict] = []
    for seed in range(1, seed_max):
        rng = np.random.default_rng(seed)
        traj = simulate_single(config, n_cycles=1, rng=rng)
        tau_T = float(traj.phase_durations_active[0, 0])
        tau_S = float(traj.phase_durations_active[0, 1])
        tau_C = float(traj.phase_durations_active[0, 2])
        if not (tau_T_range[0] < tau_T < tau_T_range[1]):
            continue
        if not (tau_S_range[0] < tau_S < tau_S_range[1]):
            continue
        if not (tau_C_range[0] < tau_C < tau_C_range[1]):
            continue
        n_legs = len(traj.search_episodes[0].leg_durations)
        if n_legs < min_search_legs:
            continue

        T_phys_S = float(traj.search_T_phys[0])
        T_total = float(traj.total_time)
        t_grid = np.arange(0.0, T_total, dt_fine)
        xy = interpolate_trajectory(traj, config, t_grid)

        panels.append({
            "seed": seed,
            "tau_T": tau_T, "tau_S": tau_S, "tau_C": tau_C,
            "T_phys_S": T_phys_S, "T_total": T_total,
            "T_turn": float(traj.climb_turn_period[0]),
            "n_legs": n_legs,
            "t_grid": t_grid,
            "xy": xy,
        })
        if len(panels) >= n_panels:
            return panels
    raise RuntimeError(
        f"Only {len(panels)}/{n_panels} typical seeds found in "
        f"[1, {seed_max}). Loosen the duration ranges or increase --seed-max."
    )


def _arrays_for_save(panels: list[dict], r0: float) -> dict[str, np.ndarray]:
    """Flatten panel dicts into an NPZ-friendly arrays dict."""
    out: dict[str, np.ndarray] = {
        "seeds":    np.array([p["seed"]     for p in panels], dtype=int),
        "tau_T":    np.array([p["tau_T"]    for p in panels]),
        "tau_S":    np.array([p["tau_S"]    for p in panels]),
        "tau_C":    np.array([p["tau_C"]    for p in panels]),
        "T_phys_S": np.array([p["T_phys_S"] for p in panels]),
        "T_total":  np.array([p["T_total"]  for p in panels]),
        "T_turn":   np.array([p["T_turn"]   for p in panels]),
        "n_legs":   np.array([p["n_legs"]   for p in panels], dtype=int),
        "r0":       np.array(r0),
    }
    for i, p in enumerate(panels):
        out[f"t_grid_{i}"] = p["t_grid"]
        out[f"xy_{i}"] = p["xy"]
    return out


def _panels_from_arrays(arrays: dict[str, np.ndarray]) -> tuple[list[dict], float]:
    """Inverse of ``_arrays_for_save``."""
    n = len(arrays["seeds"])
    panels = []
    for i in range(n):
        panels.append({
            "seed":     int(arrays["seeds"][i]),
            "tau_T":    float(arrays["tau_T"][i]),
            "tau_S":    float(arrays["tau_S"][i]),
            "tau_C":    float(arrays["tau_C"][i]),
            "T_phys_S": float(arrays["T_phys_S"][i]),
            "T_total":  float(arrays["T_total"][i]),
            "T_turn":   float(arrays["T_turn"][i]),
            "n_legs":   int(arrays["n_legs"][i]),
            "t_grid":   arrays[f"t_grid_{i}"],
            "xy":       arrays[f"xy_{i}"],
        })
    return panels, float(arrays["r0"])


def _draw_segments(ax, panel: dict, *, with_legend: bool) -> None:
    """Draw the three phase segments (T line, S dots, C line) and the
    three way-point markers (origin, T→S, S→C) on ``ax``. Legend labels
    are emitted only when ``with_legend`` is True (so the inset doesn't
    duplicate them)."""
    t = panel["t_grid"]
    xy = panel["xy"]
    tau_T = panel["tau_T"]
    T_phys_S = panel["T_phys_S"]

    is_T = t < tau_T
    is_S = (t >= tau_T) & (t < tau_T + T_phys_S)
    is_C = t >= tau_T + T_phys_S

    P_TS = xy[np.searchsorted(t, tau_T)]
    P_SC = xy[np.searchsorted(t, tau_T + T_phys_S)]

    ax.plot(
        xy[is_T, 0], xy[is_T, 1],
        "-", lw=1.4, color="#176ea8", alpha=0.9,
        label=rf"T ($\tau^T={tau_T:.0f}$s)" if with_legend else None,
    )
    ax.plot(
        xy[is_S, 0], xy[is_S, 1],
        ".", ms=3.0, color="#d62728", alpha=0.9,
        label=rf"S ($\tau^S={panel['tau_S']:.0f}$s)" if with_legend else None,
    )
    ax.plot(
        xy[is_C, 0], xy[is_C, 1],
        "-", lw=1.0, color="#2ca02c", alpha=0.9,
        label=rf"C ($\tau^C={panel['tau_C']:.0f}$s)" if with_legend else None,
    )
    ax.plot(0, 0, "k*", ms=10, zorder=5)
    ax.plot(P_TS[0], P_TS[1], "o", mec="black", mfc="white", ms=7, mew=1.0, zorder=5)
    ax.plot(P_SC[0], P_SC[1], "s", mec="black", mfc="white", ms=7, mew=1.0, zorder=5)


def _plot_one_panel(ax, panel: dict, r0: float) -> None:
    _draw_segments(ax, panel, with_legend=True)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title(
        f"seed={panel['seed']}, "
        f"{panel['n_legs']} search legs, "
        rf"$T_\mathrm{{turn}}={panel['T_turn']:.1f}$ s, "
        rf"$r_0={r0:.0f}$ m",
        fontsize=10, loc="left",
    )
    ax.legend(fontsize=9, loc="best")


def _square_limits(
    panel: dict,
    pad_frac: float = 0.05,
) -> tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` of a symmetric square bounding
    box around the trajectory, with a small padding."""
    xs = panel["xy"][:, 0]
    ys = panel["xy"][:, 1]
    extent = max(
        abs(xs.min()), abs(xs.max()), abs(ys.min()), abs(ys.max()),
    )
    extent *= (1.0 + pad_frac)
    return (-extent, extent, -extent, extent)


def _search_climb_box(
    panel: dict,
    pad_frac: float = 0.15,
) -> tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` of a square bounding box
    enclosing the S+C portion of the trajectory (excludes T)."""
    t = panel["t_grid"]
    xy = panel["xy"]
    tau_T = panel["tau_T"]
    mask = t >= tau_T
    xs = xy[mask, 0]
    ys = xy[mask, 1]
    cx = 0.5 * (xs.min() + xs.max())
    cy = 0.5 * (ys.min() + ys.max())
    half = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())
    half *= (1.0 + pad_frac)
    return (cx - half, cx + half, cy - half, cy + half)


def _add_search_climb_inset(ax, panel: dict) -> None:
    """Add a zoom inset on the search + climb portion."""
    axins = ax.inset_axes([0.62, 0.04, 0.34, 0.34])
    _draw_segments(axins, panel, with_legend=False)
    xmin, xmax, ymin, ymax = _search_climb_box(panel)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    axins.set_aspect("equal", adjustable="box")
    axins.grid(True, alpha=0.3)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("zoom: S + C", fontsize=8, loc="left", pad=2)
    ax.indicate_inset_zoom(axins, edgecolor="0.3", alpha=0.8, lw=0.8)


_AIRCRAFT_LABELS = {
    "paragliders": "paragliders",
    "hang_gliders": "hang gliders",
    "sailplanes": "sailplanes",
}
_PANEL_LETTERS = ("a", "b", "c", "d", "e", "f")


def plot_combined(
    panels_by_aircraft: dict[str, list[dict]],
    r0_by_aircraft: dict[str, float],
    output_path: Path,
    aircraft_order: list[str],
) -> None:
    """1×N figure with one sample cycle per aircraft (panels share no
    axes; each subplot has its own square box and aspect ratio so the
    very different aircraft scales coexist cleanly)."""
    n = len(aircraft_order)
    fig, axes = plt.subplots(
        1, n, figsize=(6.0 * n, 6.0), constrained_layout=True,
    )
    if n == 1:
        axes = [axes]
    for ax, aircraft, letter in zip(axes, aircraft_order, _PANEL_LETTERS):
        panel = panels_by_aircraft[aircraft][0]
        r0 = r0_by_aircraft[aircraft]
        _plot_one_panel(ax, panel, r0=r0)
        # Prefix the title with the panel letter and the aircraft name.
        old_title = ax.get_title(loc="left")
        ax.set_title(
            f"({letter}) {_AIRCRAFT_LABELS.get(aircraft, aircraft)}  —  {old_title}",
            fontsize=10, loc="left",
        )
        xmin, xmax, ymin, ymax = _square_limits(panel)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        _add_search_climb_inset(ax, panel)
    fig.suptitle(
        "Sample trajectories — single T→S→C cycle, typical durations",
        fontsize=12,
    )
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft", nargs="+",
        default=["paragliders", "hang_gliders", "sailplanes"],
        choices=["paragliders", "hang_gliders", "sailplanes"],
        help="One or more aircraft names. Default: all three, one panel each.",
    )
    parser.add_argument("--seed-max", type=int, default=5000)
    parser.add_argument(
        "--tau-T-range", nargs=2, type=float, default=[90.0, 400.0],
        metavar=("MIN", "MAX"),
        help="Default broadened to span all three aircraft.",
    )
    parser.add_argument(
        "--tau-S-range", nargs=2, type=float, default=[20.0, 200.0],
        metavar=("MIN", "MAX"),
        help="Default broadened to span all three aircraft.",
    )
    parser.add_argument(
        "--tau-C-range", nargs=2, type=float, default=[80.0, 250.0],
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--min-search-legs", type=int, default=2)
    parser.add_argument(
        "--output", type=Path,
        default=FIGURES_DIR / "example_trajectories.pdf",
    )
    parser.add_argument("--dt-fine", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    add_cache_args(parser)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    panels_by_aircraft: dict[str, list[dict]] = {}
    r0_by_aircraft: dict[str, float] = {}

    for aircraft in args.aircraft:
        print(f"=== {aircraft} ===")
        config_path = CONFIGS_DIR / f"{aircraft}.yaml"
        config = SoaringConfig.from_yaml(config_path)
        r0_value = float(getattr(config.climb_motion, "r0", float("nan")))

        npz_path, manifest_path = slot_paths(args.data_dir, SCRIPT_SLUG, aircraft)
        requested_manifest = build_manifest(
            script=SCRIPT_SLUG,
            params={
                "seed_max": args.seed_max,
                "tau_T_range": list(args.tau_T_range),
                "tau_S_range": list(args.tau_S_range),
                "tau_C_range": list(args.tau_C_range),
                "min_search_legs": args.min_search_legs,
                "dt_fine": args.dt_fine,
                "n_panels": N_PANELS,
            },
            config_paths={"aircraft": config_path},
        )
        decision = decide_action(
            npz_path=npz_path,
            manifest_path=manifest_path,
            requested_manifest=requested_manifest,
            mode=args.cache,
            slot_label=f"{SCRIPT_SLUG}/{aircraft}",
        )
        if decision.diff:
            print("  cache params differ:")
            for line in decision.diff:
                print(f"    - {line}")
        print(f"  cache decision: {decision.action}  ({decision.reason})")

        if decision.action == "reuse":
            arrays, _ = load_dataset(npz_path, manifest_path)
            panels, r0_value = _panels_from_arrays(arrays)
        else:
            print(f"  scanning seeds for a typical cycle...")
            panels = _find_typical_seeds(
                config,
                seed_max=args.seed_max,
                tau_T_range=tuple(args.tau_T_range),
                tau_S_range=tuple(args.tau_S_range),
                tau_C_range=tuple(args.tau_C_range),
                min_search_legs=args.min_search_legs,
                n_panels=N_PANELS,
                dt_fine=args.dt_fine,
            )
            if args.cache != "off":
                save_dataset(
                    npz_path=npz_path,
                    manifest_path=manifest_path,
                    manifest=requested_manifest,
                    arrays=_arrays_for_save(panels, r0=r0_value),
                )
                print(f"  saved data: {npz_path}")

        p = panels[0]
        print(
            f"    seed={p['seed']}  "
            f"tau_T={p['tau_T']:.1f}s  tau_S={p['tau_S']:.1f}s  "
            f"tau_C={p['tau_C']:.1f}s  legs={p['n_legs']}  "
            f"T_turn={p['T_turn']:.1f}s"
        )

        panels_by_aircraft[aircraft] = panels
        r0_by_aircraft[aircraft] = r0_value

    plot_combined(
        panels_by_aircraft=panels_by_aircraft,
        r0_by_aircraft=r0_by_aircraft,
        output_path=args.output,
        aircraft_order=list(args.aircraft),
    )
    print(f"\nSaved figure: {args.output}")


if __name__ == "__main__":
    main()
