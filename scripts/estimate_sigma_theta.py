"""1-D estimation of σ_θ from the universal H_eff ≈ 0.88 condition.

For each aircraft, all numerical parameters (``v_xy``, ``mu_T``,
``tau_0^T``, search/climb durations, intra-phase motion) are *fixed* at
the Table 1 values of the manuscript. Only the angular diffusivity
``sigma_theta`` is varied on a 1-D grid.

Per grid point we simulate ``--n-trajectories`` independent flights of
``--total-time`` seconds, compute the pure ensemble-averaged MSD
``⟨|r(Δ)-r(0)|²⟩`` (one pair per trajectory per lag),
and fit a single log-log slope over the empirical VDB fit window
(``[--fit-min, --fit-max]`` s, default ``[10, 7000]`` s — matching the
window used in Vilpellet, Darmon & Benzaquen (2026) for the empirical
H_eff measurement, so the calibration condition H_eff(sigma_theta*) =
0.88 is window-consistent with the data).

The resulting H_eff(σ_θ) curve is interpolated to find the σ_θ* that
satisfies H_eff(σ_θ*) = 0.88 (the universal empirical value).

Uncertainty and conventions
---------------------------
- The log-log fit uses, by default, every lag of the uniform grid in
  the window (``--fit-lag-spacing linear``, the convention quoted in
  the manuscript). ``--fit-lag-spacing log`` subsamples ``--n-log-lags``
  lags uniformly in log Δ instead; on crossover-shaped MSDs the two
  conventions differ by up to ~0.02 in H (class-dependent sign), which
  propagates to σ_θ*. The chosen convention is recorded in the
  manifest and in the calibration YAML.
- A replica-level uncertainty on σ_θ* is computed by re-extracting the
  H = 0.88 crossing on ``--n-groups`` disjoint sub-ensembles: the YAML
  stores the per-group values, their 5-95% range, and the standard
  error of the full-ensemble σ_θ* (std of groups / sqrt(n_groups)).

Two modes are produced independently:

  ``bare``  — only the transition phase carries displacement; search
              and climb phases consume their (Lomax / exponential)
              durations but contribute no horizontal motion.
  ``full``  — all three phases active with their intra-phase dynamics
              (search local CTRW + climb circling+drift) exactly as
              defined in the YAML.

Cache policy: identical to other scripts (see ``cache.py``).

Outputs per aircraft (under ``outputs/``):
    data/estimate_sigma_theta/{aircraft}_{mode}.npz
    data/estimate_sigma_theta/{aircraft}_{mode}.json
    figures/estimate_sigma_theta_{aircraft}_overlay.pdf
"""

from __future__ import annotations

import os

# Cap BLAS / OpenMP thread pools to 1 by default. We parallelise this
# script across aircraft with ProcessPoolExecutor; without these caps,
# every worker process would itself spawn cpu_count() native threads
# (numpy linked against Accelerate / OpenBLAS / MKL), oversubscribing
# the CPU and *slowing* the run. ``setdefault`` lets the user override
# from the shell (``OMP_NUM_THREADS=4 python scripts/...``) if a single
# aircraft run is preferred. Must run BEFORE numpy / scipy import.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


import hashlib

from soaring_ctrw.cache import (
    add_cache_args,
    build_manifest,
    decide_action,
    load_dataset,
    save_dataset,
    slot_paths,
)


def _seed_from(*parts) -> int:
    """Deterministic 32-bit-safe int seed from arbitrary labels."""
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:4], "big")
from soaring_ctrw.model import AngularConfig, SoaringConfig
from soaring_ctrw.observables import fit_hurst, msd_ensemble
from soaring_ctrw.paths import CONFIGS_DIR, DATA_DIR, FIGURES_DIR, REPO_ROOT
from soaring_ctrw.calibration import write_calibration_section
from soaring_ctrw.simulation import simulate_ensemble

SCRIPT_SLUG = "estimate_sigma_theta"
H_EMPIRICAL = 0.88
MODES = ("bare", "full")
AIRCRAFT_LABELS = {
    "paragliders": "paragliders",
    "hang_gliders": "hang gliders",
    "sailplanes": "sailplanes",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _config_for_mode(base: SoaringConfig, sigma_theta: float, mode: str) -> SoaringConfig:
    """Return a copy of ``base`` with ``sigma_theta`` set and the
    intra-phase motion either preserved (``full``) or stripped (``bare``)."""
    if mode == "bare":
        cfg = base.bare()
    elif mode == "full":
        cfg = base
    else:
        raise ValueError(f"unknown mode {mode!r}")
    # Replace angular config (frozen dataclasses → reconstruct).
    angular = AngularConfig(sigma_theta=sigma_theta, theta0=None)
    return SoaringConfig(
        name=f"{cfg.name}_sig{sigma_theta:.3f}",
        v_xy=cfg.v_xy,
        transition=cfg.transition,
        search=cfg.search,
        climb=cfg.climb,
        angular=angular,
        search_motion=cfg.search_motion,
        climb_motion=cfg.climb_motion,
    )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_curve(
    base: SoaringConfig,
    mode: str,
    sigma_grid: np.ndarray,
    n_trajectories: int,
    total_time: float,
    dt: float,
    fit_min: float,
    fit_max: float,
    rng: np.random.Generator,
    logger: logging.Logger,
    prefix: str = "",
    n_groups: int = 10,
    q_lo: float = 5.0,
    q_hi: float = 95.0,
    fit_lag_spacing: str = "linear",
    n_log_lags: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(H_array, msd_matrix, H_band, H_groups)`` for one
    (aircraft, mode).

    ``H_array`` has shape ``(len(sigma_grid),)`` (full-ensemble fit).
    ``msd_matrix`` has shape ``(len(sigma_grid), n_steps-1)``.
    ``H_band`` has shape ``(len(sigma_grid), 2)``: the ``q_lo``/``q_hi``
    percentiles (default 5--95) of ``H_eff`` refit on ``n_groups``
    disjoint sub-ensembles, i.e. the finite-sample spread of the
    calibration curve at sub-ensemble size ``M / n_groups``.
    ``H_groups`` has shape ``(len(sigma_grid), n_groups)``: the
    per-sub-ensemble refits themselves, used downstream to propagate a
    replica-level uncertainty onto ``sigma_theta*``.
    """
    n_steps = int(total_time / dt) + 1
    H_array = np.full(len(sigma_grid), np.nan)
    msd_matrix = np.full((len(sigma_grid), n_steps - 1), np.nan)
    H_band = np.full((len(sigma_grid), 2), np.nan)
    grp = max(2, int(n_groups))
    H_groups = np.full((len(sigma_grid), grp), np.nan)
    lags = np.arange(1, n_steps) * dt
    gsize = max(1, n_trajectories // grp)

    t_total = 0.0
    for k, sigma in enumerate(sigma_grid):
        cfg = _config_for_mode(base, sigma_theta=float(sigma), mode=mode)
        t0 = time.time()
        ens = simulate_ensemble(
            config=cfg,
            n_trajectories=n_trajectories,
            total_time=total_time,
            dt=dt,
            rng=rng,
        )
        msd = msd_ensemble(ens)[1:]
        msd_matrix[k] = msd
        try:
            H_array[k] = fit_hurst(
                lags, msd, (fit_min, fit_max),
                lag_spacing=fit_lag_spacing, n_log_lags=n_log_lags,
            ).hurst
            h_str = f"{H_array[k]:.3f}"
        except ValueError as exc:
            h_str = f"NaN ({exc!s})"
        # Sub-ensemble refits: spread band + per-group values (used for
        # the replica-level uncertainty on sigma_theta*). No bootstrap.
        if gsize >= 2:
            for j in range(grp):
                sub = ens[j * gsize:(j + 1) * gsize]
                try:
                    H_groups[k, j] = fit_hurst(
                        lags, msd_ensemble(sub)[1:], (fit_min, fit_max),
                        lag_spacing=fit_lag_spacing, n_log_lags=n_log_lags,
                    ).hurst
                except ValueError:
                    pass
            finite = np.isfinite(H_groups[k])
            if finite.sum() >= 2:
                H_band[k] = np.percentile(H_groups[k][finite], [q_lo, q_hi])
        t_cell = time.time() - t0
        t_total += t_cell
        msg = (
            f"[{k+1:>2}/{len(sigma_grid)}] mode={mode}  "
            f"sigma_theta={sigma:5.3f}  H_eff={h_str}  ({t_cell:.1f}s)"
        )
        print(f"{prefix}  {msg}", flush=True)
        logger.info("%s", msg)
    print(f"{prefix}  total wall time ({mode}): {t_total/60:.2f} min", flush=True)
    logger.info("total wall time mode=%s: %.2f min", mode, t_total / 60)
    return H_array, msd_matrix, H_band, H_groups


def _sigma_at_H(sigma_grid: np.ndarray, H_array: np.ndarray, H_target: float) -> float | None:
    """Linear interpolation of the σ_θ at which H_eff = H_target.

    Assumes H_eff is monotonically decreasing in σ_θ (more
    decorrelation → lower H). Returns ``None`` if H_target is outside
    the sampled range.
    """
    finite = ~np.isnan(H_array)
    if finite.sum() < 2:
        return None
    sig = sigma_grid[finite]
    Hf = H_array[finite]
    if H_target > Hf.max() or H_target < Hf.min():
        return None
    if Hf[0] > Hf[-1]:
        return float(np.interp(H_target, Hf[::-1], sig[::-1]))
    return float(np.interp(H_target, Hf, sig))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_one(
    aircraft: str,
    mode: str,
    sigma_grid: np.ndarray,
    H_array: np.ndarray,
    sigma_star: float | None,
    fit_min: float,
    fit_max: float,
    total_time: float,
    n_trajectories: int,
    output_path: Path,
) -> None:
    """Single-mode plot for one aircraft."""
    fig, (ax_strip, ax_curve) = plt.subplots(
        2, 1, figsize=(7, 5.2),
        gridspec_kw={"height_ratios": [0.18, 1.0]},
        constrained_layout=True, sharex=True,
    )

    # ── Top: 1-D heatmap strip ─────────────────────────────────────────
    H_strip = H_array[np.newaxis, :]
    extent = [sigma_grid[0], sigma_grid[-1], 0, 1]
    im = ax_strip.imshow(
        H_strip, aspect="auto", origin="lower", extent=extent,
        cmap="viridis", vmin=0.5, vmax=1.0,
    )
    ax_strip.set_yticks([])
    ax_strip.set_title(
        rf"H_eff strip — {AIRCRAFT_LABELS[aircraft]} ({mode}-cycle)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax_strip, label=r"$H_\mathrm{eff}$")

    # ── Bottom: line plot ──────────────────────────────────────────────
    ax_curve.plot(sigma_grid, H_array, "o-", color="C0", lw=1.5, ms=5)
    ax_curve.axhline(H_EMPIRICAL, color="k", ls="--", lw=1.2,
                     label=rf"$H={H_EMPIRICAL}$")
    if sigma_star is not None:
        ax_curve.axvline(sigma_star, color="red", ls=":", lw=1.2)
        ax_curve.plot([sigma_star], [H_EMPIRICAL], "o", mec="red",
                      mfc="white", ms=9, mew=1.5)
        ax_curve.annotate(
            rf"$\sigma_\theta^\star={sigma_star:.3f}$ rad",
            xy=(sigma_star, H_EMPIRICAL),
            xytext=(8, -18), textcoords="offset points",
            fontsize=10, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=0.6),
        )
    ax_curve.set_xlabel(r"$\sigma_\theta$ (rad)")
    ax_curve.set_ylabel(r"$H_\mathrm{eff}$")
    ax_curve.set_title(
        rf"$H_\mathrm{{eff}}(\sigma_\theta)$ — {AIRCRAFT_LABELS[aircraft]} "
        rf"({mode}-cycle, $\Delta\in[{fit_min:g},{fit_max:g}]$ s, "
        rf"{n_trajectories} traj × {total_time:.0f} s)",
        fontsize=10,
    )
    ax_curve.legend(fontsize=8)
    ax_curve.grid(True, ls=":", alpha=0.5)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_overlay(
    aircraft: str,
    results: dict[str, dict],
    fit_min: float,
    fit_max: float,
    output_path: Path,
    n_groups: int = 10,
) -> None:
    """Overlay of all available modes for one aircraft."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    colors = {"bare": "C0", "full": "C3"}

    for mode, res in results.items():
        sigma_grid = res["sigma_grid"]
        H_array = res["H_array"]
        H_band = res.get("H_band")
        sigma_star = res["sigma_star"]
        c = colors.get(mode, "C2")
        if H_band is not None and np.ndim(H_band) == 2 and H_band.shape[0] == len(sigma_grid):
            good = np.isfinite(H_band[:, 0]) & np.isfinite(H_band[:, 1])
            if good.any():
                ax.fill_between(sigma_grid[good], H_band[good, 0], H_band[good, 1],
                                color=c, alpha=0.18, lw=0,
                                label=rf"{mode}-cycle: 5--95% over {max(2, int(n_groups))} sub-ensembles")
        ax.plot(sigma_grid, H_array, "o-", color=c, lw=1.5, ms=4,
                label=rf"{mode}-cycle")
        if sigma_star is not None:
            ax.plot([sigma_star], [H_EMPIRICAL], "s", mec=c, mfc="white",
                    ms=10, mew=1.5)
            se = res.get("sigma_star_se")
            label_star = rf"$\sigma_\theta^\star={sigma_star:.3f}$"
            if se is not None:
                label_star = (
                    rf"$\sigma_\theta^\star={sigma_star:.3f}"
                    rf"\pm{se:.3f}$"
                )
            ax.annotate(
                label_star,
                xy=(sigma_star, H_EMPIRICAL),
                xytext=(6, -16 if mode == "bare" else 10),
                textcoords="offset points",
                fontsize=9, color=c,
            )

    ax.axhline(H_EMPIRICAL, color="k", ls="--", lw=1.2, label=rf"$H={H_EMPIRICAL}$")
    ax.set_xlabel(r"$\sigma_\theta$ (rad)")
    ax.set_ylabel(r"$H_\mathrm{eff}$")
    ax.set_title(
        rf"$H_\mathrm{{eff}}(\sigma_\theta)$ — {AIRCRAFT_LABELS[aircraft]} "
        rf"(fit window $[{fit_min:g},{fit_max:g}]$ s)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-aircraft worker (parallelisable across processes)
# ---------------------------------------------------------------------------


def _process_aircraft(
    aircraft: str,
    args: argparse.Namespace,
    sigma_grid: np.ndarray,
) -> dict:
    """Run the full σ_θ scan for one aircraft.

    Self-contained so it can be dispatched to a worker process.
    Each aircraft writes to its own log file under ``args.logs_dir`` and
    prefixes stdout lines with ``[aircraft]`` so that parallel output
    remains readable.
    """
    prefix = f"[{aircraft}]"

    log_path = args.logs_dir / f"{SCRIPT_SLUG}_{aircraft}_{int(time.time())}_{os.getpid()}.log"
    logger = logging.getLogger(f"{SCRIPT_SLUG}.{aircraft}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Avoid duplicate handlers if the worker reuses the logger object.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    print(f"{prefix} start  log={log_path}", flush=True)
    logger.info("aircraft=%s pid=%d", aircraft, os.getpid())

    base = SoaringConfig.from_yaml(CONFIGS_DIR / f"{aircraft}.yaml")
    results: dict[str, dict] = {}

    for mode in args.mode:
        rng = np.random.default_rng(_seed_from(args.seed, aircraft, mode))

        slot = f"{aircraft}_{mode}"
        npz_path, manifest_path = slot_paths(args.data_dir, SCRIPT_SLUG, slot)
        requested_manifest = build_manifest(
            script=SCRIPT_SLUG,
            params={
                "mode": mode,
                "n_sigma": args.n_sigma,
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "n_trajectories": args.n_trajectories,
                "total_time": args.total_time,
                "dt": args.dt,
                "fit_min": args.fit_min,
                "fit_max": args.fit_max,
                "seed": args.seed,
                "n_groups": args.n_groups,
                "band": "sample-percentile-5-95",
                "msd_estimator": "ea",
                "fit_lag_spacing": args.fit_lag_spacing,
                "n_log_lags": args.n_log_lags,
                "h_groups": True,
            },
            config_paths={"aircraft": CONFIGS_DIR / f"{aircraft}.yaml"},
        )

        decision = decide_action(
            npz_path=npz_path,
            manifest_path=manifest_path,
            requested_manifest=requested_manifest,
            mode=args.cache,
            slot_label=f"{SCRIPT_SLUG}/{slot}",
        )
        if decision.diff:
            print(f"{prefix}  [{mode}] cache params differ:", flush=True)
            for line in decision.diff:
                print(f"{prefix}    - {line}", flush=True)
        print(
            f"{prefix}  [{mode}] cache decision: {decision.action}  "
            f"({decision.reason})",
            flush=True,
        )
        logger.info("mode=%s decision=%s (%s) diff=%s",
                    mode, decision.action, decision.reason, decision.diff)

        if decision.action == "reuse":
            arrays, _ = load_dataset(npz_path, manifest_path)
            H_array = arrays["H_array"]
            msd_matrix = arrays.get("msd_matrix")
            H_band = arrays.get("H_band")
            H_groups = arrays.get("H_groups")
            sigma_grid_cached = arrays["sigma_grid"]
            if not np.allclose(sigma_grid_cached, sigma_grid):
                sigma_grid = sigma_grid_cached
        else:
            H_array, msd_matrix, H_band, H_groups = _compute_curve(
                base, mode, sigma_grid,
                n_trajectories=args.n_trajectories,
                total_time=args.total_time,
                dt=args.dt,
                fit_min=args.fit_min,
                fit_max=args.fit_max,
                rng=rng,
                logger=logger,
                prefix=prefix,
                n_groups=args.n_groups,
                fit_lag_spacing=args.fit_lag_spacing,
                n_log_lags=args.n_log_lags,
            )
            if args.cache != "off":
                save_dataset(
                    npz_path=npz_path,
                    manifest_path=manifest_path,
                    manifest=requested_manifest,
                    arrays={
                        "sigma_grid": sigma_grid,
                        "H_array": H_array,
                        "msd_matrix": msd_matrix,
                        "H_band": H_band,
                        "H_groups": H_groups,
                    },
                )
                print(f"{prefix}  [{mode}] saved data: {npz_path}", flush=True)
                logger.info("saved data: %s", npz_path)

        sigma_star = _sigma_at_H(sigma_grid, H_array, H_EMPIRICAL)

        # Replica-level uncertainty: re-extract the crossing on each
        # disjoint sub-ensemble curve. std/sqrt(n) estimates the
        # standard error of the full-ensemble sigma_star.
        sigma_star_groups: list[float] = []
        if H_groups is not None and np.ndim(H_groups) == 2:
            for j in range(H_groups.shape[1]):
                s_j = _sigma_at_H(sigma_grid, H_groups[:, j], H_EMPIRICAL)
                if s_j is not None:
                    sigma_star_groups.append(float(s_j))
        sigma_star_se: float | None = None
        sigma_star_ci: list[float] | None = None
        if len(sigma_star_groups) >= 2:
            arr = np.asarray(sigma_star_groups)
            sigma_star_se = float(arr.std(ddof=1) / np.sqrt(arr.size))
            sigma_star_ci = [
                float(np.percentile(arr, 5)), float(np.percentile(arr, 95)),
            ]

        if sigma_star is None:
            print(
                f"{prefix}  [{mode}] WARNING: H={H_EMPIRICAL} not reached in "
                f"σ_θ ∈ [{args.sigma_min}, {args.sigma_max}].",
                flush=True,
            )
            logger.warning("mode=%s sigma_star not found", mode)
        else:
            se_str = (
                f" ± {sigma_star_se:.4f} (replica s.e., "
                f"{len(sigma_star_groups)} sub-ensembles; "
                f"5–95% [{sigma_star_ci[0]:.4f}, {sigma_star_ci[1]:.4f}])"
                if sigma_star_se is not None else ""
            )
            print(
                f"{prefix}  [{mode}] σ_θ* (H={H_EMPIRICAL}) = "
                f"{sigma_star:.4f}{se_str} rad",
                flush=True,
            )
            logger.info("mode=%s sigma_star=%.4f se=%s",
                        mode, sigma_star, sigma_star_se)

        results[mode] = {
            "sigma_grid": sigma_grid,
            "H_array": H_array,
            "msd_matrix": msd_matrix,
            "H_band": H_band,
            "H_groups": H_groups,
            "sigma_star": sigma_star,
            "sigma_star_se": sigma_star_se,
            "sigma_star_ci": sigma_star_ci,
            "sigma_star_groups": sigma_star_groups,
        }

    if results:
        overlay_path = (
            args.figures_dir / f"{SCRIPT_SLUG}_{aircraft}_overlay.pdf"
        )
        _plot_overlay(
            aircraft=aircraft,
            results=results,
            fit_min=args.fit_min,
            fit_max=args.fit_max,
            output_path=overlay_path,
            n_groups=args.n_groups,
        )
        print(f"{prefix}  overlay saved: {overlay_path}", flush=True)
        logger.info("saved overlay: %s", overlay_path)

    if args.write:
        by_mode = {
            mode: (float(res["sigma_star"])
                   if res["sigma_star"] is not None else None)
            for mode, res in results.items()
        }
        primary_mode = "full" if "full" in by_mode else next(iter(by_mode))
        primary_value = by_mode.get(primary_mode)
        if primary_value is None:
            print(
                f"{prefix}  WARNING: no sigma_star found; "
                "calibration YAML not updated.",
                flush=True,
            )
        else:
            primary_res = results[primary_mode]
            payload = {
                "source_script": "estimate_sigma_theta",
                "value": primary_value,
                "value_se": primary_res.get("sigma_star_se"),
                "value_ci_5_95": primary_res.get("sigma_star_ci"),
                "value_groups": primary_res.get("sigma_star_groups"),
                "mode": primary_mode,
                "by_mode": by_mode,
                "by_mode_se": {
                    m: res.get("sigma_star_se")
                    for m, res in results.items()
                },
                "H_target": float(H_EMPIRICAL),
                "fit_window": [float(args.fit_min), float(args.fit_max)],
                "fit_lag_spacing": str(args.fit_lag_spacing),
                "n_log_lags": int(args.n_log_lags),
                "sigma_grid": [float(args.sigma_min), float(args.sigma_max),
                               int(args.n_sigma)],
                "n_trajectories": int(args.n_trajectories),
                "total_time": float(args.total_time),
                "dt": float(args.dt),
                "seed": int(args.seed),
                "n_subensembles": int(args.n_groups),
            }
            out = write_calibration_section(aircraft, "sigma_theta", payload)
            print(
                f"{prefix}  wrote {out}  (section: sigma_theta, "
                f"value={primary_value:.4f} rad, mode={primary_mode})",
                flush=True,
            )
            logger.info("wrote sigma_theta to %s", out)

    fh.close()
    logger.removeHandler(fh)

    # Strip non-picklable / heavy fields before returning to the parent.
    summary = {
        mode: {
            "sigma_star": res["sigma_star"],
        }
        for mode, res in results.items()
    }
    return {"aircraft": aircraft, "summary": summary, "log_path": str(log_path)}


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
    parser.add_argument(
        "--mode", nargs="+", default=list(MODES), choices=list(MODES),
        help="Run one or both intra-phase modes. Default: both.",
    )
    parser.add_argument("--n-sigma", type=int, default=16,
                        help="Number of σ_θ values in the 1-D grid.")
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=1.5)
    parser.add_argument(
        "--n-trajectories", type=int, default=2000,
        help="Flights per sigma_theta grid point. Heavy-tailed "
             "transition durations (mu_T < 4 for paragliders and "
             "sailplanes) reward large ensembles (see Appendix D).",
    )
    parser.add_argument(
        "--total-time", type=float, default=15_000.0,
        help="Trajectory length (s). Must be ≥ fit_max so each trajectory "
             "contributes a pair at the largest fit lag.",
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--fit-min", type=float, default=10.0,
        help="Lower edge of the log-log MSD fit window (s). Matches "
             "the VDB empirical fit range used in Ref. Vilpellet "
             "et al. (2026) for direct calibration consistency.",
    )
    parser.add_argument(
        "--fit-max", type=float, default=7_000.0,
        help="Upper edge of the log-log MSD fit window (s). Matches "
             "the VDB empirical fit range. Requires total_time >= "
             "fit_max so the EA-MSD has samples at the upper edge.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-groups", type=int, default=10,
        help="Disjoint sub-ensembles used for the 5-95%% sample band on "
             "the H_eff(sigma_theta) calibration curve and for the "
             "replica-level standard error of sigma_theta*.",
    )
    parser.add_argument(
        "--fit-lag-spacing", choices=("linear", "log"), default="linear",
        help="Lag-spacing convention of the log-log H fit: 'linear' "
             "uses every lag of the uniform grid in the window (the "
             "manuscript's declared convention); 'log' subsamples "
             "--n-log-lags lags uniformly in log(lag). The two differ "
             "by up to ~0.02 in H on crossover-shaped MSDs, so the "
             "choice is recorded in the cache manifest and in the "
             "calibration YAML.",
    )
    parser.add_argument(
        "--n-log-lags", type=int, default=40,
        help="Number of log-spaced lags when --fit-lag-spacing=log.",
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Also merge sigma_theta* into "
            "outputs/data/calibration/<aircraft>.yaml under the "
            "`sigma_theta` section. If both modes are run, the "
            "`full`-cycle value is used as the canonical `value`."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help=(
            "Number of worker processes used to parallelise the scan "
            "across aircraft (each aircraft is independent). "
            "0 (default) picks min(#aircraft, os.cpu_count()). "
            "Use 1 to run serially."
        ),
    )
    add_cache_args(parser)
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    # Parent-process orchestration log. Per-aircraft workers write their
    # own log file (see _process_aircraft); we just record dispatch here.
    parent_log = args.logs_dir / f"{SCRIPT_SLUG}_main_{int(time.time())}.log"
    logger = logging.getLogger(SCRIPT_SLUG)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(parent_log)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    print(f"Main log: {parent_log}")
    logger.info("start run; args=%s", vars(args))

    sigma_grid = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma)

    n_workers = args.workers
    if n_workers <= 0:
        n_workers = min(len(args.aircraft), os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(args.aircraft)))

    print(
        f"1-D σ_θ scan to estimate σ_θ* from H_eff = {H_EMPIRICAL}.\n"
        f"Aircraft: {', '.join(args.aircraft)}.  Modes: {', '.join(args.mode)}.\n"
        f"σ_θ grid: {args.n_sigma} points in [{args.sigma_min:g}, {args.sigma_max:g}].\n"
        f"Per cell: {args.n_trajectories} traj × {args.total_time:.0f} s, "
        f"fit window [{args.fit_min:g}, {args.fit_max:g}] s.\n"
        f"Workers: {n_workers} (across {len(args.aircraft)} aircraft).\n"
    )
    logger.info("dispatching %d aircraft across %d workers",
                len(args.aircraft), n_workers)

    if n_workers == 1 or len(args.aircraft) == 1:
        for aircraft in args.aircraft:
            res = _process_aircraft(aircraft, args, sigma_grid)
            logger.info("done aircraft=%s summary=%s",
                        res["aircraft"], res["summary"])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_aircraft, aircraft, args, sigma_grid):
                    aircraft
                for aircraft in args.aircraft
            }
            for fut in as_completed(futures):
                aircraft = futures[fut]
                try:
                    res = fut.result()
                except Exception:
                    logger.exception("worker failed for aircraft=%s", aircraft)
                    raise
                logger.info("done aircraft=%s summary=%s",
                            res["aircraft"], res["summary"])
                print(f"[{aircraft}] worker finished.", flush=True)

    logger.info("end run")


if __name__ == "__main__":
    main()
