"""Sensitivity of the fitted Hurst exponent to ±20% perturbations of
eye-read (observational) parameters.

For each aircraft class and each parameter listed in ``SCAN_PARAMS``, the
script perturbs the parameter to 0.8× and 1.2× its nominal value (keeping
all others fixed), re-runs a small cycle-counted MSD simulation, fits H,
and reports the shift  ΔH = H_perturbed − H_nominal.  This quantifies which
observational uncertainties matter most for H_fit.

Parameters scanned (Table 1 of the manuscript, eye-read from field data):
  - ``tau_0_T``     — Lomax scale of transition durations
  - ``mu_T``        — Lomax tail index of transition durations
  - ``v_xy``        — horizontal cruise speed
  - ``T_turn_mean`` — mean climb turn period
  - ``T_turn_std``  — standard deviation of climb turn period
  - ``v_drift``     — orographic drift speed during climb
  - ``sigma_theta`` — directional persistence (calibrated, included for
                      reference to check self-consistency)

Usage::

    python scripts/scan_parameter_sensitivity.py \\
        --n-cycles 300 --n-traj 500 --seed 42

Outputs a plain-text table to stdout and optionally saves it to a CSV at
``outputs/data/sensitivity_scan.csv``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np

# Ensure the src layout is importable when the script is run from the repo root.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from soaring_ctrw.calibration import load_calibrated_config
from soaring_ctrw.model import (
    ClimbMotionConfig,
    PhaseConfig,
    SoaringConfig,
)
from soaring_ctrw.observables import fit_hurst
from soaring_ctrw.paths import DATA_DIR

AIRCRAFT_ORDER = ("paragliders", "hang_gliders", "sailplanes")

# Each entry: (label_for_table, function_that_perturbs_config)
# The function receives (cfg: SoaringConfig, factor: float) → SoaringConfig.
# Returning None means "skip this parameter for this aircraft".


def _perturb_tau0_T(cfg: SoaringConfig, factor: float) -> SoaringConfig:
    old = cfg.transition.params
    new_params = dict(old, tau_0=old["tau_0"] * factor)
    new_trans = dataclasses.replace(cfg.transition, params=new_params)
    return dataclasses.replace(cfg, transition=new_trans)


def _perturb_mu_T(cfg: SoaringConfig, factor: float) -> SoaringConfig:
    old = cfg.transition.params
    new_mu = old["mu"] * factor
    # mu_T > 2 is required for finite variance; clamp at 2.05 to avoid inf.
    new_mu = max(new_mu, 2.05)
    new_params = dict(old, mu=new_mu)
    new_trans = dataclasses.replace(cfg.transition, params=new_params)
    return dataclasses.replace(cfg, transition=new_trans)


def _perturb_v_xy(cfg: SoaringConfig, factor: float) -> SoaringConfig:
    return dataclasses.replace(cfg, v_xy=cfg.v_xy * factor)


def _perturb_T_turn_mean(cfg: SoaringConfig, factor: float) -> SoaringConfig:
    cm = cfg.climb_motion
    if cm is None:
        return cfg
    new_cm = dataclasses.replace(cm, T_turn_mean=cm.T_turn_mean * factor)
    return dataclasses.replace(cfg, climb_motion=new_cm)


def _perturb_T_turn_std(cfg: SoaringConfig, factor: float) -> SoaringConfig:
    cm = cfg.climb_motion
    if cm is None:
        return cfg
    new_cm = dataclasses.replace(cm, T_turn_std=cm.T_turn_std * factor)
    return dataclasses.replace(cfg, climb_motion=new_cm)


def _perturb_v_drift(cfg: SoaringConfig, factor: float) -> SoaringConfig:
    cm = cfg.climb_motion
    if cm is None:
        return cfg
    new_cm = dataclasses.replace(cm, v_drift=cm.v_drift * factor)
    return dataclasses.replace(cfg, climb_motion=new_cm)


def _perturb_sigma_theta(cfg: SoaringConfig, factor: float) -> SoaringConfig:
    new_ang = dataclasses.replace(
        cfg.angular, sigma_theta=cfg.angular.sigma_theta * factor
    )
    return dataclasses.replace(cfg, angular=new_ang)


SCAN_PARAMS: list[tuple[str, object]] = [
    ("tau_0_T",      _perturb_tau0_T),
    ("mu_T",         _perturb_mu_T),
    ("v_xy",         _perturb_v_xy),
    ("T_turn_mean",  _perturb_T_turn_mean),
    ("T_turn_std",   _perturb_T_turn_std),
    ("v_drift",      _perturb_v_drift),
    ("sigma_theta",  _perturb_sigma_theta),
]


def simulate_cycle_msd(
    cfg: SoaringConfig,
    n_cycles: int,
    n_traj: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the EA-MSD ``(n_cycles+1,)`` from ``n_traj`` trajectories."""
    from soaring_ctrw.simulation import simulate_single

    r2 = np.zeros((n_traj, n_cycles + 1), dtype=np.float64)
    for m in range(n_traj):
        traj = simulate_single(cfg, n_cycles=n_cycles, rng=rng)
        pos = traj.positions          # shape (n_cycles+1, 2) or similar
        disp = pos - pos[0:1, :]
        r2[m] = np.sum(disp ** 2, axis=-1)
    return r2.mean(axis=0)


def _fit_H(msd: np.ndarray, n_cycles: int, fit_frac: tuple[float, float]) -> float:
    """Fit H on the cycle-count axis using log-spaced lags."""
    lags = np.arange(n_cycles + 1, dtype=float)
    lags[0] = np.nan
    valid = lags[1:]
    msd_v = msd[1:]
    lo = fit_frac[0] * n_cycles
    hi = fit_frac[1] * n_cycles
    lo = max(lo, 1.0)
    try:
        result = fit_hurst(valid, msd_v, (lo, hi), lag_spacing="log", n_log_lags=30)
        return result.hurst
    except ValueError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft",
        nargs="+",
        default=list(AIRCRAFT_ORDER),
        help="Aircraft configs to scan.",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=300,
        help="Number of cycles per trajectory (default: 300).",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=500,
        help="Trajectories per configuration (default: 500).",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=0.20,
        help="Fractional perturbation applied (default: 0.20 = ±20%%).",
    )
    parser.add_argument(
        "--fit-frac-lo",
        type=float,
        default=0.05,
        help="Lower end of the fit window as a fraction of n_cycles (default: 0.05).",
    )
    parser.add_argument(
        "--fit-frac-hi",
        type=float,
        default=0.80,
        help="Upper end of the fit window as a fraction of n_cycles (default: 0.80).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed (default: 42).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="If given, save results to this CSV file in addition to printing.",
    )
    args = parser.parse_args()

    fit_frac = (args.fit_frac_lo, args.fit_frac_hi)
    delta = args.perturb
    factors = {f"-{int(100*delta)}%": 1.0 - delta, f"+{int(100*delta)}%": 1.0 + delta}

    # -------------------------------------------------------------------------
    # Load configs
    # -------------------------------------------------------------------------
    configs: dict[str, SoaringConfig] = {}
    for ac in args.aircraft:
        try:
            configs[ac] = load_calibrated_config(ac)
        except Exception as exc:
            print(f"WARNING: could not load config for {ac}: {exc}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Nominal H for each aircraft
    # -------------------------------------------------------------------------
    H_nominal: dict[str, float] = {}
    print("Computing nominal H values …", file=sys.stderr)
    for i, ac in enumerate(args.aircraft):
        if ac not in configs:
            continue
        rng = np.random.default_rng(args.seed + i * 1000)
        msd = simulate_cycle_msd(configs[ac], args.n_cycles, args.n_traj, rng)
        H_nominal[ac] = _fit_H(msd, args.n_cycles, fit_frac)
        print(f"  {ac}: H_nominal = {H_nominal[ac]:.4f}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Sensitivity scan
    # -------------------------------------------------------------------------
    # rows: list of dicts with columns aircraft, param, direction, H, ΔH
    rows: list[dict] = []

    for param_name, perturb_fn in SCAN_PARAMS:
        print(f"\nScanning {param_name} …", file=sys.stderr)
        for i, ac in enumerate(args.aircraft):
            if ac not in configs or ac not in H_nominal:
                continue
            cfg_base = configs[ac]
            for direction_label, factor in factors.items():
                cfg_p = perturb_fn(cfg_base, factor)
                rng = np.random.default_rng(args.seed + i * 1000 + hash(param_name + direction_label) % 10_000)
                msd = simulate_cycle_msd(cfg_p, args.n_cycles, args.n_traj, rng)
                H_p = _fit_H(msd, args.n_cycles, fit_frac)
                dH = H_p - H_nominal[ac]
                rows.append({
                    "aircraft": ac,
                    "param": param_name,
                    "direction": direction_label,
                    "H_nominal": H_nominal[ac],
                    "H_perturbed": H_p,
                    "dH": dH,
                })
                print(
                    f"  {ac:14s}  {param_name:14s}  {direction_label:5s}  "
                    f"H = {H_p:.4f}  ΔH = {dH:+.4f}",
                    file=sys.stderr,
                )

    # -------------------------------------------------------------------------
    # Pretty-print table
    # -------------------------------------------------------------------------
    col_w = 14
    header = (
        f"{'aircraft':<{col_w}}  {'parameter':<{col_w}}  "
        f"{'direction':>7}  {'H_nom':>7}  {'H_pert':>7}  {'ΔH':>7}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    prev_param = None
    for r in rows:
        if r["param"] != prev_param:
            if prev_param is not None:
                print()
            prev_param = r["param"]
        print(
            f"{r['aircraft']:<{col_w}}  {r['param']:<{col_w}}  "
            f"{r['direction']:>7}  {r['H_nominal']:>7.4f}  "
            f"{r['H_perturbed']:>7.4f}  {r['dH']:>+7.4f}"
        )
    print(sep)

    # -------------------------------------------------------------------------
    # Optional CSV output
    # -------------------------------------------------------------------------
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["aircraft", "param", "direction",
                             "H_nominal", "H_perturbed", "dH"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {args.csv}")
    else:
        default_csv = DATA_DIR / "sensitivity_scan.csv"
        default_csv.parent.mkdir(parents=True, exist_ok=True)
        with default_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["aircraft", "param", "direction",
                             "H_nominal", "H_perturbed", "dH"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults also saved to {default_csv}")


if __name__ == "__main__":
    main()
