"""Compute the pre-asymptotic crossover (``critical``) time ``t_c``.

The manuscript (Di Sante 2026, App. ``The coherent slope and the
crossover``) shows that the ballistic-to-diffusive crossover of the
transition-phase displacement is controlled by the number of cycles

    n_c = 2 / sigma_theta^2

and the corresponding lag

    t_c = n_c * <T> = 2 * <T> / sigma_theta^2,

where ``<T> = <tau_T> + <tau_S> + <tau_C>`` is the mean cycle duration
and ``sigma_theta`` is the cycle-to-cycle heading dispersion calibrated
against ``H_eff = 0.88``.

Inputs per aircraft:

    * ``sigma_theta`` — read from
      ``outputs/data/calibration/<aircraft>.yaml`` (full-cycle value,
      written by ``scripts/estimate_sigma_theta.py --write``).
    * Phase-duration scales — read from ``configs/<aircraft>.yaml``.

This script must be run *after* ``estimate_sigma_theta.py --write``.

Outputs:

    * stdout table.
    * with ``--write``: a ``critical_time`` section merged into
      ``outputs/data/calibration/<aircraft>.yaml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


from soaring_ctrw.paths import CONFIGS_DIR  # noqa: E402
from soaring_ctrw.calibration import (  # noqa: E402
    calibrated_sigma_theta,
    calibration_path,
    read_calibration,
    write_calibration_section,
)

DEFAULT_AIRCRAFT = ("paragliders", "hang_gliders", "sailplanes")


def phase_mean(phase: dict) -> float:
    """Analytic mean duration of a phase block from its YAML spec.

    Supports ``lomax`` (``mu``, ``tau_0``; requires ``mu > 1``) and
    ``exponential`` (``tau_mean``). Pareto / Mittag-Leffler are not
    used by the empirical configs and are rejected explicitly.
    """
    dist = str(phase["distribution"]).lower()
    p = phase.get("params", {})
    if dist == "lomax":
        mu, tau_0 = float(p["mu"]), float(p["tau_0"])
        if mu <= 1.0:
            raise ValueError(
                f"Lomax mean diverges for mu={mu} <= 1; cannot compute <T>."
            )
        return tau_0 / (mu - 1.0)
    if dist == "exponential":
        return float(p["tau_mean"])
    raise ValueError(
        f"phase_mean: unsupported distribution {dist!r}; expected "
        "'lomax' or 'exponential'."
    )


def cycle_mean(cfg: dict) -> tuple[float, float, float, float]:
    """Return ``(<tau_T>, <tau_S>, <tau_C>, <T>)`` for an aircraft config."""
    tT = phase_mean(cfg["transition"])
    tS = phase_mean(cfg["search"])
    tC = phase_mean(cfg["climb"])
    return tT, tS, tC, tT + tS + tC


def critical_time(sigma_theta: float, mean_T: float) -> tuple[float, float]:
    """Return ``(n_c, t_c)`` for given ``sigma_theta`` and ``<T>``."""
    if sigma_theta <= 0:
        raise ValueError(f"sigma_theta must be > 0, got {sigma_theta}")
    n_c = 2.0 / (sigma_theta ** 2)
    return n_c, n_c * mean_T


def _load_aircraft_yaml(aircraft: str) -> dict:
    p = CONFIGS_DIR / f"{aircraft}.yaml"
    if not p.is_file():
        raise SystemExit(f"Config not found: {p}")
    with p.open() as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft",
        nargs="+",
        default=list(DEFAULT_AIRCRAFT),
        help="Aircraft config names (without .yaml) under configs/.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Merge results into outputs/data/calibration/<aircraft>.yaml "
            "under the `critical_time` section."
        ),
    )
    args = parser.parse_args()

    print(
        f"{'aircraft':<14} {'sigma_theta':>12} {'<tau_T>':>9} "
        f"{'<tau_S>':>9} {'<tau_C>':>9} {'<T>':>9} "
        f"{'n_c':>9} {'t_c [s]':>10}"
    )
    for aircraft in args.aircraft:
        try:
            sigma = calibrated_sigma_theta(aircraft)
        except FileNotFoundError as exc:
            print(f"{aircraft:<14}  SKIP: {exc}")
            continue

        # Trace which sigma_theta we are using (full vs bare).
        cal = read_calibration(aircraft)
        sec = cal.get("sigma_theta", {})
        mode = sec.get("mode", "?")

        cfg = _load_aircraft_yaml(aircraft)
        tT, tS, tC, T = cycle_mean(cfg)
        n_c, t_c = critical_time(sigma, T)

        print(
            f"{aircraft:<14} {sigma:>12.6f} {tT:>9.2f} "
            f"{tS:>9.2f} {tC:>9.2f} {T:>9.2f} "
            f"{n_c:>9.2f} {t_c:>10.2f}    (sigma mode={mode})"
        )

        if args.write:
            payload = {
                "source_script": "compute_critical_time",
                "sigma_theta": float(sigma),
                "sigma_theta_mode": str(mode),
                "mean_tau_T": float(tT),
                "mean_tau_S": float(tS),
                "mean_tau_C": float(tC),
                "mean_T": float(T),
                "n_c": float(n_c),
                "t_c": float(t_c),
                "formula": "t_c = (2 / sigma_theta^2) * <T>",
            }
            out = write_calibration_section(
                aircraft, "critical_time", payload
            )
            print(f"    wrote {out}  (section: critical_time)")


if __name__ == "__main__":
    main()
