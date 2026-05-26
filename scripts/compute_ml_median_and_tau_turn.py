"""Compute the dimensionless median ``m_{1/2}(alpha)`` of the
Mittag-Leffler waiting-time distribution and derive ``tau_turn^S``.

The Mittag-Leffler survival in the manuscript reads
``S(t) = E_alpha[-(t / tau_turn^S)^alpha]`` (Pillai-ML convention).
Its median ``t_{1/2}`` satisfies ``E_alpha(-m_{1/2}^alpha) = 1/2``;
the dimensionless median ``m_{1/2}(alpha) = t_{1/2} / tau_turn^S``
is the unique solution.

The manuscript calibrates ``tau_turn^S`` by associating the median
turning event with a 90 deg reorientation:
``t_{1/2} = (pi / 2) / Omega_S``. Hence

    tau_turn^S = (pi / 2) / (Omega_S * m_{1/2}(alpha_S)).

By default this script reads ``alpha_S`` and ``Omega_S`` from the
YAML configs under ``configs/`` (one per aircraft) and prints both
the calibrated ``tau_turn^S`` and the value currently stored in the
YAML so any mismatch is immediately visible. Per-aircraft overrides
are available via ``--alpha`` / ``--omega`` for debugging.

Requires ``pymittagleffler``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from pymittagleffler import mittag_leffler
from scipy.optimize import root_scalar


from soaring_ctrw.paths import CONFIGS_DIR  # noqa: E402
from soaring_ctrw.calibration import write_calibration_section  # noqa: E402

DEFAULT_AIRCRAFT = ("paragliders", "hang_gliders", "sailplanes")


def median_mittag_leffler(alpha: float) -> float:
    """Return the dimensionless median ``m_{1/2}(alpha)``."""

    def f(m: float) -> float:
        return float(np.real(mittag_leffler(-(m ** alpha), alpha, 1.0))) - 0.5

    sol = root_scalar(f, bracket=(1e-6, 50.0))
    if not sol.converged:
        raise RuntimeError(f"Did not converge for alpha={alpha}")
    return float(sol.root)


def tau_turn(Omega_S: float, alpha_S: float) -> float:
    """``tau_turn^S = (pi / 2) / (Omega_S * m_{1/2}(alpha_S))``."""
    m = median_mittag_leffler(alpha_S)
    return float(np.pi / (2.0 * Omega_S * m))


def _load_search_motion(aircraft: str) -> tuple[float, float, float]:
    """Read (alpha_S, Omega_S, tau_turn_S) from ``configs/<aircraft>.yaml``."""
    cfg_path = CONFIGS_DIR / f"{aircraft}.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)
    try:
        sm = cfg["search_motion"]
        return (
            float(sm["alpha_S"]),
            float(sm["Omega_S"]),
            float(sm["tau_turn_S"]),
        )
    except (KeyError, TypeError) as exc:
        raise SystemExit(
            f"{cfg_path}: missing search_motion.alpha_S / Omega_S / tau_turn_S"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft",
        nargs="+",
        default=list(DEFAULT_AIRCRAFT),
        help="Aircraft config names (without .yaml) under configs/.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help="Override alpha_S from the YAML (one per --aircraft).",
    )
    parser.add_argument(
        "--omega",
        type=float,
        nargs="+",
        default=None,
        help="Override Omega_S from the YAML (one per --aircraft).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Also merge the results into "
            "outputs/data/calibration/<aircraft>.yaml under the "
            "`mittag_leffler` section."
        ),
    )
    args = parser.parse_args()

    n = len(args.aircraft)
    if args.alpha is not None and len(args.alpha) != n:
        raise SystemExit(
            f"--alpha must have one value per --aircraft "
            f"(got {len(args.alpha)} vs {n})"
        )
    if args.omega is not None and len(args.omega) != n:
        raise SystemExit(
            f"--omega must have one value per --aircraft "
            f"(got {len(args.omega)} vs {n})"
        )

    print(
        f"{'aircraft':<14} {'alpha_S':>8} {'m_{1/2}':>10} "
        f"{'Omega_S':>10} {'tau_turn^S':>12} {'YAML':>10}"
    )
    for i, aircraft in enumerate(args.aircraft):
        alpha_cfg, omega_cfg, tau_cfg = _load_search_motion(aircraft)
        alpha = args.alpha[i] if args.alpha is not None else alpha_cfg
        omega = args.omega[i] if args.omega is not None else omega_cfg
        m = median_mittag_leffler(alpha)
        tt = tau_turn(Omega_S=omega, alpha_S=alpha)
        print(
            f"{aircraft:<14} {alpha:>8.4f} {m:>10.6f} "
            f"{omega:>10.4f} {tt:>12.4f} {tau_cfg:>10.4f}"
        )
        if args.write:
            payload = {
                "source_script": "compute_ml_median_and_tau_turn",
                "alpha_S": float(alpha),
                "Omega_S": float(omega),
                "m_half_alpha": float(m),
                "tau_turn_calibrated": float(tt),
                "tau_turn_yaml": float(tau_cfg),
                "formula": "(pi/2) / (Omega_S * m_{1/2}(alpha_S))",
            }
            out = write_calibration_section(aircraft, "mittag_leffler", payload)
            print(f"    wrote {out}  (section: mittag_leffler)")


if __name__ == "__main__":
    main()
