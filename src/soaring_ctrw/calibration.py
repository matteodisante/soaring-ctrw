"""Per-aircraft calibration files.

Some model parameters are not fixed by the YAML configuration but
inferred from empirical observables. The most important is the
cycle-to-cycle heading dispersion ``sigma_theta``, calibrated by
``scripts/estimate_sigma_theta.py`` against the empirical
``H_eff = 0.88``. Less centrally, the search-phase scales
(``tau_turn_calibrated``, ``u_S_fitted``) produced by
``scripts/calibrate_search.py`` and the Mittag-Leffler median
produced by ``scripts/compute_ml_median_and_tau_turn.py`` also live
here.

Each aircraft has one YAML file under
``outputs/data/calibration/<aircraft>.yaml`` with one section per
calibration script::

    aircraft: paragliders
    sigma_theta:
      source_script: estimate_sigma_theta
      value: 0.346
      mode: full
      by_mode:
        bare: 0.302
        full: 0.346
      H_target: 0.88
      fit_window: [10.0, 6000.0]
      n_trajectories: 1000
      total_time: 15000.0
    search:
      source_script: calibrate_search
      m_half_alpha: 0.6789
      tau_turn_calibrated: 9.6427
      u_S: 35.0
      u_S_fitted: 39.0555
      ...
    mittag_leffler:
      source_script: compute_ml_median_and_tau_turn
      alpha_S: 0.7
      m_half_alpha: 0.6789
      tau_turn_calibrated: 9.6427

Sections are independent: re-running one calibration script merges
its section into the file without touching the others. The downstream
simulation scripts read ``sigma_theta`` (and only ``sigma_theta``) via
:func:`apply_calibration`, which substitutes it into the
:class:`AngularConfig` of a loaded :class:`SoaringConfig`.

The ``sigma_theta`` value in the input ``configs/<aircraft>.yaml`` is
treated as a placeholder and ignored: simulation scripts that need
``sigma_theta`` must read it from the calibration file via this
module.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .model import AngularConfig, SoaringConfig
from .paths import CONFIGS_DIR, OUTPUTS_DIR

__all__ = [
    "CALIBRATION_DIR",
    "apply_calibration",
    "calibrated_sigma_theta",
    "calibration_path",
    "load_calibrated_config",
    "read_calibration",
    "write_calibration_section",
]

CALIBRATION_DIR = OUTPUTS_DIR / "data" / "calibration"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def calibration_path(aircraft: str) -> Path:
    """Path of the YAML calibration file for ``aircraft``."""
    return CALIBRATION_DIR / f"{aircraft}.yaml"


def read_calibration(aircraft: str) -> dict[str, Any]:
    """Read the calibration YAML for ``aircraft``. Returns ``{}`` if
    the file does not exist."""
    p = calibration_path(aircraft)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text())
    return data or {}


def write_calibration_section(
    aircraft: str,
    section: str,
    payload: dict[str, Any],
) -> Path:
    """Merge ``payload`` into ``<aircraft>.yaml`` under ``section``.

    All other sections of the file are preserved. The write is atomic
    (temp file + rename) so a crash mid-write cannot corrupt an
    existing calibration.
    """
    p = calibration_path(aircraft)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = read_calibration(aircraft)
    existing["aircraft"] = aircraft
    existing[section] = _normalize(payload)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(existing, sort_keys=False, default_flow_style=False)
    )
    os.replace(tmp, p)
    return p


# ---------------------------------------------------------------------------
# Readers + overrides
# ---------------------------------------------------------------------------


def calibrated_sigma_theta(aircraft: str) -> float:
    """Return the calibrated ``sigma_theta`` for ``aircraft``.

    Raises ``FileNotFoundError`` with an actionable message if the
    calibration section is missing.
    """
    cal = read_calibration(aircraft)
    section = cal.get("sigma_theta")
    if not section or "value" not in section:
        raise FileNotFoundError(
            f"No sigma_theta calibration for {aircraft!r} at "
            f"{calibration_path(aircraft)}. Run "
            f"`python scripts/estimate_sigma_theta.py --aircraft {aircraft} "
            f"--write` to produce it."
        )
    return float(section["value"])


def apply_calibration(config: SoaringConfig) -> SoaringConfig:
    """Return a copy of ``config`` with ``angular.sigma_theta`` replaced
    by the calibrated value stored in
    ``outputs/data/calibration/<config.name>.yaml``.

    ``config.name`` is the aircraft key (e.g. ``"paragliders"``).
    """
    sigma = calibrated_sigma_theta(config.name)
    new_angular = AngularConfig(
        sigma_theta=sigma, theta0=config.angular.theta0,
    )
    return dataclasses.replace(config, angular=new_angular)


def load_calibrated_config(
    aircraft: str,
    configs_dir: Path | None = None,
) -> SoaringConfig:
    """Load ``configs/<aircraft>.yaml`` and apply the
    ``sigma_theta`` calibration in one step.

    Equivalent to::

        apply_calibration(SoaringConfig.from_yaml(...))
    """
    base = configs_dir or CONFIGS_DIR
    config = SoaringConfig.from_yaml(base / f"{aircraft}.yaml")
    return apply_calibration(config)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _normalize(obj: Any) -> Any:
    """Coerce numpy and Path types to plain Python so YAML is
    human-readable (no ``!!python/object`` tags)."""
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
