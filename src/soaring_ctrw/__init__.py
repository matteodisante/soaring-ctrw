"""Step-based CTRW model with angular persistence for soaring flights.

This package implements the minimal model described in the accompanying
manuscript and in docs/model.md. See the README for a quickstart and the
module-level docstrings for API details.
"""

from soaring_ctrw.model import (
    SoaringConfig,
    PhaseConfig,
    AngularConfig,
    SearchMotionConfig,
    ClimbMotionConfig,
)
from soaring_ctrw.simulation import simulate_ensemble, simulate_single
from soaring_ctrw.observables import msd_time_averaged, fit_hurst

__version__ = "0.6.0"
__all__ = [
    "SoaringConfig",
    "PhaseConfig",
    "AngularConfig",
    "SearchMotionConfig",
    "ClimbMotionConfig",
    "simulate_ensemble",
    "simulate_single",
    "msd_time_averaged",
    "fit_hurst",
]
