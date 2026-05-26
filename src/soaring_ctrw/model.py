"""Configuration objects for the cycle-based CTRW soaring-flight model.

This module exposes only the *parameter containers* of the model.
The actual stochastic dynamics live in :mod:`simulation`.

Naming convention follows the companion manuscript (Di Sante, 2026):

  - ``v_xy``                 horizontal cruise speed during the transition.
  - ``sigma_theta``          cycle-to-cycle heading dispersion (rad).
  - ``u_S``                  speed during a search ballistic leg.
  - ``tau_b_S``              mean exponential duration of a search ballistic leg.
  - ``tau_turn_S``           scale of the Mittag-Leffler search turning time.
  - ``Omega_S``              angular velocity used for the direction update
                             ``psi_{j+1} = psi_j + eps_j * Omega_S * tau_turn_j``.
  - ``alpha_S``              Mittag-Leffler stability index.
  - ``r0``                   thermalling radius during the climb phase.
  - ``T_turn_mean``          mean climb turn period (clipped Gaussian).
  - ``T_turn_std``           standard deviation of the climb turn period.
  - ``v_drift``              orographic drift speed during the climb.

Phase-duration distributions for transition and search are Lomax
survivals ``S(tau) = (1 + tau/tau_0)^{-mu}``; the climb is exponential
with a common ``mu_C_eff`` across aircraft (Table 1 of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .distributions import (
    Exponential,
    LomaxTail,
    MittagLeffler,
    ParetoTail,
    WaitingTimeSampler,
)

__all__ = [
    "PhaseConfig",
    "AngularConfig",
    "SearchMotionConfig",
    "ClimbMotionConfig",
    "SoaringConfig",
]


@dataclass(frozen=True)
class PhaseConfig:
    """Duration scheduler for a single soaring phase.

    Supported distributions: ``"lomax"``, ``"pareto"``, ``"mittag_leffler"``,
    ``"exponential"``. Parameters required by each are documented on the
    corresponding sampler class in :mod:`distributions`.
    """

    distribution: str
    params: dict[str, float] = field(default_factory=dict)

    def build(self) -> WaitingTimeSampler:
        dist = self.distribution.lower()
        if dist == "lomax":
            return LomaxTail(**self.params)
        if dist == "pareto":
            return ParetoTail(**self.params)
        if dist == "mittag_leffler":
            return MittagLeffler(**self.params)
        if dist == "exponential":
            return Exponential(**self.params)
        raise ValueError(
            f"Unknown phase distribution {self.distribution!r}. "
            "Expected: 'lomax', 'pareto', 'mittag_leffler', 'exponential'."
        )


@dataclass(frozen=True)
class AngularConfig:
    r"""Heading-angle dynamics between successive cycles.

    ``theta_n = theta_{n-1} + eta_n``,
    ``eta_n ~ N(0, sigma_theta^2)``.

    ``theta0`` is the heading of the very first transition. If ``None``
    (the default), the simulator draws it uniformly on ``[0, 2*pi)``,
    matching the assumption of the paper that the initial heading of
    each independent trajectory is isotropic.
    """

    sigma_theta: float
    theta0: float | None = None

    def __post_init__(self) -> None:
        if self.sigma_theta < 0:
            raise ValueError(
                f"sigma_theta must be non-negative, got {self.sigma_theta}"
            )

    @property
    def persistence_cycles(self) -> float:
        r"""Directional-memory length in cycles, ``n_c = 2 / sigma_theta^2``."""
        if self.sigma_theta == 0:
            return float("inf")
        return 2.0 / self.sigma_theta**2


@dataclass(frozen=True)
class SearchMotionConfig:
    r"""Intra-search local CTRW with physical-duration stopping.

    Inside one search episode the pilot alternates ballistic relocations
    (speed ``u_S``, exponentially distributed durations of mean
    ``tau_b_S``) with compact reorientation manoeuvres modelled as
    Mittag-Leffler turning waits (stability ``alpha_S``, scale
    ``tau_turn_S``). The direction is updated, after a fully completed
    wait, as

    .. math::
        \psi_{j+1} = \psi_j + \epsilon_j\, \Omega_S\, \tau^{\mathrm{turn}}_j,
        \qquad \epsilon_j = \pm 1\ \text{equiprobable}.

    The local CTRW is stopped when the cumulative *physical* time
    (legs + waits) reaches the Lomax-sampled search duration
    ``tau_S_n``. Whichever component straddles ``tau_S_n`` is truncated
    so that ``T_phys^S = sum(tau_b) + sum(tau_turn) = tau_S_n`` exactly.
    This keeps the physical search duration finite even though the
    Mittag-Leffler waits have infinite mean for ``alpha_S < 1``.
    """

    u_S: float
    tau_b_S: float
    tau_turn_S: float
    alpha_S: float
    Omega_S: float

    def __post_init__(self) -> None:
        if self.u_S < 0:
            raise ValueError(f"u_S must be non-negative, got {self.u_S}")
        if self.tau_b_S <= 0:
            raise ValueError(f"tau_b_S must be positive, got {self.tau_b_S}")
        if self.tau_turn_S <= 0:
            raise ValueError(f"tau_turn_S must be positive, got {self.tau_turn_S}")
        if not (0.0 < self.alpha_S < 1.0):
            raise ValueError(f"alpha_S must be in (0, 1), got {self.alpha_S}")
        if self.Omega_S < 0:
            raise ValueError(f"Omega_S must be non-negative, got {self.Omega_S}")


@dataclass(frozen=True)
class ClimbMotionConfig:
    r"""Horizontal motion during the climb phase.

    The pilot circles a thermal core of radius ``r0`` with angular
    frequency ``omega_n = 2*pi / T_turn_n``, where the turn period is
    drawn cycle-by-cycle from a Gaussian ``N(T_turn_mean, T_turn_std^2)``
    and clipped at ``0.2 * T_turn_mean`` for positivity. A slow linear
    orographic drift of magnitude ``v_drift`` is superposed; its
    direction is sampled uniformly on ``[0, 2*pi)`` independently per
    cycle.
    """

    r0: float
    T_turn_mean: float
    T_turn_std: float
    v_drift: float

    def __post_init__(self) -> None:
        if self.r0 < 0:
            raise ValueError(f"r0 must be non-negative, got {self.r0}")
        if self.T_turn_mean <= 0:
            raise ValueError(f"T_turn_mean must be positive, got {self.T_turn_mean}")
        if self.T_turn_std < 0:
            raise ValueError(f"T_turn_std must be non-negative, got {self.T_turn_std}")
        if self.v_drift < 0:
            raise ValueError(f"v_drift must be non-negative, got {self.v_drift}")


@dataclass(frozen=True)
class SoaringConfig:
    """Top-level container for one aircraft class.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``"paragliders"``).
    v_xy : float
        Characteristic horizontal speed during transitions (m/s).
    transition, search, climb : PhaseConfig
        Phase-duration scheduler.
    angular : AngularConfig
        Inter-cycle heading dynamics.
    search_motion : SearchMotionConfig
        Local CTRW parameters for the search phase.
    climb_motion : ClimbMotionConfig
        Circular motion + drift parameters for the climb phase.
    """

    name: str
    v_xy: float
    transition: PhaseConfig
    search: PhaseConfig
    climb: PhaseConfig
    angular: AngularConfig
    # The intra-phase dynamics are optional; when ``None`` the
    # corresponding phase contributes no horizontal displacement and the
    # cycle clock advances by the Lomax/exponential duration draw. This
    # is the "bare-cycle" variant, in which only the transition phase
    # carries displacement.
    search_motion: SearchMotionConfig | None = None
    climb_motion: ClimbMotionConfig | None = None

    def __post_init__(self) -> None:
        if self.v_xy <= 0:
            raise ValueError(f"v_xy must be positive, got {self.v_xy}")

    def bare(self) -> "SoaringConfig":
        """Return a copy with search_motion and climb_motion stripped.

        In the bare-cycle variant only the transition phase carries a
        non-trivial heading-correlated displacement; search and climb
        consume their sampled durations but contribute no horizontal
        motion.
        """
        return SoaringConfig(
            name=self.name + "_bare",
            v_xy=self.v_xy,
            transition=self.transition,
            search=self.search,
            climb=self.climb,
            angular=self.angular,
            search_motion=None,
            climb_motion=None,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SoaringConfig":
        """Load a configuration from a YAML file (see ``configs/*.yaml``)."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)

        sm = data.get("search_motion")
        cm = data.get("climb_motion")
        angular_data = dict(data["angular"])
        if angular_data.get("sigma_theta") is None:
            # sigma_theta is set by scripts/estimate_sigma_theta.py (step 6
            # of the pipeline); upstream steps that only need per-phase
            # observables can run with a placeholder value.
            angular_data["sigma_theta"] = 0.0
        return cls(
            name=data["name"],
            v_xy=float(data["v_xy"]),
            transition=PhaseConfig(**data["transition"]),
            search=PhaseConfig(**data["search"]),
            climb=PhaseConfig(**data["climb"]),
            angular=AngularConfig(**angular_data),
            search_motion=SearchMotionConfig(**sm) if sm else None,
            climb_motion=ClimbMotionConfig(**cm) if cm else None,
        )
