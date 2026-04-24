"""Step-based CTRW model for soaring flights.

The model treats each complete ``transition → search → climb`` cycle as
a single step of a 2-D continuous-time random walk. Horizontal
displacement is dominated by the transition phase, search and climb
contribute negligibly to the *net* horizontal motion (the search phase
spans an effective radius of a few hundred metres with no preferred
direction; the climb is approximately localised up to a small drift).

In the simplest version implemented here, cycle ``n`` produces a
displacement

    Δx_n = v_xy · τ^T_n · ê(θ_n),

with heading θ_n evolving as a Gaussian random walk on the circle,
θ_n = θ_{n-1} + η_n, η_n i.i.d. 𝒩(0, σ²_θ). The total elapsed time
after ``n`` cycles is T_n = Σ_{k≤n} (τ^T_k + τ^S_k + τ^C_k).

This minimal formulation is sufficient to explore how the interplay
between heavy-tailed transition times and angular decorrelation
generates an effective Hurst exponent in the observation window
10¹–10³ s. Extensions (stochastic step-length beyond v_xy·τ, finite
search displacement, wind advection) are deferred to later versions.

References
----------
- Zaburdaev, Denisov, Klafter, Rev. Mod. Phys. 87, 483 (2015) for the
  Lévy-walk / CTRW connection.
- Vilpellet, Darmon, Benzaquen, arXiv:2601.01293 (2026) for the
  empirical statistics used to calibrate the default parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from soaring_ctrw.distributions import (
    Exponential,
    MittagLeffler,
    ParetoTail,
    WaitingTimeSampler,
)

__all__ = [
    "PhaseConfig",
    "AngularConfig",
    "SoaringConfig",
]


@dataclass(frozen=True)
class PhaseConfig:
    """Parameters for a single soaring phase.

    Attributes
    ----------
    distribution : str
        One of ``"pareto"``, ``"mittag_leffler"``, or ``"exponential"``.
    params : dict
        Distribution-specific parameters. For ``pareto``:
        ``{"mu": float, "tau_min": float}``. For ``mittag_leffler``:
        ``{"alpha": float, "tau_0": float}``. For ``exponential``:
        ``{"tau_mean": float}``.

    Note. For :math:`\\alpha > 1` the Mittag-Leffler distribution is
    *not* heavy-tailed; we nonetheless expose it as a primary option
    here because its survival function
    :math:`S(\\tau) = E_\\alpha(-(\\tau/\\tau_0)^\\alpha)` interpolates
    smoothly between a Pareto tail (:math:`\\alpha < 1`) and a
    stretched-exponential regime, and in particular for the empirical
    tail exponents :math:`\\mu_T, \\mu_S > 1` of~\\cite{vilpellet2026}
    the Mittag-Leffler and Pareto distributions share the same
    power-law tail.
    """

    distribution: str
    params: dict[str, float] = field(default_factory=dict)

    def build(self) -> WaitingTimeSampler:
        """Instantiate the sampler described by this config."""
        dist = self.distribution.lower()
        if dist == "pareto":
            return ParetoTail(**self.params)
        if dist == "mittag_leffler":
            # Map the tail-exponent parameter "alpha" of the ML to our
            # sampler, and enforce that the stability index is at most 1.
            # For "alpha > 1" (finite-moment regime, as used empirically
            # here for the cycle-duration statistics), we fall back to
            # a Pareto tail with mu = alpha and tau_min = tau_0 since
            # ML is only defined for 0 < alpha <= 1 in the strict
            # stable-subordinator sense. This keeps the user-facing YAML
            # API consistent with a single "mittag_leffler" name across
            # all phases.
            alpha = float(self.params["alpha"])
            tau_0 = float(self.params["tau_0"])
            if alpha > 1.0:
                return ParetoTail(mu=alpha, tau_min=tau_0)
            return MittagLeffler(alpha=alpha, tau_0=tau_0)
        if dist == "exponential":
            return Exponential(**self.params)
        raise ValueError(
            f"Unknown phase distribution {self.distribution!r}. "
            "Expected one of: 'pareto', 'mittag_leffler', 'exponential'."
        )


@dataclass(frozen=True)
class AngularConfig:
    """Heading-angle dynamics between successive cycles.

    θ_n = θ_{n-1} + η_n,  η_n ~ 𝒩(0, σ²_θ).

    Attributes
    ----------
    sigma_theta : float
        Standard deviation (in radians) of the per-cycle heading
        increment. ``sigma_theta = 0`` corresponds to a perfectly
        persistent walker (always the same direction); very large
        values (σ_θ ≳ π) produce a nearly uniform reshuffling of the
        heading between cycles.
    theta0 : float
        Initial heading (in radians).
    """

    sigma_theta: float
    theta0: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_theta < 0:
            raise ValueError(
                f"sigma_theta must be non-negative, got {self.sigma_theta}"
            )

    @property
    def persistence_cycles(self) -> float:
        r"""Persistence length in cycles: N_p = 2 / σ²_θ.

        Derived from the correlation ⟨cos(θ_j - θ_i)⟩ = exp(-|j-i|σ²_θ/2)
        between headings at different cycles; the correlation drops to
        1/e after N_p = 2/σ²_θ cycles. Returns ``inf`` when σ_θ = 0.
        """
        if self.sigma_theta == 0:
            return float("inf")
        return 2.0 / self.sigma_theta**2


@dataclass(frozen=True)
class SearchMotionConfig:
    """Horizontal motion during the search phase (subordinated Lévy walk).

    Based on Vilpellet et al. (2026, Fig. 3), the conditional MSD of the
    search phase exhibits a ballistic regime at short lags (slope 2) and
    a sub-diffusive tail :math:`\\Delta^{\\alpha_S}` at longer lags with
    :math:`\\alpha_S \\in (0, 1)`.

    We implement this as a **subordinated Lévy walk** (Fogedby 1994;
    Magdziarz-Weron 2007): inside each search phase of total duration
    :math:`\\tau_n^S`, the pilot alternates between:

    1. A ballistic *leg* of duration
       :math:`\\sigma_k \\sim \\mathrm{Exp}(\\sigma_0)` during which
       the walker moves at constant speed :math:`v_c^S` in direction
       :math:`\\psi_k` (angular Gaussian random walk on the circle
       with per-leg increment variance :math:`\\sigma_\\psi^2`).

    2. A *waiting time* of duration
       :math:`\\xi_k \\sim \\mathrm{ML}(\\alpha_S, \\tau_0^S)` during
       which the walker is stationary in the horizontal plane. The
       physical interpretation is that the pilot circles tightly in
       place while evaluating a local updraft candidate; in the
       xy-projection this appears as a pause.

    The two timescales :math:`\\sigma_0` and :math:`\\tau_0^S` separate
    three regimes in the conditional MSD:

    - :math:`\\Delta \\ll \\sigma_0`: ballistic,
      :math:`\\delta^2_S \\simeq v_c^2 \\Delta^2`.
    - :math:`\\sigma_0 \\ll \\Delta \\ll \\tau_0^S`: transient (effective
      diffusive or slightly sub-ballistic).
    - :math:`\\Delta \\gg \\tau_0^S`: sub-diffusive,
      :math:`\\delta^2_S \\propto \\Delta^{\\alpha_S}` by the
      Metzler-Klafter renewal theorem applied to the subordinator.

    Attributes
    ----------
    v_c_S : float
        Walker speed during each ballistic leg (m/s). Typically
        :math:`v_{xy}`.
    sigma_0 : float
        Scale of the exponential distribution of leg durations
        :math:`\\sigma_k` (s). Sets the end of the ballistic regime.
    tau_0_S : float
        Scale of the Mittag-Leffler distribution of intra-phase
        waiting times :math:`\\xi_k` (s) — the symbol
        :math:`\\tau_w^S` of the PRL companion paper. Sets the onset
        of the sub-diffusive regime. Not to be confused with the
        Pareto scale :math:`\\tau_{\\min}^S` of the *cycle-duration*
        scheduler in the ``search`` :class:`PhaseConfig`, which is
        unrelated.
    alpha_S : float
        Stability index of the waiting-time ML distribution, in (0, 1).
        Matches the empirical asymptotic sub-diffusive exponent
        (:math:`\\alpha_S = 0.6` for paragliders/hang gliders,
        :math:`\\alpha_S = 0.4` for sailplanes, per Fig. 3 of
        Vilpellet et al.).
    sigma_psi_S : float
        Per-leg Gaussian heading-increment standard deviation (rad).

    ASSUMPTION. The initial leg heading :math:`\\psi_0` is uniform on
    :math:`\\mathbb{S}^1` and independent of the transition heading
    :math:`\\theta_n`.
    """

    v_c_S: float
    sigma_0: float
    tau_0_S: float
    alpha_S: float = 0.6
    sigma_psi_S: float = 0.5

    def __post_init__(self) -> None:
        if self.v_c_S < 0:
            raise ValueError(f"v_c_S must be non-negative, got {self.v_c_S}")
        if self.sigma_0 <= 0:
            raise ValueError(f"sigma_0 must be positive, got {self.sigma_0}")
        if self.tau_0_S <= 0:
            raise ValueError(f"tau_0_S must be positive, got {self.tau_0_S}")
        if not (0.0 < self.alpha_S < 1.0):
            raise ValueError(f"alpha_S must be in (0, 1), got {self.alpha_S}")
        if self.sigma_psi_S < 0:
            raise ValueError(f"sigma_psi_S must be non-negative, got {self.sigma_psi_S}")


@dataclass(frozen=True)
class ClimbMotionConfig:
    """Horizontal motion during the climb phase (harmonic oscillator + drift).

    Per Vilpellet et al. (2026, Fig. 3), the conditional MSD of the
    climb phase shows three regimes:

    - :math:`\\Delta \\lesssim 1/\\omega_0`: ballistic (tangent speed of
      the thermalling turn).
    - :math:`\\Delta \\sim \\pi/\\omega_0`: anti-correlation due to the
      pilot returning to near the starting point (particularly visible
      for sailplanes with large thermalling radius).
    - :math:`\\Delta \\gg 1/\\omega_0`: quasi-ballistic tail from the
      orographic drift.

    We reproduce this by modelling the climb trajectory as a 2-D
    harmonic oscillator (pilot circling around a thermal centre) plus
    a linear drift (orographic tilt):

    .. math::
        \\mathbf{x}^C_n(c) = r_0 \\bigl[
            \\cos(\\omega_0 c + \\phi_0) - \\cos\\phi_0,\\;
            \\sin(\\omega_0 c + \\phi_0) - \\sin\\phi_0
        \\bigr]
        + v_{\\mathrm{drift}}\\,c\\,
        \\hat{\\mathbf{e}}(\\phi_n^{\\mathrm{drift}}),

    where :math:`\\omega_0 = 2\\pi / T_{\\mathrm{turn}}` is the angular
    frequency of thermalling, :math:`r_0` is the thermalling radius
    (Fig. 4h of~\\cite{vilpellet2026}), :math:`\\phi_0` is a random
    initial phase uniform on :math:`[0, 2\\pi)` for each cycle, and
    :math:`\\phi_n^{\\mathrm{drift}}` is a drift direction drawn
    i.i.d. uniform per cycle from :math:`\\mathbb{S}^1`.

    After ensemble-averaging over :math:`\\phi_0` (uniform on the
    circle) and the per-cycle turn period (Gaussian of mean
    :math:`\\bar T_{\\mathrm{turn}}` and standard deviation
    :math:`\\sigma_T`, positivity-clipped), the conditional MSD is

    .. math::
        \\delta^2_C(\\Delta) = 2 r_0^2 \\!\\left[1
            - \\mathrm{e}^{-\\sigma_T^2 \\bar\\omega_0^2 \\Delta^2/2}
              \\cos(\\bar\\omega_0 \\Delta)\\right]
        + v_{\\mathrm{drift}}^2 \\Delta^2,

    with :math:`\\bar\\omega_0 = 2\\pi/\\bar T_{\\mathrm{turn}}`. The
    Gaussian damping factor smears out the anti-correlation dip when
    :math:`\\sigma_T \\bar\\omega_0 \\gtrsim 1` (paragliders, hang
    gliders); the no-dispersion limit
    :math:`\\sigma_T \\to 0` recovers
    :math:`2 r_0^2 [1 - \\cos(\\bar\\omega_0\\Delta)] +
    v_{\\mathrm{drift}}^2 \\Delta^2` and keeps the marked dip at
    :math:`\\Delta \\approx \\pi/\\bar\\omega_0` used for sailplanes.

    Attributes
    ----------
    thermal_radius : float
        Radius of the thermalling circle, :math:`r_0` (m). Empirically
        39, 58, 130 m for paragliders, hang gliders, sailplanes
        (Fig. 4h of Vilpellet et al.).
    turn_period : float
        Mean thermalling turn period :math:`T_{\\mathrm{turn}} =
        2\\pi/\\omega_0` (s). Universal mean value :math:`\\approx 30`
        s (§III of Vilpellet et al.).
    turn_period_std : float
        Standard deviation of the per-cycle turn period (s). Modelling
        the physiological and atmospheric dispersion of circling
        speeds across pilots and cycles, this is the parameter that
        smooths the ensemble-averaged conditional MSD of the climb
        phase (otherwise the coherent single-frequency oscillations
        of :math:`\\delta^2_C` survive ensemble averaging and create
        unphysical periodic dips not seen in Vilpellet, Fig. 3).
        Default: 5 s.
    v_drift : float
        Magnitude of the orographic drift (m/s), typically
        :math:`v_{xy}/100`.

    ASSUMPTION. The drift direction is independent across cycles and
    independent of the transition heading.
    """

    thermal_radius: float
    turn_period: float = 30.0
    turn_period_std: float = 5.0
    v_drift: float = 0.0

    def __post_init__(self) -> None:
        if self.thermal_radius < 0:
            raise ValueError(f"thermal_radius must be non-negative, got {self.thermal_radius}")
        if self.turn_period <= 0:
            raise ValueError(f"turn_period must be positive, got {self.turn_period}")
        if self.turn_period_std < 0:
            raise ValueError(f"turn_period_std must be non-negative, got {self.turn_period_std}")
        if self.v_drift < 0:
            raise ValueError(f"v_drift must be non-negative, got {self.v_drift}")


@dataclass(frozen=True)
class SoaringConfig:
    """Complete specification of a step-based CTRW soaring model.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"paragliders"``).
    v_xy : float
        Characteristic horizontal velocity during the transition phase
        (m/s). Taken as the mean value reported in Fig. 4b of the paper.
    transition, search, climb : PhaseConfig
        Duration distributions for each of the three phases.
    angular : AngularConfig
        Heading-angle dynamics between successive cycles.
    search_motion : SearchMotionConfig, optional
        Intra-phase horizontal motion during search (subordinated Lévy
        walk). If ``None``, the search phase is held horizontally
        stationary (minimal baseline). Defaults to ``None`` for
        backward compatibility.
    climb_motion : ClimbMotionConfig, optional
        Intra-phase horizontal motion during climb (2-D harmonic
        oscillator with Gaussian-damped period plus linear drift).
        If ``None``, the climb phase is held horizontally stationary.
        Defaults to ``None`` for backward compatibility.
    """

    name: str
    v_xy: float
    transition: PhaseConfig
    search: PhaseConfig
    climb: PhaseConfig
    angular: AngularConfig
    search_motion: SearchMotionConfig | None = None
    climb_motion: ClimbMotionConfig | None = None

    def __post_init__(self) -> None:
        if self.v_xy <= 0:
            raise ValueError(f"v_xy must be positive, got {self.v_xy}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SoaringConfig":
        """Load a configuration from a YAML file.

        The ``search_motion`` and ``climb_motion`` blocks are optional;
        their absence holds the corresponding phase horizontally
        stationary (minimal baseline).
        """
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)

        search_motion = (
            SearchMotionConfig(**data["search_motion"])
            if "search_motion" in data and data["search_motion"] is not None
            else None
        )
        climb_motion = (
            ClimbMotionConfig(**data["climb_motion"])
            if "climb_motion" in data and data["climb_motion"] is not None
            else None
        )
        return cls(
            name=data["name"],
            v_xy=float(data["v_xy"]),
            transition=PhaseConfig(**data["transition"]),
            search=PhaseConfig(**data["search"]),
            climb=PhaseConfig(**data["climb"]),
            angular=AngularConfig(**data["angular"]),
            search_motion=search_motion,
            climb_motion=climb_motion,
        )
