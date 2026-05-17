"""Samplers for heavy-tailed and thin-tailed waiting-time distributions.

The paper's Fig. 4c,f,i reports empirical tail exponents μ for the
survival function S(τ) = P(T > τ) ~ τ^{-μ} of the three phase durations.
Transition and search phases are well fitted by Pareto-like tails, while
climb phases are consistent with exponential distributions (Fig. 4i).

We provide minimal, explicitly tested samplers for these two families.
More general distributions (Mittag-Leffler, truncated Pareto with
arbitrary lower cutoff) can be added when needed.

Notation
--------
Throughout, we follow the convention of Vilpellet et al.:
    P(T > τ) ~ 1/τ^μ
for the Pareto tail, so that μ is the *tail exponent* (not the shape in
some other conventions). In scipy.stats.pareto, this corresponds to
``b = μ`` and ``scale = τ_min``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "ParetoTail",
    "Exponential",
    "MittagLeffler",
    "WaitingTimeSampler",
]


class WaitingTimeSampler:
    """Abstract base class for a non-negative random-variable sampler.

    Concrete subclasses implement ``sample`` and expose the theoretical
    mean and variance (``np.inf`` when undefined).
    """

    mean: float
    variance: float

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class ParetoTail(WaitingTimeSampler):
    r"""Pareto distribution with tail exponent ``mu`` and lower cutoff ``tau_min``.

    Density:
        f(τ) = μ · τ_min^μ · τ^{-(μ+1)},  τ ≥ τ_min.
    Survival:
        S(τ) = (τ_min / τ)^μ.

    The mean exists only for μ > 1 (``τ_min · μ / (μ−1)``) and the
    variance only for μ > 2.

    Parameters
    ----------
    mu : float
        Tail exponent. Must be positive.
    tau_min : float
        Lower cutoff. Must be positive.
    """

    mu: float
    tau_min: float

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.tau_min <= 0:
            raise ValueError(f"tau_min must be positive, got {self.tau_min}")

    @property
    def mean(self) -> float:
        if self.mu <= 1:
            return float("inf")
        return self.tau_min * self.mu / (self.mu - 1)

    @property
    def variance(self) -> float:
        if self.mu <= 2:
            return float("inf")
        m = self.mean
        second = self.tau_min**2 * self.mu / (self.mu - 2)
        return second - m**2

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` i.i.d. samples."""
        return stats.pareto.rvs(
            b=self.mu, scale=self.tau_min, size=size, random_state=rng
        )


@dataclass(frozen=True)
class Exponential(WaitingTimeSampler):
    r"""Exponential distribution with mean ``tau_mean``.

    Density:
        f(τ) = (1/τ̄) · exp(-τ/τ̄),  τ ≥ 0.

    Used for the climb phase, whose empirical survival function in
    Fig. 4i is consistent with an exponential (thin) tail.
    """

    tau_mean: float

    def __post_init__(self) -> None:
        if self.tau_mean <= 0:
            raise ValueError(f"tau_mean must be positive, got {self.tau_mean}")

    @property
    def mean(self) -> float:
        return self.tau_mean

    @property
    def variance(self) -> float:
        return self.tau_mean**2

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(scale=self.tau_mean, size=size)


@dataclass(frozen=True)
class MittagLeffler(WaitingTimeSampler):
    r"""Mittag-Leffler distribution with stability index ``alpha`` and scale ``tau_0``.

    The Mittag-Leffler (ML) distribution is the canonical heavy-tailed
    waiting-time distribution of continuous-time random walks [Metzler,
    Klafter 2000; Scalas 2004]. Its survival function is

        S(τ) = E_α(-(τ/τ_0)^α),

    where E_α is the one-parameter Mittag-Leffler function. For
    ``alpha ∈ (0, 1)`` the tail decays as

        S(τ) ~ (τ/τ_0)^{-α} / Γ(1-α),  τ → ∞,

    so the tail survival exponent is ``alpha`` (same convention as our
    ``ParetoTail.mu``). All moments diverge for ``0 < alpha < 1``. For
    ``alpha = 1`` the ML reduces to the exponential with mean ``tau_0``.

    Sampling
    --------
    We use the Kozubowski-Rachev (2001) one-line form: if ``U`` is
    uniform on ``(0, π)`` and ``V`` uniform on ``(0, 1)`` are
    independent, then

        τ = τ_0 · sin(α U) · [sin((1 − α) U) / (− log V)]^{(1−α)/α}
              / sin(U)^{1/α}

    is distributed as the stable subordinator evaluated at unit time
    with stability index α (equivalently, ``τ = τ_0 · |W|^{1/α}``
    with W a positive α-stable random variable). This form is robust
    across the full ``α ∈ (0, 1)`` range; for ``α = 1`` it reduces to
    the exponential of mean ``τ_0``.

    Parameters
    ----------
    alpha : float
        Stability index in (0, 1].
    tau_0 : float
        Scale parameter (>0). Sets the crossover from small- to
        large-τ regimes.
    """

    alpha: float
    tau_0: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.tau_0 <= 0:
            raise ValueError(f"tau_0 must be positive, got {self.tau_0}")

    @property
    def mean(self) -> float:
        if self.alpha < 1.0:
            return float("inf")
        return self.tau_0

    @property
    def variance(self) -> float:
        if self.alpha < 1.0:
            return float("inf")
        return self.tau_0**2

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        r"""Sample ``size`` i.i.d. Mittag-Leffler waiting times.

        For α = 1, returns exponential samples. For α < 1, uses the
        Kozubowski-Rachev formula [Kozubowski 2001, Eq. 4.4]:

            τ = τ_0 · sin(α π U) · [sin((1−α) π U) / (− log V)]^{(1−α)/α}
                  / sin(π U)^{1/α}

        where U ~ Uniform(0, π) and V ~ Uniform(0, 1) are independent.
        This form is robust across the full α ∈ (0, 1) range.
        """
        if self.alpha == 1.0:
            return rng.exponential(scale=self.tau_0, size=size)

        U = rng.uniform(0.0, np.pi, size=size)
        V = rng.uniform(0.0, 1.0, size=size)
        alpha = self.alpha

        # Kozubowski-Rachev
        num = np.sin(alpha * U) * (
            np.sin((1.0 - alpha) * U) / (-np.log(V))
        ) ** ((1.0 - alpha) / alpha)
        den = np.sin(U) ** (1.0 / alpha)
        return self.tau_0 * num / den
