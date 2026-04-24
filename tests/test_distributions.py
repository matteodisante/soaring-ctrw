"""Tests for heavy-tailed waiting-time samplers."""

from __future__ import annotations

import numpy as np
import pytest

from soaring_ctrw.distributions import Exponential, ParetoTail


class TestParetoTail:
    def test_invalid_mu_rejected(self):
        with pytest.raises(ValueError):
            ParetoTail(mu=-1.0, tau_min=1.0)

    def test_invalid_tau_min_rejected(self):
        with pytest.raises(ValueError):
            ParetoTail(mu=2.0, tau_min=0.0)

    def test_mean_infinite_below_critical_mu(self):
        d = ParetoTail(mu=0.9, tau_min=1.0)
        assert d.mean == float("inf")

    def test_mean_finite_above_critical_mu(self):
        d = ParetoTail(mu=3.0, tau_min=2.0)
        # m = tau_min * mu / (mu - 1) = 2 * 3 / 2 = 3
        assert d.mean == pytest.approx(3.0)

    def test_variance_infinite_when_mu_leq_2(self):
        d = ParetoTail(mu=1.8, tau_min=1.0)
        assert d.variance == float("inf")

    def test_empirical_tail_matches_theory(self):
        """Survival S(tau) ~ (tau_min / tau)^mu at large tau."""
        mu, tau_min = 3.5, 10.0
        d = ParetoTail(mu=mu, tau_min=tau_min)
        rng = np.random.default_rng(0)
        samples = d.sample(size=200_000, rng=rng)

        # empirical survival at a reference level
        tau_ref = 50.0
        empirical_survival = (samples > tau_ref).mean()
        theoretical_survival = (tau_min / tau_ref) ** mu
        assert empirical_survival == pytest.approx(
            theoretical_survival, rel=0.05
        )

    def test_lower_support(self):
        d = ParetoTail(mu=2.5, tau_min=7.0)
        rng = np.random.default_rng(0)
        samples = d.sample(size=10_000, rng=rng)
        assert samples.min() >= 7.0 - 1e-9


class TestExponential:
    def test_invalid_mean_rejected(self):
        with pytest.raises(ValueError):
            Exponential(tau_mean=-1.0)

    def test_empirical_mean(self):
        d = Exponential(tau_mean=42.0)
        rng = np.random.default_rng(0)
        samples = d.sample(size=200_000, rng=rng)
        assert samples.mean() == pytest.approx(42.0, rel=0.02)

    def test_analytical_moments(self):
        d = Exponential(tau_mean=7.0)
        assert d.mean == 7.0
        assert d.variance == pytest.approx(49.0)
