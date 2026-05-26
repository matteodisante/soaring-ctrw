"""Tests for MSD computation and Hurst-exponent fitting."""

from __future__ import annotations

import numpy as np
import pytest
from soaring_ctrw.observables import (
    fit_hurst,
    msd_ensemble,
)


class TestMSDEnsemble:
    """``msd_ensemble`` is the pure EA-MSD:
    ``out[k] = mean_m |r_m(k) - r_m(0)|^2``.
    """

    def test_shape(self):
        ens = np.zeros((5, 100, 2))
        assert msd_ensemble(ens).shape == (100,)

    def test_zero_at_lag_zero(self):
        rng = np.random.default_rng(0)
        ens = rng.normal(size=(10, 50, 2)).cumsum(axis=1)
        msd = msd_ensemble(ens)
        assert msd[0] == pytest.approx(0.0, abs=1e-12)

    def test_uses_origin_at_t0_per_trajectory(self):
        """Each trajectory's origin is its own ``[0]`` row, not zero."""
        # Build two trajectories with non-zero starting points; the
        # EA-MSD must be invariant to translations of each trajectory.
        rng = np.random.default_rng(1)
        ens = rng.normal(size=(4, 30, 2)).cumsum(axis=1)
        msd_origin = msd_ensemble(ens)
        # Translate each trajectory by a different constant offset.
        offsets = rng.normal(size=(4, 1, 2)) * 7.0
        msd_shifted = msd_ensemble(ens + offsets)
        np.testing.assert_allclose(msd_origin, msd_shifted, atol=1e-10)

    def test_straight_line_is_ballistic(self):
        """``r_m(t) = v_m·t`` from r=0 → EA-MSD = ⟨v²⟩·Δ² exactly."""
        n_steps = 80
        v = np.array([1.0, 3.0, 5.0])           # M trajectories, 1-D
        t = np.arange(n_steps)
        ens = np.stack(
            [np.column_stack([vi * t, np.zeros(n_steps)]) for vi in v]
        )                                       # (M, N, 2)
        msd = msd_ensemble(ens)
        expected = (v**2).mean() * t**2
        np.testing.assert_allclose(msd, expected, atol=1e-12)

    def test_brownian_motion_is_diffusive(self):
        """Independent Gaussian increments → EA-MSD(Δ) = d · Δ."""
        rng = np.random.default_rng(2)
        n_traj, n_steps, d = 5_000, 200, 2
        incr = rng.normal(size=(n_traj, n_steps, d))   # unit variance
        traj = np.concatenate(
            [np.zeros((n_traj, 1, d)), np.cumsum(incr, axis=1)], axis=1
        )
        msd = msd_ensemble(traj)
        for k in [10, 50, 150]:
            assert msd[k] == pytest.approx(d * k, rel=0.05)

    def test_rejects_non_3d(self):
        with pytest.raises(ValueError):
            msd_ensemble(np.zeros((5, 100)))


class TestHurstFit:
    def test_recovers_diffusive_exponent(self):
        """Synthetic MSD ~ Δ^1 must give H = 0.5."""
        lags = np.linspace(1.0, 100.0, 200)
        msd = 3.0 * lags  # exactly diffusive
        fit = fit_hurst(lags, msd, lag_range=(5.0, 80.0))
        assert fit.hurst == pytest.approx(0.5, abs=1e-10)
        assert fit.slope == pytest.approx(1.0, abs=1e-10)

    def test_recovers_ballistic_exponent(self):
        lags = np.linspace(1.0, 100.0, 200)
        msd = 2.0 * lags**2
        fit = fit_hurst(lags, msd, lag_range=(5.0, 80.0))
        assert fit.hurst == pytest.approx(1.0, abs=1e-10)

    def test_recovers_anomalous_exponent(self):
        lags = np.linspace(1.0, 1000.0, 500)
        msd = lags**1.76  # paper's reported scaling
        fit = fit_hurst(lags, msd, lag_range=(10.0, 500.0))
        assert fit.hurst == pytest.approx(0.88, abs=1e-10)

    def test_rejects_zero_lags(self):
        with pytest.raises(ValueError):
            fit_hurst(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), (0.5, 1.5))

    def test_rejects_insufficient_points(self):
        lags = np.linspace(1.0, 100.0, 50)
        msd = lags
        with pytest.raises(ValueError):
            fit_hurst(lags, msd, lag_range=(10_000.0, 20_000.0))
