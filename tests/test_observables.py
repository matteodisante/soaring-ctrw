"""Tests for MSD computation and Hurst-exponent fitting."""

from __future__ import annotations

import numpy as np
import pytest

from soaring_ctrw.observables import (
    fit_hurst,
    msd_ensemble,
    msd_time_averaged,
)


class TestMSDTimeAveraged:
    def test_zero_at_lag_zero(self):
        rng = np.random.default_rng(0)
        traj = rng.normal(size=(100, 2)).cumsum(axis=0)
        msd = msd_time_averaged(traj)
        assert msd[0] == pytest.approx(0.0, abs=1e-10)

    def test_shape(self):
        traj = np.zeros((50, 2))
        assert msd_time_averaged(traj).shape == (50,)

    def test_straight_line_is_ballistic(self):
        """x(t) = v·t → MSD(Δ) = v²·Δ² exactly."""
        n = 200
        v = 3.0
        dt = 1.0
        t = np.arange(n) * dt
        traj = np.column_stack([v * t, np.zeros_like(t)])
        msd = msd_time_averaged(traj)
        lags = np.arange(n)
        # test at a handful of mid-range lags where averaging is stable
        for k in [5, 20, 50, 100]:
            assert msd[k] == pytest.approx(v**2 * lags[k] ** 2, rel=1e-10)

    def test_brownian_motion_is_diffusive(self):
        """Independent Gaussian increments → MSD(Δ) = d · Δ (with unit variance)."""
        rng = np.random.default_rng(1)
        n, d = 10_000, 2
        increments = rng.normal(size=(n, d))  # variance 1 per component
        traj = np.concatenate([np.zeros((1, d)), np.cumsum(increments, axis=0)])
        msd = msd_time_averaged(traj)
        # MSD(k) ≈ d · k for k << n
        for k in [10, 50, 200]:
            assert msd[k] == pytest.approx(d * k, rel=0.10)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            msd_time_averaged(np.zeros(10))


class TestMSDEnsemble:
    def test_shape(self):
        ens = np.zeros((5, 100, 2))
        assert msd_ensemble(ens).shape == (100,)

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
