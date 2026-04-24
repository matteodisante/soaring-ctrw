"""Tests for the step-based CTRW model and Monte Carlo simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from soaring_ctrw.model import AngularConfig, PhaseConfig, SoaringConfig
from soaring_ctrw.simulation import simulate_ensemble, simulate_single


@pytest.fixture
def minimal_config() -> SoaringConfig:
    return SoaringConfig(
        name="test",
        v_xy=10.0,
        transition=PhaseConfig("pareto", {"mu": 4.0, "tau_min": 10.0}),
        search=PhaseConfig("pareto", {"mu": 3.5, "tau_min": 5.0}),
        climb=PhaseConfig("exponential", {"tau_mean": 30.0}),
        angular=AngularConfig(sigma_theta=0.5),
    )


class TestAngularConfig:
    def test_persistence_length(self):
        c = AngularConfig(sigma_theta=1.0)
        assert c.persistence_cycles == pytest.approx(2.0)

    def test_infinite_persistence_at_zero(self):
        assert AngularConfig(sigma_theta=0.0).persistence_cycles == float("inf")

    def test_negative_sigma_rejected(self):
        with pytest.raises(ValueError):
            AngularConfig(sigma_theta=-0.1)


class TestSingleTrajectory:
    def test_shape(self, minimal_config):
        rng = np.random.default_rng(0)
        traj = simulate_single(minimal_config, n_cycles=100, rng=rng)

        assert traj.positions.shape == (101, 2)
        assert traj.cycle_end_times.shape == (101,)
        assert traj.headings.shape == (100,)
        assert traj.phase_durations.shape == (100, 3)

    def test_origin(self, minimal_config):
        rng = np.random.default_rng(0)
        traj = simulate_single(minimal_config, n_cycles=10, rng=rng)
        assert np.allclose(traj.positions[0], [0.0, 0.0])
        assert traj.cycle_end_times[0] == 0.0

    def test_times_monotonic(self, minimal_config):
        rng = np.random.default_rng(0)
        traj = simulate_single(minimal_config, n_cycles=100, rng=rng)
        assert np.all(np.diff(traj.cycle_end_times) > 0)

    def test_zero_sigma_theta_yields_straight_line(self, minimal_config):
        """With σ_θ = 0 and θ₀ fixed, all displacements are collinear."""
        config = SoaringConfig(
            name="straight",
            v_xy=minimal_config.v_xy,
            transition=minimal_config.transition,
            search=minimal_config.search,
            climb=minimal_config.climb,
            angular=AngularConfig(sigma_theta=0.0, theta0=0.3),
        )
        rng = np.random.default_rng(0)
        traj = simulate_single(config, n_cycles=50, rng=rng)
        # all heading values equal theta0
        assert np.allclose(traj.headings, 0.3)
        # final y / x must equal tan(theta0) (up to numerical noise)
        x_end, y_end = traj.positions[-1]
        assert y_end / x_end == pytest.approx(np.tan(0.3), rel=1e-10)

    def test_reject_non_positive_cycles(self, minimal_config):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            simulate_single(minimal_config, n_cycles=0, rng=rng)


class TestEnsemble:
    def test_shape(self, minimal_config):
        rng = np.random.default_rng(0)
        ens = simulate_ensemble(
            minimal_config,
            n_trajectories=5,
            total_time=1000.0,
            dt=1.0,
            rng=rng,
        )
        # total_time=1000, dt=1 → 1001 time steps (inclusive of both ends)
        assert ens.shape == (5, 1001, 2)

    def test_first_step_is_origin(self, minimal_config):
        rng = np.random.default_rng(0)
        ens = simulate_ensemble(
            minimal_config,
            n_trajectories=3,
            total_time=500.0,
            dt=1.0,
            rng=rng,
        )
        assert np.allclose(ens[:, 0, :], 0.0)


class TestYamlLoading:
    def test_roundtrip(self, tmp_path: Path):
        data = {
            "name": "test_aircraft",
            "v_xy": 12.0,
            "transition": {"distribution": "pareto", "params": {"mu": 3.0, "tau_min": 20.0}},
            "search": {"distribution": "pareto", "params": {"mu": 2.5, "tau_min": 10.0}},
            "climb": {"distribution": "exponential", "params": {"tau_mean": 100.0}},
            "angular": {"sigma_theta": 0.4, "theta0": 0.0},
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data))

        config = SoaringConfig.from_yaml(path)
        assert config.name == "test_aircraft"
        assert config.v_xy == 12.0
        assert config.transition.params["mu"] == 3.0
        assert config.angular.sigma_theta == 0.4
