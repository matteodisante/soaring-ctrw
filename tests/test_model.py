"""Tests for the configuration objects and the cycle-based MC simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from model import (
    AngularConfig,
    ClimbMotionConfig,
    PhaseConfig,
    SearchMotionConfig,
    SoaringConfig,
)
from simulation import simulate_ensemble, simulate_single


def _full_config(theta0: float | None = None) -> SoaringConfig:
    return SoaringConfig(
        name="test_full",
        v_xy=10.0,
        transition=PhaseConfig("lomax", {"mu": 4.0, "tau_0": 200.0}),
        search=PhaseConfig("lomax", {"mu": 3.5, "tau_0": 100.0}),
        climb=PhaseConfig("exponential", {"tau_mean": 100.0}),
        angular=AngularConfig(sigma_theta=0.5, theta0=theta0),
        search_motion=SearchMotionConfig(
            u_S=10.0, tau_b_S=8.0, tau_turn_S=5.0,
            alpha_S=0.6, Omega_S=0.3,
        ),
        climb_motion=ClimbMotionConfig(
            r0=40.0, T_turn_mean=30.0, T_turn_std=5.0, v_drift=1.0,
        ),
    )


def _bare_config(theta0: float | None = None) -> SoaringConfig:
    return SoaringConfig(
        name="test_bare",
        v_xy=10.0,
        transition=PhaseConfig("lomax", {"mu": 4.0, "tau_0": 200.0}),
        search=PhaseConfig("lomax", {"mu": 3.5, "tau_0": 100.0}),
        climb=PhaseConfig("exponential", {"tau_mean": 100.0}),
        angular=AngularConfig(sigma_theta=0.5, theta0=theta0),
        search_motion=None,
        climb_motion=None,
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


class TestSingleTrajectoryFull:
    def test_shape(self):
        rng = np.random.default_rng(0)
        traj = simulate_single(_full_config(theta0=0.0), n_cycles=50, rng=rng)
        assert traj.positions.shape == (51, 2)
        assert traj.cycle_end_times.shape == (51,)
        assert traj.headings.shape == (50,)
        assert traj.phase_durations_active.shape == (50, 3)
        assert len(traj.search_episodes) == 50

    def test_origin(self):
        rng = np.random.default_rng(0)
        traj = simulate_single(_full_config(theta0=0.0), n_cycles=10, rng=rng)
        assert np.allclose(traj.positions[0], [0.0, 0.0])
        assert traj.cycle_end_times[0] == 0.0

    def test_cycle_clock_monotonic(self):
        rng = np.random.default_rng(0)
        traj = simulate_single(_full_config(theta0=0.0), n_cycles=20, rng=rng)
        assert np.all(np.diff(traj.cycle_end_times) > 0)

    def test_search_T_phys_equals_tau_S(self):
        """The physical search duration equals the Lomax draw exactly
        (legs + waits stopping rule)."""
        rng = np.random.default_rng(7)
        traj = simulate_single(_full_config(theta0=0.0), n_cycles=20, rng=rng)
        for n, ep in enumerate(traj.search_episodes):
            tau_S_n = traj.phase_durations_active[n, 1]
            total = float(np.sum(ep.leg_durations) + np.sum(ep.turn_durations))
            assert np.isclose(total, tau_S_n, rtol=1e-12)
            assert np.isclose(ep.T_phys, tau_S_n, rtol=1e-12)

    def test_search_ballistic_at_most_tau_S(self):
        """Total ballistic time in a search episode cannot exceed tau_S^n."""
        rng = np.random.default_rng(0)
        traj = simulate_single(_full_config(theta0=0.0), n_cycles=50, rng=rng)
        tau_S = traj.phase_durations_active[:, 1]
        for n, ep in enumerate(traj.search_episodes):
            assert float(np.sum(ep.leg_durations)) <= tau_S[n] + 1e-12


class TestSingleTrajectoryBare:
    def test_search_T_phys_equals_tau_S(self):
        """Bare model: search physical duration coincides with the Lomax draw."""
        rng = np.random.default_rng(0)
        traj = simulate_single(_bare_config(theta0=0.0), n_cycles=10, rng=rng)
        tau_S = traj.phase_durations_active[:, 1]
        T_phys = traj.search_T_phys
        assert np.allclose(T_phys, tau_S)

    def test_zero_sigma_theta_yields_straight_line(self):
        """With sigma_theta = 0 and theta0 fixed, all transitions are collinear."""
        cfg = SoaringConfig(
            name="straight",
            v_xy=10.0,
            transition=PhaseConfig("lomax", {"mu": 4.0, "tau_0": 200.0}),
            search=PhaseConfig("lomax", {"mu": 3.5, "tau_0": 100.0}),
            climb=PhaseConfig("exponential", {"tau_mean": 100.0}),
            angular=AngularConfig(sigma_theta=0.0, theta0=0.3),
        )
        rng = np.random.default_rng(0)
        traj = simulate_single(cfg, n_cycles=30, rng=rng)
        # all headings equal theta0
        assert np.allclose(traj.headings, 0.3)
        x_end, y_end = traj.positions[-1]
        assert y_end / x_end == pytest.approx(np.tan(0.3), rel=1e-10)

    def test_reject_non_positive_cycles(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            simulate_single(_bare_config(theta0=0.0), n_cycles=0, rng=rng)


class TestEnsembleFull:
    def test_shape(self):
        rng = np.random.default_rng(0)
        ens = simulate_ensemble(
            _full_config(theta0=0.0),
            n_trajectories=5, total_time=2000.0, dt=1.0, rng=rng,
        )
        assert ens.shape == (5, 2001, 2)

    def test_first_step_is_origin(self):
        rng = np.random.default_rng(0)
        ens = simulate_ensemble(
            _full_config(theta0=0.0),
            n_trajectories=3, total_time=500.0, dt=1.0, rng=rng,
        )
        assert np.allclose(ens[:, 0, :], 0.0)


class TestEnsembleBare:
    def test_shape(self):
        rng = np.random.default_rng(0)
        ens = simulate_ensemble(
            _bare_config(theta0=0.0),
            n_trajectories=4, total_time=1000.0, dt=1.0, rng=rng,
        )
        assert ens.shape == (4, 1001, 2)


class TestYamlLoading:
    def test_roundtrip(self, tmp_path: Path):
        data = {
            "name": "test_aircraft",
            "v_xy": 12.0,
            "transition": {"distribution": "lomax", "params": {"mu": 3.0, "tau_0": 20.0}},
            "search":     {"distribution": "lomax", "params": {"mu": 2.5, "tau_0": 10.0}},
            "climb":      {"distribution": "exponential", "params": {"tau_mean": 100.0}},
            "angular":    {"sigma_theta": 0.4, "theta0": 0.0},
            "search_motion": {
                "u_S": 8.0, "tau_b_S": 10.0, "tau_turn_S": 5.0,
                "alpha_S": 0.6, "Omega_S": 0.3,
            },
            "climb_motion": {
                "r0": 40.0, "T_turn_mean": 30.0, "T_turn_std": 5.0,
                "v_drift": 1.0,
            },
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data))

        cfg = SoaringConfig.from_yaml(path)
        assert cfg.name == "test_aircraft"
        assert cfg.v_xy == 12.0
        assert cfg.transition.params["mu"] == 3.0
        assert cfg.search_motion.u_S == 8.0
        assert cfg.climb_motion.r0 == 40.0

    def test_yaml_omitting_intra_phase_motion(self, tmp_path: Path):
        data = {
            "name": "bare",
            "v_xy": 10.0,
            "transition": {"distribution": "lomax", "params": {"mu": 3.0, "tau_0": 20.0}},
            "search":     {"distribution": "lomax", "params": {"mu": 2.5, "tau_0": 10.0}},
            "climb":      {"distribution": "exponential", "params": {"tau_mean": 100.0}},
            "angular":    {"sigma_theta": 0.4, "theta0": None},
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data))

        cfg = SoaringConfig.from_yaml(path)
        assert cfg.search_motion is None
        assert cfg.climb_motion is None
