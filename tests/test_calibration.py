"""Tests for ``src/calibration.py``.

Covers atomic merging of calibration sections, missing-file handling,
numpy/Path normalisation, and the ``apply_calibration`` /
``load_calibrated_config`` overrides on :class:`SoaringConfig`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import soaring_ctrw.calibration as cal_mod
from soaring_ctrw.calibration import (
    apply_calibration,
    calibrated_sigma_theta,
    calibration_path,
    load_calibrated_config,
    read_calibration,
    write_calibration_section,
    _normalize,
)
from soaring_ctrw.model import (
    AngularConfig,
    ClimbMotionConfig,
    PhaseConfig,
    SearchMotionConfig,
    SoaringConfig,
)


@pytest.fixture
def calib_tmp(tmp_path, monkeypatch):
    """Redirect ``CALIBRATION_DIR`` to a tmp folder for the test."""
    target = tmp_path / "calibration"
    monkeypatch.setattr(cal_mod, "CALIBRATION_DIR", target)
    return target


def _make_config(name: str = "test_aircraft") -> SoaringConfig:
    return SoaringConfig(
        name=name,
        v_xy=10.0,
        transition=PhaseConfig("lomax", {"mu": 4.0, "tau_0": 200.0}),
        search=PhaseConfig("lomax", {"mu": 3.5, "tau_0": 100.0}),
        climb=PhaseConfig("exponential", {"tau_mean": 100.0}),
        angular=AngularConfig(sigma_theta=0.1, theta0=0.0),
        search_motion=SearchMotionConfig(
            u_S=10.0, tau_b_S=8.0, tau_turn_S=5.0,
            alpha_S=0.6, Omega_S=0.3,
        ),
        climb_motion=ClimbMotionConfig(
            r0=40.0, T_turn_mean=30.0, T_turn_std=5.0, v_drift=1.0,
        ),
    )


# ---------------------------------------------------------------------------
# I/O primitives
# ---------------------------------------------------------------------------

class TestReadCalibration:
    def test_missing_file_returns_empty_dict(self, calib_tmp):
        assert read_calibration("nonexistent") == {}

    def test_calibration_path_lives_under_calibration_dir(self, calib_tmp):
        p = calibration_path("paragliders")
        assert p == calib_tmp / "paragliders.yaml"


class TestWriteCalibrationSection:
    def test_creates_file_with_section(self, calib_tmp):
        out = write_calibration_section(
            "paragliders", "sigma_theta",
            {"value": 0.346, "mode": "full"},
        )
        assert out.exists()
        assert out == calibration_path("paragliders")
        data = yaml.safe_load(out.read_text())
        assert data["aircraft"] == "paragliders"
        assert data["sigma_theta"] == {"value": 0.346, "mode": "full"}

    def test_merges_sections_without_overwriting_others(self, calib_tmp):
        write_calibration_section(
            "hang_gliders", "sigma_theta",
            {"value": 0.30, "mode": "bare"},
        )
        write_calibration_section(
            "hang_gliders", "search",
            {"u_S_fitted": 22.5, "tau_turn_calibrated": 4.1},
        )
        data = read_calibration("hang_gliders")
        assert set(data) == {"aircraft", "sigma_theta", "search"}
        assert data["sigma_theta"]["value"] == 0.30
        assert data["search"]["u_S_fitted"] == 22.5

    def test_overwrites_only_the_named_section(self, calib_tmp):
        write_calibration_section(
            "sailplanes", "sigma_theta",
            {"value": 0.20, "mode": "bare"},
        )
        write_calibration_section(
            "sailplanes", "sigma_theta",
            {"value": 0.25, "mode": "full"},
        )
        data = read_calibration("sailplanes")
        assert data["sigma_theta"] == {"value": 0.25, "mode": "full"}

    def test_no_python_tags_in_yaml(self, calib_tmp):
        """``_normalize`` must coerce numpy/Path so YAML stays clean."""
        out = write_calibration_section(
            "paragliders", "search",
            {
                "u_S_fitted": np.float64(39.05),
                "n_iter": np.int64(7),
                "config_path": Path("/tmp/configs/paragliders.yaml"),
                "grid": np.array([1.0, 2.0, 3.0]),
                "nested": {"by_mode": {"full": np.float32(0.346)}},
            },
        )
        text = out.read_text()
        assert "!!python/" not in text
        data = yaml.safe_load(text)
        section = data["search"]
        assert isinstance(section["u_S_fitted"], float)
        assert isinstance(section["n_iter"], int)
        assert section["config_path"] == "/tmp/configs/paragliders.yaml"
        assert section["grid"] == [1.0, 2.0, 3.0]
        assert isinstance(section["nested"]["by_mode"]["full"], float)

    def test_no_tmp_file_left_behind(self, calib_tmp):
        out = write_calibration_section(
            "paragliders", "sigma_theta", {"value": 0.3},
        )
        tmp_path = out.with_suffix(out.suffix + ".tmp")
        assert not tmp_path.exists()


# ---------------------------------------------------------------------------
# _normalize helper
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_passes_through_plain_python(self):
        obj = {"a": 1, "b": [2, 3], "c": "x"}
        assert _normalize(obj) == obj

    def test_coerces_numpy_scalars(self):
        out = _normalize({"x": np.int32(7), "y": np.float64(1.5)})
        assert isinstance(out["x"], int)
        assert isinstance(out["y"], float)

    def test_coerces_ndarray_to_list(self):
        out = _normalize(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert out == [[1.0, 2.0], [3.0, 4.0]]

    def test_coerces_path_to_str(self):
        assert _normalize(Path("/a/b")) == "/a/b"

    def test_recurses_into_tuples_as_lists(self):
        out = _normalize((1, (2, 3)))
        assert out == [1, [2, 3]]


# ---------------------------------------------------------------------------
# Readers + overrides
# ---------------------------------------------------------------------------

class TestCalibratedSigmaTheta:
    def test_returns_value_when_present(self, calib_tmp):
        write_calibration_section(
            "paragliders", "sigma_theta", {"value": 0.346},
        )
        assert calibrated_sigma_theta("paragliders") == pytest.approx(0.346)

    def test_raises_when_file_missing(self, calib_tmp):
        with pytest.raises(FileNotFoundError, match="No sigma_theta"):
            calibrated_sigma_theta("paragliders")

    def test_raises_when_section_missing(self, calib_tmp):
        write_calibration_section(
            "paragliders", "search", {"u_S_fitted": 39.0},
        )
        with pytest.raises(FileNotFoundError, match="No sigma_theta"):
            calibrated_sigma_theta("paragliders")

    def test_raises_when_value_key_missing(self, calib_tmp):
        write_calibration_section(
            "paragliders", "sigma_theta", {"mode": "full"},
        )
        with pytest.raises(FileNotFoundError, match="No sigma_theta"):
            calibrated_sigma_theta("paragliders")


class TestApplyCalibration:
    def test_replaces_only_sigma_theta(self, calib_tmp):
        cfg = _make_config(name="paragliders")
        write_calibration_section(
            "paragliders", "sigma_theta", {"value": 0.5},
        )
        out = apply_calibration(cfg)
        # sigma_theta updated
        assert out.angular.sigma_theta == pytest.approx(0.5)
        # everything else preserved
        assert out.angular.theta0 == cfg.angular.theta0
        assert out.v_xy == cfg.v_xy
        assert out.name == cfg.name
        assert out.search_motion == cfg.search_motion
        assert out.climb_motion == cfg.climb_motion
        # original unchanged (frozen dataclass)
        assert cfg.angular.sigma_theta == 0.1

    def test_propagates_missing_calibration(self, calib_tmp):
        cfg = _make_config(name="paragliders")
        with pytest.raises(FileNotFoundError):
            apply_calibration(cfg)


class TestLoadCalibratedConfig:
    def test_reads_config_and_applies_calibration(self, tmp_path, calib_tmp):
        # Minimal aircraft YAML in a tmp configs/ dir.
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        (cfg_dir / "myac.yaml").write_text(
            yaml.safe_dump({
                "name": "myac",
                "v_xy": 12.0,
                "transition": {"distribution": "lomax",
                               "params": {"mu": 4.0, "tau_0": 200.0}},
                "search":     {"distribution": "lomax",
                               "params": {"mu": 3.5, "tau_0": 100.0}},
                "climb":      {"distribution": "exponential",
                               "params": {"tau_mean": 100.0}},
                "angular":    {"sigma_theta": 99.0, "theta0": 0.0},
            })
        )
        write_calibration_section("myac", "sigma_theta", {"value": 0.42})
        cfg = load_calibrated_config("myac", configs_dir=cfg_dir)
        assert cfg.angular.sigma_theta == pytest.approx(0.42)
        # The placeholder value in the YAML must have been overridden.
        assert cfg.angular.sigma_theta != 99.0
