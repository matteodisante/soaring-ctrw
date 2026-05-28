"""Cycle-based CTRW model with angular persistence for soaring flights.

This package implements the simulation and analysis code for the
manuscript *"A cycle-based model for the universal Hurst exponent in
thermal soaring flights"* (Di Sante, 2026).

Submodules
----------
:mod:`~soaring_ctrw.model`
    Frozen-dataclass parameter containers
    (:class:`~soaring_ctrw.model.SoaringConfig` and its sub-configs)
    and YAML loading.
:mod:`~soaring_ctrw.distributions`
    Waiting-time samplers (Pareto, Lomax, Exponential, Mittag-Leffler).
:mod:`~soaring_ctrw.simulation`
    The stochastic dynamics: per-cycle trajectory generation and
    ensemble simulation.
:mod:`~soaring_ctrw.observables`
    Time/ensemble-averaged MSD and Hurst-exponent fitting.
:mod:`~soaring_ctrw.calibration`
    Read/write the per-aircraft calibration YAML and inject the
    calibrated ``sigma_theta`` into a config.
:mod:`~soaring_ctrw.cache`
    Manifest-based NPZ caching shared by the Monte-Carlo scripts.
:mod:`~soaring_ctrw.paths`
    Repo-relative output locations.

The package namespace is intentionally kept empty: import the
submodules explicitly, e.g. ``from soaring_ctrw.model import
SoaringConfig``.
"""
