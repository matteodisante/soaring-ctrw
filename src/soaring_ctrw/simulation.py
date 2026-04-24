"""Monte Carlo simulation of the step-based CTRW model.

Two simulation modes are provided:

- :func:`simulate_single`: generates one trajectory expressed *per cycle*
  (i.e. one sample per complete transition→search→climb cycle). This is
  the native representation of the model and is used for analytical
  cross-checks where the natural time unit is the cycle index.

- :func:`simulate_ensemble`: produces a regularly sampled ensemble of
  trajectories on a common time grid with step ``dt``. This is the
  quantity needed to compute the time-lag MSD in physical seconds,
  directly comparable to Fig. 1 of the paper.

Intra-phase motion has three pieces:

- **Transition**: constant velocity ``v_xy·ê(θ_n)`` over the whole
  duration ``τ^T_n`` (heading-correlated ballistic kernel).
- **Search**: subordinated Lévy walk implemented in
  :func:`_sample_ctrw_legs` — alternating ballistic legs of speed
  ``v_c^S`` (exponential durations of scale ``σ_0``, Gaussian-random-walk
  heading on the circle of variance ``σ_ψ²``) and stationary
  Mittag-Leffler waits of stability ``α_S∈(0,1)`` and scale
  ``τ_w^S``. Reproduces the ballistic → sub-diffusive ``Δ^{α_S}``
  crossover of Fig. 3 of Vilpellet et al. (2026). Controlled by
  :class:`SearchMotionConfig`.
- **Climb**: 2-D harmonic oscillator (pilot circling a thermal core
  of radius ``r_0`` with mean turn period ``T_turn``, Gaussian
  per-cycle dispersion ``σ_T``) plus a linear orographic drift of
  magnitude ``v_drift`` and uniform per-cycle direction. Controlled
  by :class:`ClimbMotionConfig`.

If ``search_motion`` and ``climb_motion`` are omitted from the config,
the corresponding phase is held horizontally stationary (minimal
baseline used as a sanity check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soaring_ctrw.model import SoaringConfig

__all__ = [
    "CycleTrajectory",
    "simulate_single",
    "simulate_ensemble",
]


@dataclass(frozen=True)
class CycleTrajectory:
    """One realisation of the step-based CTRW, indexed by cycle number.

    Attributes
    ----------
    positions : ndarray, shape (N+1, 2)
        Horizontal position at the end of each complete cycle.
    cycle_end_times : ndarray, shape (N+1,)
        Physical time at the end of each cycle, ``cycle_end_times[0]=0``.
    headings : ndarray, shape (N,)
        Transition-phase heading angle used for cycle ``n`` (rad).
    phase_durations : ndarray, shape (N, 3)
        Columns are (τ^T, τ^S, τ^C) for each cycle.
    search_legs : list[tuple[ndarray, ndarray]]
        For each cycle ``n``, a tuple ``(durations, angles)`` of 1D
        arrays giving the duration and direction of each leg of the
        intra-search Lévy walk. Empty tuple if ``search_motion is None``.
    climb_phase0 : ndarray, shape (N,)
        Initial phase :math:`\\phi_0` of the thermalling harmonic
        oscillator for each cycle (rad). Uniform on :math:`[0, 2\\pi)`.
    climb_drift_angles : ndarray, shape (N,)
        Direction :math:`\\phi^{\\mathrm{drift}}_n` of the orographic
        drift during climb (rad). Uniform on :math:`[0, 2\\pi)`.
    """

    positions: np.ndarray
    cycle_end_times: np.ndarray
    headings: np.ndarray
    phase_durations: np.ndarray
    search_legs: list
    climb_phase0: np.ndarray
    climb_drift_angles: np.ndarray
    climb_turn_periods: np.ndarray  # per-cycle T_turn, shape (N,)

    @property
    def n_cycles(self) -> int:
        return len(self.headings)

    @property
    def total_time(self) -> float:
        return float(self.cycle_end_times[-1])


def simulate_single(
    config: SoaringConfig,
    n_cycles: int,
    rng: np.random.Generator,
) -> CycleTrajectory:
    """Simulate one trajectory of exactly ``n_cycles`` cycles.

    Each cycle has three phases:

    1. **Transition**: deterministic ballistic step
       :math:`v_{xy}\\,\\tau^T_n\\,\\hat e(\\theta_n)` where the
       heading :math:`\\theta_n` follows a Gaussian random walk (A3).

    2. **Search**: coupled Lévy walk of rectilinear legs inside the
       total phase duration :math:`\\tau^S_n` drawn from (A1). Leg
       durations are i.i.d. Mittag-Leffler with stability
       :math:`\\alpha_S` and scale :math:`\\tau_{\\ell,\\min}^S`,
       truncated so that the sum equals :math:`\\tau^S_n`. During each
       leg the walker moves at constant speed :math:`v_c^S` in a
       direction :math:`\\psi_k` that follows a Gaussian random walk
       on the circle with increment variance
       :math:`\\sigma_{\\psi,S}^2`.

    3. **Climb**: 2D harmonic oscillator (pilot circling a thermal
       at radius :math:`r_0` with period :math:`T_{\\mathrm{turn}}`)
       plus a linear drift of magnitude :math:`v_{\\mathrm{drift}}`
       in a direction :math:`\\phi_n^{\\mathrm{drift}}` drawn i.i.d.
       uniform on :math:`[0, 2\\pi)` per cycle.
    """
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be positive, got {n_cycles}")

    transition_sampler = config.transition.build()
    search_sampler = config.search.build()
    climb_sampler = config.climb.build()

    tau_T = transition_sampler.sample(n_cycles, rng)
    tau_S = search_sampler.sample(n_cycles, rng)
    tau_C = climb_sampler.sample(n_cycles, rng)
    phase_durations = np.column_stack([tau_T, tau_S, tau_C])

    # Transition heading: Gaussian random walk on the circle
    eta = rng.normal(loc=0.0, scale=config.angular.sigma_theta, size=n_cycles)
    theta = config.angular.theta0 + np.cumsum(eta)

    # Transition displacement
    dx_T = config.v_xy * tau_T * np.cos(theta)
    dy_T = config.v_xy * tau_T * np.sin(theta)

    # Search: subordinated Lévy walk (ballistic legs + ML waits)
    search_legs = []
    dx_S = np.zeros(n_cycles)
    dy_S = np.zeros(n_cycles)
    if config.search_motion is not None:
        v_c_S = config.search_motion.v_c_S
        sigma_0 = config.search_motion.sigma_0
        tau_0_S = config.search_motion.tau_0_S
        alpha_S = config.search_motion.alpha_S
        sigma_psi_S = config.search_motion.sigma_psi_S
        for n in range(n_cycles):
            leg_d, wait_d, angs, leg_t0 = _sample_ctrw_legs(
                total_time=tau_S[n],
                alpha=alpha_S,
                sigma_0=sigma_0,
                tau_0=tau_0_S,
                v_c=v_c_S,
                sigma_angular=sigma_psi_S,
                rng=rng,
            )
            search_legs.append((leg_d, wait_d, angs, leg_t0))
            # End-of-phase displacement: sum of ballistic leg contributions
            dx_S[n] = v_c_S * np.sum(leg_d * np.cos(angs))
            dy_S[n] = v_c_S * np.sum(leg_d * np.sin(angs))
    else:
        search_legs = [
            (np.array([]), np.array([]), np.array([]), np.array([]))
            for _ in range(n_cycles)
        ]

    # Climb: 2D harmonic oscillator + drift
    dx_C = np.zeros(n_cycles)
    dy_C = np.zeros(n_cycles)
    climb_turn_periods = np.full(n_cycles, 30.0)
    if config.climb_motion is not None:
        r0 = config.climb_motion.thermal_radius
        T_turn_mean = config.climb_motion.turn_period
        T_turn_std = config.climb_motion.turn_period_std
        v_drift = config.climb_motion.v_drift

        # Per-cycle turn period: Gaussian, positivity-clipped
        if T_turn_std > 0:
            T_turn_n = rng.normal(loc=T_turn_mean, scale=T_turn_std, size=n_cycles)
            T_turn_n = np.clip(T_turn_n, 0.2 * T_turn_mean, None)
        else:
            T_turn_n = np.full(n_cycles, T_turn_mean)
        climb_turn_periods = T_turn_n
        omega_n = 2 * np.pi / T_turn_n

        phi0 = rng.uniform(0.0, 2 * np.pi, size=n_cycles)
        phi_drift = rng.uniform(0.0, 2 * np.pi, size=n_cycles)

        osc_end_x = r0 * (np.cos(omega_n * tau_C + phi0) - np.cos(phi0))
        osc_end_y = r0 * (np.sin(omega_n * tau_C + phi0) - np.sin(phi0))
        drift_end_x = v_drift * tau_C * np.cos(phi_drift)
        drift_end_y = v_drift * tau_C * np.sin(phi_drift)

        dx_C = osc_end_x + drift_end_x
        dy_C = osc_end_y + drift_end_y
    else:
        phi0 = np.zeros(n_cycles)
        phi_drift = np.zeros(n_cycles)

    dx_cycle = dx_T + dx_S + dx_C
    dy_cycle = dy_T + dy_S + dy_C
    x = np.concatenate(([0.0], np.cumsum(dx_cycle)))
    y = np.concatenate(([0.0], np.cumsum(dy_cycle)))
    positions = np.column_stack([x, y])

    cycle_durations = tau_T + tau_S + tau_C
    cycle_end_times = np.concatenate(([0.0], np.cumsum(cycle_durations)))

    return CycleTrajectory(
        positions=positions,
        cycle_end_times=cycle_end_times,
        headings=theta,
        phase_durations=phase_durations,
        search_legs=search_legs,
        climb_phase0=phi0,
        climb_drift_angles=phi_drift,
        climb_turn_periods=climb_turn_periods,
    )


def _sample_ctrw_legs(
    total_time: float,
    alpha: float,
    sigma_0: float,
    tau_0: float,
    v_c: float,
    sigma_angular: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Sample a subordinated Lévy walk inside a search phase.

    The process alternates ballistic legs (walker moves at speed
    :math:`v_c`) with stationary waiting times (walker at rest). Legs
    have exponential durations with scale :math:`\sigma_0`; waits have
    Mittag-Leffler durations with stability :math:`\alpha` and scale
    :math:`\tau_0`. Leg directions follow a Gaussian random walk on the
    circle with per-step variance :math:`\sigma_{\mathrm{angular}}^2`,
    initialised uniformly on :math:`[0, 2\pi)`.

    The total physical time is enforced to equal ``total_time`` by
    truncating the last interval (whether it is a leg or a wait).

    Returns
    -------
    leg_durations : ndarray, shape (L,)
        Duration of each ballistic leg. Positive.
    wait_durations : ndarray, shape (L,)
        Duration of each waiting time, one per leg, appearing *after*
        the corresponding leg. May be zero if truncated.
    leg_angles : ndarray, shape (L,)
        Direction of each ballistic leg (rad).
    leg_start_times : ndarray, shape (L,)
        Physical time at the start of each leg, relative to the phase
        onset.
    """
    if total_time <= 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    from soaring_ctrw.distributions import MittagLeffler

    batch = max(10, int(3 * total_time / (sigma_0 + tau_0)))
    ml = MittagLeffler(alpha=alpha, tau_0=tau_0)

    leg_durations = []
    wait_durations = []
    leg_angles = []
    leg_start_times = []

    psi = rng.uniform(0.0, 2 * np.pi)
    elapsed = 0.0
    while elapsed < total_time:
        sigmas = rng.exponential(scale=sigma_0, size=batch)
        waits = ml.sample(batch, rng)
        xis = rng.normal(loc=0.0, scale=sigma_angular, size=batch)

        for sigma_leg, wait_leg, xi_leg in zip(sigmas, waits, xis):
            leg_remaining = total_time - elapsed
            if leg_remaining <= 0:
                break
            # Truncate the leg if it overshoots
            actual_leg = min(sigma_leg, leg_remaining)
            leg_durations.append(actual_leg)
            leg_angles.append(psi)
            leg_start_times.append(elapsed)
            elapsed += actual_leg
            # Waiting time (may be truncated, may be zero)
            wait_remaining = total_time - elapsed
            if wait_remaining <= 0:
                wait_durations.append(0.0)
                break
            actual_wait = min(wait_leg, wait_remaining)
            wait_durations.append(actual_wait)
            elapsed += actual_wait
            # Update heading for next leg
            psi = psi + xi_leg

    return (
        np.array(leg_durations),
        np.array(wait_durations),
        np.array(leg_angles),
        np.array(leg_start_times),
    )


def _search_endpoint(
    tau: float,
    K_S: float,
    H_S: float,
    seed: int,
) -> tuple[float, float]:
    """[DEPRECATED] kept for backward compatibility of tests; see
    :func:`_fbm_endpoint`."""
    return _fbm_endpoint(tau=tau, sigma=K_S, H=H_S, seed=seed)


def _fbm_covariance(times: np.ndarray, H: float, sigma: float) -> np.ndarray:
    r"""Covariance matrix of 1-D fractional Brownian motion sampled at ``times``.

    For fBm :math:`B_H(t)` with Hurst index :math:`H \in (0,1)` and
    scaling :math:`\sigma`, the covariance is

    .. math::
        \mathrm{Cov}[B_H(s), B_H(t)] =
        \tfrac{\sigma^2}{2}\,(s^{2H} + t^{2H} - |t-s|^{2H}).

    This ensures :math:`\mathrm{Var}[B_H(t)] = \sigma^2 t^{2H}` and
    :math:`\mathrm{Var}[B_H(t)-B_H(s)] = \sigma^2 |t-s|^{2H}`.
    """
    s = times[:, None]
    t = times[None, :]
    return 0.5 * sigma**2 * (s ** (2 * H) + t ** (2 * H) - np.abs(t - s) ** (2 * H))


def _sample_fbm_path_at_times(
    tau: float,
    sigma: float,
    H: float,
    times: np.ndarray,
    seed: int,
) -> np.ndarray:
    r"""Sample a 2-D fractional Brownian motion path at the given times.

    The two Cartesian components are independent fBm with Hurst index
    :math:`H` and scaling :math:`\sigma` (m/s^H), satisfying
    :math:`X(0) = 0`. Sampling is via Cholesky decomposition of the
    covariance matrix on the requested times, :math:`O(N^3)` in
    ``len(times)``.

    Parameters
    ----------
    tau : float
        Total duration of the phase (s). Used only to clip ``times``
        into :math:`[0, \tau]` and to ensure the endpoint is present.
    sigma : float
        Scaling parameter of the fBm (m/s^H).
    H : float
        Hurst exponent, 0 < H < 1.
    times : ndarray, shape (N,)
        Query times in [0, tau].
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    ndarray, shape (N, 2)
    """
    if tau <= 0 or len(times) == 0 or sigma == 0:
        return np.zeros((len(times), 2))

    t = np.clip(times.astype(float), 0.0, tau)

    # Build time grid: prepend 0 and tau so that endpoint sampling is
    # consistent across invocations with different query times.
    t_unique, inverse_map = np.unique(
        np.concatenate([[0.0, tau], t]),
        return_inverse=True,
    )

    t_pos = t_unique[t_unique > 0]
    cov = _fbm_covariance(t_pos, H=H, sigma=sigma)
    cov = cov + 1e-12 * np.eye(len(t_pos))
    L = np.linalg.cholesky(cov)

    sub_rng = np.random.default_rng(seed)
    z = sub_rng.standard_normal(size=(len(t_pos), 2))
    X_pos = L @ z

    X_full = np.zeros((len(t_unique), 2))
    X_full[1:] = X_pos

    return X_full[inverse_map[2:]]


# Backward-compatibility alias for deprecated tests
_sample_search_path_at_times = _sample_fbm_path_at_times


def simulate_ensemble(
    config: SoaringConfig,
    n_trajectories: int,
    total_time: float,
    dt: float,
    rng: np.random.Generator,
    cycles_safety_factor: float = 3.0,
) -> np.ndarray:
    """Generate an ensemble of trajectories on a regular time grid.

    See module docstring for the treatment of intra-phase motion.

    Parameters
    ----------
    config : SoaringConfig
        Model parameters.
    n_trajectories : int
        Ensemble size.
    total_time : float
        Physical duration (seconds) of each interpolated trajectory.
    dt : float
        Sampling interval (seconds).
    rng : numpy.random.Generator
        Pre-seeded RNG.
    cycles_safety_factor : float, optional
        Multiplier on the expected number of cycles to cover
        ``total_time``.

    Returns
    -------
    ndarray, shape (n_trajectories, n_steps, 2)
        Sampled positions, where ``n_steps = int(total_time / dt) + 1``.
    """
    if n_trajectories <= 0:
        raise ValueError(f"n_trajectories must be positive, got {n_trajectories}")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    mean_cycle = _estimate_mean_cycle_duration(config)
    n_cycles_initial = max(1, int(cycles_safety_factor * total_time / mean_cycle))

    time_grid = np.arange(0.0, total_time + dt, dt)
    n_steps = len(time_grid)
    ensemble = np.zeros((n_trajectories, n_steps, 2), dtype=float)

    for i in range(n_trajectories):
        n_cycles = n_cycles_initial
        while True:
            traj = simulate_single(config, n_cycles=n_cycles, rng=rng)
            if traj.total_time >= total_time:
                break
            n_cycles *= 2

        ensemble[i] = _interpolate_physical(
            traj=traj,
            config=config,
            target_times=time_grid,
        )
    return ensemble


def _estimate_mean_cycle_duration(config: SoaringConfig) -> float:
    """Return a finite estimate of the mean cycle duration."""

    def _finite_mean(phase_cfg) -> float:
        sampler = phase_cfg.build()
        m = sampler.mean
        if np.isfinite(m):
            return float(m)
        return 10.0 * float(phase_cfg.params.get("tau_min", 1.0))

    return (
        _finite_mean(config.transition)
        + _finite_mean(config.search)
        + _finite_mean(config.climb)
    )


def _interpolate_physical(
    traj: CycleTrajectory,
    config: SoaringConfig,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpolate the cycle-level trajectory to a regular time grid.

    Within each cycle ``n`` the position evolves as:

    - **Transition** (0 ≤ t − t_start < τ^T): linear motion at
      ``v_xy·ê(θ_n)`` from the cycle-start position.

    - **Search** (τ^T ≤ t − t_start < τ^T + τ^S): velocity-limited
      CTRW built from ``traj.search_legs[n] = (durations, angles)``.
      Position evolves linearly within each leg at velocity
      ``v_c_S·ê(angle)``. The CTRW starts at the transition endpoint
      ``T_n``. If ``search_motion is None``, the walker is held at
      ``T_n``.

    - **Climb** (t − t_start ≥ τ^T + τ^S): velocity-limited CTRW with
      climb parameters, starting from ``S_n`` (end of search),
      plus a linear drift ``v_drift·ê(phi)·c``. If ``climb_motion is
      None``, held at ``S_n``.
    """
    n_cycles = traj.n_cycles
    positions = traj.positions
    cycle_end_times = traj.cycle_end_times
    tau_T = traj.phase_durations[:, 0]
    tau_S = traj.phase_durations[:, 1]
    tau_C = traj.phase_durations[:, 2]
    headings = traj.headings

    out = np.zeros((len(target_times), 2), dtype=float)

    cycle_idx = np.searchsorted(cycle_end_times, target_times, side="right") - 1
    cycle_idx = np.clip(cycle_idx, 0, n_cycles - 1)

    for n in range(n_cycles):
        mask = cycle_idx == n
        if not np.any(mask):
            continue

        t_in_cycle = target_times[mask] - cycle_end_times[n]
        tau_T_n = tau_T[n]
        tau_S_n = tau_S[n]
        tau_C_n = tau_C[n]
        theta_n = headings[n]
        P_n = positions[n]

        # Transition endpoint
        dx_T_n = config.v_xy * tau_T_n * np.cos(theta_n)
        dy_T_n = config.v_xy * tau_T_n * np.sin(theta_n)
        T_n = P_n + np.array([dx_T_n, dy_T_n])

        # Compute search-phase endpoint S_n by summing leg displacements
        if config.search_motion is not None and tau_S_n > 0:
            leg_d, _wait_d, angs_S, _leg_t0 = traj.search_legs[n]
            v_c_S = config.search_motion.v_c_S
            S_n = T_n + np.array([
                v_c_S * np.sum(leg_d * np.cos(angs_S)),
                v_c_S * np.sum(leg_d * np.sin(angs_S)),
            ])
        else:
            S_n = T_n

        # Classify target times by phase
        in_T = t_in_cycle < tau_T_n
        in_S = (~in_T) & (t_in_cycle < tau_T_n + tau_S_n)
        in_C = ~(in_T | in_S)

        out_slice = np.zeros((len(t_in_cycle), 2), dtype=float)

        # --- Transition ---
        if np.any(in_T):
            t_T = t_in_cycle[in_T]
            out_slice[in_T, 0] = P_n[0] + config.v_xy * t_T * np.cos(theta_n)
            out_slice[in_T, 1] = P_n[1] + config.v_xy * t_T * np.sin(theta_n)

        # --- Search ---
        if np.any(in_S):
            s_t = t_in_cycle[in_S] - tau_T_n
            if config.search_motion is not None:
                leg_d, wait_d, angs_S, leg_t0 = traj.search_legs[n]
                v_c_S = config.search_motion.v_c_S
                xy_S = _search_ctrw_positions_at_times(
                    leg_durations=leg_d,
                    wait_durations=wait_d,
                    leg_angles=angs_S,
                    leg_start_times=leg_t0,
                    v_c=v_c_S,
                    times=s_t,
                )
                out_slice[in_S, 0] = T_n[0] + xy_S[:, 0]
                out_slice[in_S, 1] = T_n[1] + xy_S[:, 1]
            else:
                out_slice[in_S] = T_n

        # --- Climb ---
        if np.any(in_C):
            c_t = t_in_cycle[in_C] - tau_T_n - tau_S_n
            c_t = np.clip(c_t, 0.0, tau_C_n)
            if config.climb_motion is not None:
                r0 = config.climb_motion.thermal_radius
                T_turn_n = traj.climb_turn_periods[n]
                omega_n = 2 * np.pi / T_turn_n
                v_drift = config.climb_motion.v_drift
                phi0 = traj.climb_phase0[n]
                phi_drift = traj.climb_drift_angles[n]

                # Oscillator displacement from start of climb
                osc_dx = r0 * (np.cos(omega_n * c_t + phi0) - np.cos(phi0))
                osc_dy = r0 * (np.sin(omega_n * c_t + phi0) - np.sin(phi0))
                # Linear drift
                drift_dx = v_drift * c_t * np.cos(phi_drift)
                drift_dy = v_drift * c_t * np.sin(phi_drift)

                out_slice[in_C, 0] = S_n[0] + osc_dx + drift_dx
                out_slice[in_C, 1] = S_n[1] + osc_dy + drift_dy
            else:
                out_slice[in_C] = S_n

        out[mask] = out_slice

    return out


def _ctrw_positions_at_times(
    leg_durations: np.ndarray,
    leg_angles: np.ndarray,
    velocity: float,
    times: np.ndarray,
) -> np.ndarray:
    """Position of a piecewise-linear CTRW at queried times.

    The CTRW starts at the origin at time 0, and during each leg moves
    at constant velocity ``velocity`` in direction ``leg_angles[i]``
    for a duration ``leg_durations[i]``. Returns positions at
    ``times``, linearly interpolated inside legs.

    Parameters
    ----------
    leg_durations : ndarray, shape (L,)
    leg_angles : ndarray, shape (L,)
    velocity : float
    times : ndarray, shape (N,)
        Query times in [0, sum(leg_durations)]. Out-of-range times
        are clipped.
    """
    n = len(times)
    if len(leg_durations) == 0 or n == 0:
        return np.zeros((n, 2))

    leg_ends = np.cumsum(leg_durations)
    total_duration = leg_ends[-1]
    t_clip = np.clip(times, 0.0, total_duration)

    # Positions at the end of each leg
    leg_dx = velocity * leg_durations * np.cos(leg_angles)
    leg_dy = velocity * leg_durations * np.sin(leg_angles)
    cum_x = np.concatenate(([0.0], np.cumsum(leg_dx)))
    cum_y = np.concatenate(([0.0], np.cumsum(leg_dy)))
    leg_starts = np.concatenate(([0.0], leg_ends[:-1]))

    # Find the leg containing each query time
    leg_idx = np.searchsorted(leg_ends, t_clip, side="right")
    leg_idx = np.clip(leg_idx, 0, len(leg_durations) - 1)

    # Linear interpolation within the leg
    t_rel = t_clip - leg_starts[leg_idx]
    x_start = cum_x[leg_idx]
    y_start = cum_y[leg_idx]
    x = x_start + velocity * t_rel * np.cos(leg_angles[leg_idx])
    y = y_start + velocity * t_rel * np.sin(leg_angles[leg_idx])

    return np.column_stack([x, y])


def _search_ctrw_positions_at_times(
    leg_durations: np.ndarray,
    wait_durations: np.ndarray,
    leg_angles: np.ndarray,
    leg_start_times: np.ndarray,
    v_c: float,
    times: np.ndarray,
) -> np.ndarray:
    r"""Position of a subordinated Lévy walk at queried times.

    The walker starts at the origin at phase time 0. For each index
    ``k``, the walker moves ballistically at speed ``v_c`` in direction
    ``leg_angles[k]`` from ``leg_start_times[k]`` for a duration
    ``leg_durations[k]`` (the *leg*), then sits still for
    ``wait_durations[k]`` (the *wait*), before the next leg begins.

    Parameters
    ----------
    leg_durations, wait_durations, leg_angles, leg_start_times :
        Arrays of the same length produced by :func:`_sample_ctrw_legs`.
    v_c : float
        Walker speed during ballistic legs.
    times : ndarray, shape (N,)
        Query times in [0, total_phase_duration].

    Returns
    -------
    ndarray, shape (N, 2)
    """
    n = len(times)
    if len(leg_durations) == 0 or n == 0:
        return np.zeros((n, 2))

    # Endpoint of each leg (position at leg_start + leg_duration)
    jx = v_c * leg_durations * np.cos(leg_angles)
    jy = v_c * leg_durations * np.sin(leg_angles)
    cum_x_end = np.cumsum(jx)
    cum_y_end = np.cumsum(jy)
    # Starting position of leg k (cumulative up through leg k-1)
    start_x = np.concatenate(([0.0], cum_x_end[:-1]))
    start_y = np.concatenate(([0.0], cum_y_end[:-1]))
    # Leg end times (t when the leg finishes moving)
    leg_end_times = leg_start_times + leg_durations

    total_duration = leg_start_times[-1] + leg_durations[-1] + wait_durations[-1]
    t_clip = np.clip(times, 0.0, total_duration)

    # For each query time, determine which leg/wait it falls in.
    # Leg k is active for t in [leg_start_times[k], leg_end_times[k])
    # Wait k follows, for t in [leg_end_times[k], leg_end_times[k] + wait_durations[k])
    # We use searchsorted on leg_start_times to find the relevant leg index.
    idx = np.searchsorted(leg_start_times, t_clip, side="right") - 1
    idx = np.clip(idx, 0, len(leg_durations) - 1)

    x_out = np.zeros(n)
    y_out = np.zeros(n)
    for i in range(n):
        k = idx[i]
        t_local = t_clip[i] - leg_start_times[k]
        if t_local <= leg_durations[k]:
            # Inside the ballistic leg k
            x_out[i] = start_x[k] + v_c * t_local * np.cos(leg_angles[k])
            y_out[i] = start_y[k] + v_c * t_local * np.sin(leg_angles[k])
        else:
            # Inside the wait after leg k: walker at leg-k end position
            x_out[i] = cum_x_end[k]
            y_out[i] = cum_y_end[k]

    return np.column_stack([x_out, y_out])
