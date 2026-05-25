"""Monte Carlo simulation of the cycle-based CTRW soaring-flight model.

A single flight is a renewal sequence of soaring *cycles*. Each cycle
contains three phases (transition, search, climb) generated as follows.

**Transition (T).** A straight-line glide at constant speed ``v_xy`` in
direction ``theta_n``. The cycle-to-cycle heading evolves as
``theta_n = theta_{n-1} + eta_n`` with ``eta_n ~ N(0, sigma_theta^2)``.
The initial heading ``theta_0`` is drawn uniformly on ``[0, 2*pi)``
unless ``AngularConfig.theta0`` is explicitly set.

**Search (S).** A local CTRW with physical-duration stopping.

    - Initial local heading ``psi_0 ~ U(0, 2*pi)``, independent of
      ``theta_n``.
    - At each step ``j`` the pilot performs a ballistic relocation of
      duration ``tau_b_j ~ Exp(tau_b_S)`` at speed ``u_S`` in
      direction ``psi_j``, then waits for a turning time
      ``tau_turn_j`` drawn from a Mittag-Leffler with stability
      ``alpha_S`` and scale ``tau_turn_S`` (walker stationary).
    - Direction update *after a fully completed turning wait*:
      ``psi_{j+1} = psi_j + eps_j * Omega_S * tau_turn_j``,
      ``eps_j = +-1`` equiprobable.
    - The local CTRW stops when the cumulative *physical* time
      ``sum(tau_b) + sum(tau_turn)`` reaches the Lomax-sampled search
      duration ``tau_S_n``. Whichever component (leg or wait) straddles
      ``tau_S_n`` is truncated so that
      ``T_phys^S = sum(tau_b) + sum(tau_turn) = tau_S_n`` exactly.
    - As a consequence, the search-phase physical duration coincides
      with the Lomax draw (no heavy-tail blow-up from infinite-mean
      Mittag-Leffler waits).

**Climb (C).** Circular motion plus drift, all starting at the end of
the search.

    - Per-cycle turn period ``T_turn_n ~ N(T_turn_mean, T_turn_std^2)``,
      clipped at ``0.2 * T_turn_mean`` for positivity. Angular
      frequency ``omega_n = 2*pi / T_turn_n``.
    - Initial orbital phase ``phi_0 ~ U(0, 2*pi)``, independent of
      ``theta_n``.
    - Slow orographic drift of magnitude ``v_drift`` along a direction
      ``phi_drift ~ U(0, 2*pi)`` drawn independently per cycle.

Two simulation modes are provided.

- :func:`simulate_single`: returns a :class:`CycleTrajectory` with the
  end-of-cycle positions and the bookkeeping needed to reconstruct the
  full continuous-time trajectory on demand (random angles, search
  legs, climb periods).
- :func:`simulate_ensemble`: returns an array of trajectories sampled
  on a regular time grid, suitable for direct MSD computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from distributions import MittagLeffler
from model import SoaringConfig

__all__ = [
    "CycleTrajectory",
    "SearchEpisode",
    "simulate_single",
    "simulate_ensemble",
    "interpolate_trajectory",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchEpisode:
    """Bookkeeping of one search-phase realisation.

    Attributes
    ----------
    leg_durations : ndarray, shape (L,)
        Duration of each ballistic relocation (s). If the episode ends
        inside a leg, the last entry is truncated so the total physical
        time matches the Lomax-sampled search duration ``tau_S_n``.
    leg_angles : ndarray, shape (L,)
        Heading of each ballistic relocation (rad).
    leg_start_phys : ndarray, shape (L,)
        Phase-local physical start time of each leg (s), with
        ``leg_start_phys[0] = 0``.
    turn_durations : ndarray, shape (L,)
        Turning-time wait that follows each leg (s). The last entry
        is 0 if the episode ended inside a leg (no following wait), or
        truncated if the episode ended inside a wait.
    T_phys : float
        Total physical time occupied by the search episode (s); equals
        ``tau_S_n`` by construction (up to floating-point round-off).
    """

    leg_durations: np.ndarray
    leg_angles: np.ndarray
    leg_start_phys: np.ndarray
    turn_durations: np.ndarray
    T_phys: float

    def endpoint(self, u_S: float) -> tuple[float, float]:
        """Cartesian net displacement produced by the ballistic legs."""
        dx = u_S * np.sum(self.leg_durations * np.cos(self.leg_angles))
        dy = u_S * np.sum(self.leg_durations * np.sin(self.leg_angles))
        return float(dx), float(dy)


@dataclass(frozen=True)
class CycleTrajectory:
    """One realisation of the cycle-based CTRW.

    Attributes
    ----------
    positions : ndarray, shape (N+1, 2)
        Horizontal position at the end of each complete cycle, with
        ``positions[0] = 0``.
    cycle_end_times : ndarray, shape (N+1,)
        Physical time at the end of each cycle (s), with
        ``cycle_end_times[0] = 0``.
    headings : ndarray, shape (N,)
        Transition-phase heading angle of each cycle (rad).
    phase_durations_active : ndarray, shape (N, 3)
        Columns ``(tau_T, tau_S, tau_C)`` of the Lomax/exponential
        draws. Under the physical-duration stopping rule the search
        column also coincides with the physical search duration
        ``T_phys^S`` of the corresponding episode.
    search_episodes : list[SearchEpisode]
        Local-CTRW bookkeeping for each cycle.
    climb_initial_phase : ndarray, shape (N,)
        Initial orbital phase ``phi_0`` of the thermalling oscillator
        for each cycle (rad).
    climb_drift_angle : ndarray, shape (N,)
        Direction of the orographic drift during each cycle (rad).
    climb_turn_period : ndarray, shape (N,)
        Per-cycle (clipped Gaussian) turn period (s).
    """

    positions: np.ndarray
    cycle_end_times: np.ndarray
    headings: np.ndarray
    phase_durations_active: np.ndarray
    search_episodes: list[SearchEpisode]
    climb_initial_phase: np.ndarray
    climb_drift_angle: np.ndarray
    climb_turn_period: np.ndarray

    @property
    def n_cycles(self) -> int:
        return len(self.headings)

    @property
    def total_time(self) -> float:
        return float(self.cycle_end_times[-1])

    @property
    def search_T_phys(self) -> np.ndarray:
        """Per-cycle physical search durations (sum of legs + turns)."""
        return np.array([ep.T_phys for ep in self.search_episodes])


# ---------------------------------------------------------------------------
# Per-phase sampling
# ---------------------------------------------------------------------------


def _sample_search_episode(
    tau_S_n: float,
    cfg_search: "object",  # SearchMotionConfig
    rng: np.random.Generator,
) -> SearchEpisode:
    r"""Generate one search episode with the physical-duration stopping rule.

    The local CTRW alternates ballistic legs (speed ``u_S``,
    exponential duration ``tau_b_S``) and Mittag-Leffler turning waits
    (stability ``alpha_S``, scale ``tau_turn_S``). The episode ends
    when the cumulative *physical* time (legs + waits) reaches the
    Lomax-sampled search duration ``tau_S_n``. Whichever component
    (leg or wait) straddles ``tau_S_n`` is truncated so that
    ``T_phys = sum(leg_durations) + sum(turn_durations) = tau_S_n``
    exactly.

    Direction update (paper Eq. 8) is applied only after a fully
    completed turning wait, never after a truncated one (no subsequent
    leg makes the rotation observable).
    """
    if tau_S_n <= 0.0:
        return SearchEpisode(
            leg_durations=np.zeros(0),
            leg_angles=np.zeros(0),
            leg_start_phys=np.zeros(0),
            turn_durations=np.zeros(0),
            T_phys=0.0,
        )

    u_S = cfg_search.u_S
    tau_b_S = cfg_search.tau_b_S
    tau_turn_S = cfg_search.tau_turn_S
    alpha_S = cfg_search.alpha_S
    Omega_S = cfg_search.Omega_S

    ml = MittagLeffler(alpha=alpha_S, tau_0=tau_turn_S)

    # Pre-sample in geometric batches. Expected leg count is bounded
    # above by tau_S_n / tau_b_S (zero-wait limit); waits make it
    # smaller, so this is a safe over-estimate.
    batch = max(16, int(3.0 * tau_S_n / tau_b_S))

    legs: list[float] = []
    turns: list[float] = []
    angles: list[float] = []
    start_phys: list[float] = []

    psi = float(rng.uniform(0.0, 2.0 * np.pi))
    elapsed_phys = 0.0

    while True:
        bal = rng.exponential(scale=tau_b_S, size=batch)
        tur = ml.sample(batch, rng)
        eps = rng.choice(np.array([-1.0, 1.0]), size=batch)

        for tau_b_j, tau_turn_j, eps_j in zip(bal, tur, eps):
            # Leg phase.
            remaining = tau_S_n - elapsed_phys
            if remaining <= 0.0:
                # Should not happen — the explicit returns below cover
                # every termination path.
                break

            if tau_b_j >= remaining:
                # Episode ends inside this leg; no following wait.
                legs.append(float(remaining))
                turns.append(0.0)
                angles.append(psi)
                start_phys.append(elapsed_phys)
                elapsed_phys += remaining
                return SearchEpisode(
                    leg_durations=np.array(legs),
                    leg_angles=np.array(angles),
                    leg_start_phys=np.array(start_phys),
                    turn_durations=np.array(turns),
                    T_phys=elapsed_phys,
                )

            # Full leg completes.
            legs.append(float(tau_b_j))
            angles.append(psi)
            start_phys.append(elapsed_phys)
            elapsed_phys += tau_b_j

            # Wait phase.
            remaining = tau_S_n - elapsed_phys
            if remaining <= 0.0:
                # Episode ends exactly at end-of-leg.
                turns.append(0.0)
                return SearchEpisode(
                    leg_durations=np.array(legs),
                    leg_angles=np.array(angles),
                    leg_start_phys=np.array(start_phys),
                    turn_durations=np.array(turns),
                    T_phys=elapsed_phys,
                )

            if tau_turn_j >= remaining:
                # Episode ends inside this wait. Truncate and stop.
                turns.append(float(remaining))
                elapsed_phys += remaining
                return SearchEpisode(
                    leg_durations=np.array(legs),
                    leg_angles=np.array(angles),
                    leg_start_phys=np.array(start_phys),
                    turn_durations=np.array(turns),
                    T_phys=elapsed_phys,
                )

            # Full wait completes.
            turns.append(float(tau_turn_j))
            elapsed_phys += tau_turn_j

            # Direction update (paper Eq. 8) after a full wait only.
            psi = psi + eps_j * Omega_S * tau_turn_j


def _sample_climb_geometry(
    n_cycles: int,
    cfg_climb: "object",  # ClimbMotionConfig
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample per-cycle climb parameters: ``T_turn_n``, ``phi_0_n``, ``phi_drift_n``."""
    T_mean = cfg_climb.T_turn_mean
    T_std = cfg_climb.T_turn_std

    if T_std > 0:
        T_turn = rng.normal(loc=T_mean, scale=T_std, size=n_cycles)
        T_turn = np.clip(T_turn, 0.2 * T_mean, None)
    else:
        T_turn = np.full(n_cycles, T_mean)

    phi0 = rng.uniform(0.0, 2.0 * np.pi, size=n_cycles)
    phi_drift = rng.uniform(0.0, 2.0 * np.pi, size=n_cycles)
    return T_turn, phi0, phi_drift


def _climb_position(
    t: np.ndarray,
    r0: float,
    omega: float,
    phi0: float,
    v_drift: float,
    phi_drift: float,
) -> np.ndarray:
    """Climb-relative position at intra-climb times ``t``.

    The motion is ``r0 * (cos(omega t + phi0) - cos phi0,
    sin(omega t + phi0) - sin phi0)`` plus ``v_drift * t * (cos phi_drift,
    sin phi_drift)``, so the climb starts at the origin of its own
    local frame and is continuous with the preceding search endpoint.
    """
    cos_phi0 = np.cos(phi0)
    sin_phi0 = np.sin(phi0)
    arg = omega * t + phi0
    x = r0 * (np.cos(arg) - cos_phi0) + v_drift * t * np.cos(phi_drift)
    y = r0 * (np.sin(arg) - sin_phi0) + v_drift * t * np.sin(phi_drift)
    return np.column_stack([x, y])


def _search_position(
    t: np.ndarray,
    episode: SearchEpisode,
    u_S: float,
) -> np.ndarray:
    """Search-relative position at intra-search physical times ``t``.

    During the ballistic leg ``k`` (time interval
    ``[leg_start_phys[k], leg_start_phys[k] + leg_durations[k]]``) the
    walker moves at speed ``u_S`` in direction ``leg_angles[k]``.
    During the subsequent turning wait the walker is stationary.
    """
    n = len(t)
    if n == 0 or episode.T_phys == 0.0:
        return np.zeros((n, 2))

    leg_d = episode.leg_durations
    angs = episode.leg_angles
    start = episode.leg_start_phys
    turn = episode.turn_durations

    # End-of-leg cumulative position (relative to search start).
    jx = u_S * leg_d * np.cos(angs)
    jy = u_S * leg_d * np.sin(angs)
    cum_x_end = np.cumsum(jx)
    cum_y_end = np.cumsum(jy)
    start_x = np.concatenate(([0.0], cum_x_end[:-1]))
    start_y = np.concatenate(([0.0], cum_y_end[:-1]))
    leg_end = start + leg_d

    t_clip = np.clip(t, 0.0, episode.T_phys)
    # leg index k such that start[k] <= t < start[k+1] (or last leg).
    k_arr = np.searchsorted(start, t_clip, side="right") - 1
    k_arr = np.clip(k_arr, 0, len(leg_d) - 1)

    x_out = np.empty(n)
    y_out = np.empty(n)
    for i in range(n):
        k = int(k_arr[i])
        t_local = t_clip[i] - start[k]
        if t_local <= leg_d[k]:
            x_out[i] = start_x[k] + u_S * t_local * np.cos(angs[k])
            y_out[i] = start_y[k] + u_S * t_local * np.sin(angs[k])
        else:
            # Inside the turning wait after leg k.
            x_out[i] = cum_x_end[k]
            y_out[i] = cum_y_end[k]
    return np.column_stack([x_out, y_out])


# ---------------------------------------------------------------------------
# Top-level simulation
# ---------------------------------------------------------------------------


def simulate_single(
    config: SoaringConfig,
    n_cycles: int,
    rng: np.random.Generator,
) -> CycleTrajectory:
    """Simulate one trajectory of exactly ``n_cycles`` cycles."""
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be positive, got {n_cycles}")

    tau_T = config.transition.build().sample(n_cycles, rng)
    tau_S = config.search.build().sample(n_cycles, rng)
    tau_C = config.climb.build().sample(n_cycles, rng)

    # --- Transition headings: GRW on the circle, isotropic theta_0 ----
    if config.angular.theta0 is None:
        theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
    else:
        theta0 = float(config.angular.theta0)
    eta = rng.normal(loc=0.0, scale=config.angular.sigma_theta, size=n_cycles)
    headings = np.empty(n_cycles)
    headings[0] = theta0
    if n_cycles > 1:
        headings[1:] = theta0 + np.cumsum(eta[1:])
    # (eta[0] is unused: theta_0 is the prior, not theta_{-1} + eta_0.)

    # --- Search episodes (active-budget local CTRW) -------------------
    if config.search_motion is not None:
        search_episodes: list[SearchEpisode] = [
            _sample_search_episode(tau_S[n], config.search_motion, rng)
            for n in range(n_cycles)
        ]
        T_phys_S = np.array([ep.T_phys for ep in search_episodes])
        u_S = config.search_motion.u_S
    else:
        # Bare-cycle mode: search phase has duration tau_S (Lomax) and
        # contributes no horizontal displacement. We still record empty
        # SearchEpisode objects so the trajectory bookkeeping is uniform.
        search_episodes = [
            SearchEpisode(
                leg_durations=np.zeros(0),
                leg_angles=np.zeros(0),
                leg_start_phys=np.zeros(0),
                turn_durations=np.zeros(0),
                T_phys=float(tau_S[n]),
            )
            for n in range(n_cycles)
        ]
        T_phys_S = tau_S.copy()
        u_S = 0.0

    # --- Climb geometry -----------------------------------------------
    if config.climb_motion is not None:
        T_turn_n, phi0_n, phi_drift_n = _sample_climb_geometry(
            n_cycles, config.climb_motion, rng
        )
        r0 = config.climb_motion.r0
        v_drift = config.climb_motion.v_drift
    else:
        # Bare-cycle mode: no circular motion, no drift.
        T_turn_n = np.ones(n_cycles)  # arbitrary, unused with r0=0
        phi0_n = np.zeros(n_cycles)
        phi_drift_n = np.zeros(n_cycles)
        r0 = 0.0
        v_drift = 0.0

    # --- Per-cycle net displacements ----------------------------------
    dx_T = config.v_xy * tau_T * np.cos(headings)
    dy_T = config.v_xy * tau_T * np.sin(headings)

    dx_S = np.zeros(n_cycles)
    dy_S = np.zeros(n_cycles)
    for n, ep in enumerate(search_episodes):
        dxs, dys = ep.endpoint(u_S)
        dx_S[n] = dxs
        dy_S[n] = dys

    omega_n = 2.0 * np.pi / T_turn_n
    arg_end = omega_n * tau_C + phi0_n
    osc_dx = r0 * (np.cos(arg_end) - np.cos(phi0_n))
    osc_dy = r0 * (np.sin(arg_end) - np.sin(phi0_n))
    drift_dx = v_drift * tau_C * np.cos(phi_drift_n)
    drift_dy = v_drift * tau_C * np.sin(phi_drift_n)
    dx_C = osc_dx + drift_dx
    dy_C = osc_dy + drift_dy

    # --- Stitch into a global trajectory ------------------------------
    dx_cycle = dx_T + dx_S + dx_C
    dy_cycle = dy_T + dy_S + dy_C
    x = np.concatenate(([0.0], np.cumsum(dx_cycle)))
    y = np.concatenate(([0.0], np.cumsum(dy_cycle)))
    positions = np.column_stack([x, y])

    cycle_durations = tau_T + T_phys_S + tau_C
    cycle_end_times = np.concatenate(([0.0], np.cumsum(cycle_durations)))

    return CycleTrajectory(
        positions=positions,
        cycle_end_times=cycle_end_times,
        headings=headings,
        phase_durations_active=np.column_stack([tau_T, tau_S, tau_C]),
        search_episodes=search_episodes,
        climb_initial_phase=phi0_n,
        climb_drift_angle=phi_drift_n,
        climb_turn_period=T_turn_n,
    )


def interpolate_trajectory(
    traj: CycleTrajectory,
    config: SoaringConfig,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpolate a :class:`CycleTrajectory` on the regular time grid
    ``target_times``.

    Within each cycle the position evolves piecewise:

    - **Transition**: straight line at ``v_xy * ê(theta_n)``.
    - **Search**: piecewise according to the local-CTRW legs (constant
      velocity inside each leg, stationary during turning waits).
    - **Climb**: circular motion + drift from the search endpoint.

    Returns
    -------
    ndarray, shape ``(len(target_times), 2)``
    """
    n_cycles = traj.n_cycles
    cycle_end_times = traj.cycle_end_times
    tau_T = traj.phase_durations_active[:, 0]
    T_phys_S = traj.search_T_phys
    tau_C = traj.phase_durations_active[:, 2]

    out = np.zeros((len(target_times), 2))

    cycle_idx = np.searchsorted(cycle_end_times, target_times, side="right") - 1
    cycle_idx = np.clip(cycle_idx, 0, n_cycles - 1)

    u_S = config.search_motion.u_S if config.search_motion is not None else 0.0
    r0 = config.climb_motion.r0 if config.climb_motion is not None else 0.0
    v_drift = (
        config.climb_motion.v_drift if config.climb_motion is not None else 0.0
    )

    for n in range(n_cycles):
        mask = cycle_idx == n
        if not np.any(mask):
            continue

        t_in_cycle = target_times[mask] - cycle_end_times[n]
        theta_n = traj.headings[n]
        P_n = traj.positions[n]

        # Endpoint of transition.
        dxT = config.v_xy * tau_T[n] * np.cos(theta_n)
        dyT = config.v_xy * tau_T[n] * np.sin(theta_n)
        T_end = P_n + np.array([dxT, dyT])

        # Endpoint of search.
        ep = traj.search_episodes[n]
        dxS, dyS = ep.endpoint(u_S)
        S_end = T_end + np.array([dxS, dyS])

        # Classify times by phase boundary.
        boundary_TS = tau_T[n]
        boundary_SC = tau_T[n] + T_phys_S[n]
        boundary_end = boundary_SC + tau_C[n]

        in_T = t_in_cycle < boundary_TS
        in_S = (~in_T) & (t_in_cycle < boundary_SC)
        in_C = ~(in_T | in_S)

        seg = np.empty((len(t_in_cycle), 2))

        if np.any(in_T):
            t_T = t_in_cycle[in_T]
            seg[in_T, 0] = P_n[0] + config.v_xy * t_T * np.cos(theta_n)
            seg[in_T, 1] = P_n[1] + config.v_xy * t_T * np.sin(theta_n)

        if np.any(in_S):
            if config.search_motion is not None and ep.T_phys > 0:
                t_S = t_in_cycle[in_S] - boundary_TS
                xy_S = _search_position(t_S, ep, u_S)
                seg[in_S, 0] = T_end[0] + xy_S[:, 0]
                seg[in_S, 1] = T_end[1] + xy_S[:, 1]
            else:
                seg[in_S] = T_end

        if np.any(in_C):
            if config.climb_motion is not None:
                t_C = np.clip(t_in_cycle[in_C] - boundary_SC, 0.0, tau_C[n])
                omega = 2.0 * np.pi / traj.climb_turn_period[n]
                xy_C = _climb_position(
                    t=t_C,
                    r0=r0,
                    omega=omega,
                    phi0=traj.climb_initial_phase[n],
                    v_drift=v_drift,
                    phi_drift=traj.climb_drift_angle[n],
                )
                seg[in_C] = S_end + xy_C
            else:
                seg[in_C] = S_end

        # Points past the last cycle are pinned to the end of the trajectory.
        past_end = t_in_cycle >= boundary_end
        if np.any(past_end):
            # Use the cycle-end position from the global trajectory to remain
            # consistent with positions[n + 1].
            seg[past_end] = traj.positions[n + 1]

        out[mask] = seg

    return out


def _estimate_mean_cycle_duration(config: SoaringConfig) -> float:
    """Heuristic for the mean physical cycle duration (used to budget
    the number of cycles per trajectory).

    Under the physical-duration stopping rule the search-phase physical
    time equals the Lomax-sampled ``tau_S^n`` (its mean is the Lomax
    mean), in both bare and full configurations. The transition and
    climb physical durations also coincide with their Lomax /
    exponential draws.
    """
    s_T = config.transition.build().mean
    s_S = config.search.build().mean
    s_C = config.climb.build().mean

    return float(_finite(s_T) + _finite(s_S) + _finite(s_C))


def _finite(x: float) -> float:
    return x if np.isfinite(x) else 1e3


def simulate_ensemble(
    config: SoaringConfig,
    n_trajectories: int,
    total_time: float,
    dt: float,
    rng: np.random.Generator,
    cycles_safety_factor: float = 3.0,
) -> np.ndarray:
    """Generate ``n_trajectories`` trajectories of ``total_time`` seconds
    sampled on a regular grid of step ``dt``.

    Returns an array of shape ``(n_trajectories, n_steps, 2)`` with
    ``n_steps = int(total_time / dt) + 1``.
    """
    if n_trajectories <= 0:
        raise ValueError(f"n_trajectories must be positive, got {n_trajectories}")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    mean_cycle = _estimate_mean_cycle_duration(config)
    n_cycles_initial = max(2, int(cycles_safety_factor * total_time / mean_cycle))

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
        ensemble[i] = interpolate_trajectory(traj, config, time_grid)
    return ensemble
