"""Closed-form (analytical) expressions of the manuscript.

Single source of truth for every analytical formula used by the
comparison scripts and by the test suite. Equation labels refer to the
companion manuscript (Di Sante 2026, ``soaring_ctrw.tex``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Manuscript equation
   * - ``lomax_mean``
     - Lomax mean ``tau_0 / (mu - 1)`` (after Eq. ``eq:lomax``)
   * - ``lomax_var``
     - Lomax variance (finite for ``mu > 2``)
   * - ``lomax_alpha_moment``
     - fractional moment ``<tau^alpha>`` (Appendix A)
   * - ``search_msd_short`` / ``search_msd_long``
     - Eq. ``eq:msd-search`` asymptotic branches
   * - ``climb_msd_theory``
     - Eq. ``eq:msd-climb`` (first-order T→omega)
   * - ``climb_msd_exact``
     - numerical average over the *clipped-Gaussian* turn period
       actually sampled by the simulator (no first-order expansion;
       Appendix D)
   * - ``G_N``
     - Eq. ``eq:GN`` / ``eq:GN-closed``
   * - ``G_N_prime``
     - ``dG_N/dN`` (used by Eq. ``eq:Heff-convex``)
   * - ``compute_Sigma_S``
     - Eq. ``eq:SigmaS``
   * - ``compute_Sigma_C``
     - Eq. ``eq:SigmaC`` — evaluated in the *exact* Faddeeva form (the
       manuscript's plateau formula is its ``O(10^-3)`` approximation)
   * - ``compute_AB``
     - amplitudes ``A``, ``B`` of Eq. ``eq:msd-closed``
   * - ``Heff_theory_N``
     - Eq. ``eq:Heff-convex``
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfcx, gamma as gamma_fn
from scipy.stats import norm

from .model import SoaringConfig

__all__ = [
    "lomax_mean",
    "lomax_var",
    "lomax_alpha_moment",
    "search_msd_short",
    "search_msd_long",
    "climb_msd_theory",
    "climb_msd_exact",
    "G_N",
    "G_N_prime",
    "compute_Sigma_S",
    "compute_Sigma_C",
    "compute_AB",
    "Heff_theory_N",
]

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Lomax moments
# ---------------------------------------------------------------------------


def lomax_mean(mu: float, tau_0: float) -> float:
    """Lomax mean ``tau_0 / (mu - 1)`` (finite for ``mu > 1``)."""
    return tau_0 / (mu - 1.0)


def lomax_var(mu: float, tau_0: float) -> float:
    """Lomax variance ``mu tau_0^2 / ((mu-1)^2 (mu-2))`` (finite for ``mu > 2``)."""
    if mu <= 2:
        return float("inf")
    return mu * tau_0 ** 2 / ((mu - 1.0) ** 2 * (mu - 2.0))


def lomax_alpha_moment(alpha: float, mu: float, tau_0: float) -> float:
    """``E[tau^alpha]`` for Lomax ``(mu, tau_0)``; finite for ``mu > alpha``.

    Beta-integral identity of Appendix A:
    ``<tau^alpha> = tau_0^alpha Gamma(1+alpha) Gamma(mu-alpha) / Gamma(mu)``.
    """
    return tau_0 ** alpha * gamma_fn(alpha + 1) * gamma_fn(mu - alpha) / gamma_fn(mu)


# ---------------------------------------------------------------------------
# Eq. eq:msd-search — conditional search MSD (asymptotic branches)
# ---------------------------------------------------------------------------


def search_msd_short(lags: np.ndarray, u_S: float) -> np.ndarray:
    """Short-lag ballistic branch ``u_S^2 Delta^2`` of Eq. ``eq:msd-search``."""
    return u_S ** 2 * lags ** 2


def search_msd_long(lags: np.ndarray, alpha_S: float, tau_b_S: float,
                    tau_turn_S: float, u_S: float) -> np.ndarray:
    """Long-lag subdiffusive branch of Eq. ``eq:msd-search``.

    Leading-order Montroll--Weiss prefactor (instantaneous-jump limit
    ``tau_b^S << tau_turn^S``); the exponent ``alpha_S`` is exact.
    """
    return (
        2.0 * u_S ** 2 * tau_b_S ** 2
        / (tau_turn_S ** alpha_S * gamma_fn(1.0 + alpha_S))
        * lags ** alpha_S
    )


# ---------------------------------------------------------------------------
# Eq. eq:msd-climb — climb MSD
# ---------------------------------------------------------------------------


def climb_msd_theory(lags: np.ndarray, r0: float, T_turn_mean: float,
                     T_turn_std: float, v_drift: float) -> np.ndarray:
    """Closed form Eq. ``eq:msd-climb`` (first-order ``T -> omega`` Gaussian).

    Accurate to a few percent at short and long lags; near the turn lag
    ``Delta ~ T_turn_mean`` it deviates from the exact period average by
    up to ~10-15% for ``T_turn_std/T_turn_mean ~ 0.3-0.4`` (use
    :func:`climb_msd_exact` for the distribution actually simulated).
    """
    omega_bar = 2.0 * np.pi / T_turn_mean
    sigma_omega = omega_bar * T_turn_std / T_turn_mean
    return (
        2.0 * r0 ** 2
        * (1.0 - np.exp(-0.5 * sigma_omega ** 2 * lags ** 2)
                * np.cos(omega_bar * lags))
        + v_drift ** 2 * lags ** 2
    )


def climb_msd_exact(lags: np.ndarray, r0: float, T_turn_mean: float,
                    T_turn_std: float, v_drift: float,
                    n_grid: int = 4001) -> np.ndarray:
    """Climb MSD with the circular term averaged *numerically* over the
    clipped-Gaussian turn period actually sampled by the simulator.

    The simulator draws ``T ~ N(T_turn_mean, T_turn_std^2)`` clipped at
    ``0.2 T_turn_mean`` (an atom of mass ``Phi((0.2-1) T_mean/T_std)``
    sits at the clip). This routine computes
    ``<cos(2 pi Delta / T)>_T`` by direct quadrature over that law —
    no first-order ``T -> omega`` expansion — and adds the drift term.
    It is the reference curve against which the accuracy of
    :func:`climb_msd_theory` is assessed in Appendix D.
    """
    lags = np.atleast_1d(np.asarray(lags, dtype=float))
    if T_turn_std <= 0.0:
        omega = 2.0 * np.pi / T_turn_mean
        return 2.0 * r0 ** 2 * (1.0 - np.cos(omega * lags)) \
            + v_drift ** 2 * lags ** 2

    lo = 0.2 * T_turn_mean
    hi = T_turn_mean + 8.0 * T_turn_std
    w_atom = float(norm.cdf((lo - T_turn_mean) / T_turn_std))
    T = np.linspace(lo, hi, n_grid)
    pdf = norm.pdf(T, loc=T_turn_mean, scale=T_turn_std)

    # <cos(2 pi Delta / T)> over the continuous part + the clip atom.
    cos_mat = np.cos(2.0 * np.pi * np.outer(lags, 1.0 / T))
    mean_cos = _trapezoid(cos_mat * pdf, T, axis=1)
    mean_cos = mean_cos + w_atom * np.cos(2.0 * np.pi * lags / lo)
    norm_mass = float(_trapezoid(pdf, T)) + w_atom
    mean_cos = mean_cos / norm_mass

    return 2.0 * r0 ** 2 * (1.0 - mean_cos) + v_drift ** 2 * lags ** 2


# ---------------------------------------------------------------------------
# Eqs. eq:GN / eq:msd-closed / eq:Heff-convex — coherent sum and exponent
# ---------------------------------------------------------------------------


def G_N(N: np.ndarray, rho: float) -> np.ndarray:
    """Closed form Eq. ``eq:GN``, evaluated at (possibly continuous) ``N``."""
    N = np.asarray(N, dtype=float)
    return N * (1.0 + rho) / (1.0 - rho) \
        - 2.0 * rho * (1.0 - rho ** N) / (1.0 - rho) ** 2


def G_N_prime(N: np.ndarray, rho: float) -> np.ndarray:
    """``dG_N/dN`` treating ``N`` as continuous (used by Eq. ``eq:Heff-convex``)."""
    N = np.asarray(N, dtype=float)
    return (1.0 + rho) / (1.0 - rho) \
        + 2.0 * rho ** (N + 1.0) * np.log(rho) / (1.0 - rho) ** 2


def compute_Sigma_S(cfg: SoaringConfig) -> float:
    """Per-cycle search displacement variance, Eq. ``eq:SigmaS``.

    Uses ``T_phys^S = tau_S^n`` (physical-duration stopping rule), so
    ``<(T_phys^S)^alpha_S>`` is the Lomax fractional moment.
    """
    sm = cfg.search_motion
    if sm is None:
        return 0.0
    mu_S = cfg.search.params["mu"]
    tau_0_S = cfg.search.params["tau_0"]
    T_phys_alpha = lomax_alpha_moment(sm.alpha_S, mu_S, tau_0_S)
    return (
        2.0 * sm.u_S ** 2 * sm.tau_b_S ** 2
        / (sm.tau_turn_S ** sm.alpha_S * gamma_fn(1.0 + sm.alpha_S))
        * T_phys_alpha
    )


def compute_Sigma_C(cfg: SoaringConfig) -> float:
    """Per-cycle climb displacement variance, Eq. ``eq:SigmaC`` (exact form).

    Computes ``Re<exp(i omega tau)>`` exactly with ``omega = omega_bar +
    delta_omega``, ``delta_omega ~ N(0, sigma_omega^2)`` and ``tau ~
    Exp(mu_C)`` independent, via the scaled complementary error function
    of complex argument (Faddeeva). The manuscript's plateau formula
    ``2 r0^2 + 2 v_drift^2 mu_C^2`` is the ``O(10^-3)`` approximation of
    this expression; it is this exact form that enters the figures.
    """
    cm = cfg.climb_motion
    if cm is None:
        return 0.0
    mu_C = cfg.climb.params["tau_mean"]
    omega_bar = 2.0 * np.pi / cm.T_turn_mean
    sigma_omega = omega_bar * cm.T_turn_std / cm.T_turn_mean
    if sigma_omega == 0.0:
        Re_phi = 1.0 / (1.0 + (omega_bar * mu_C) ** 2)
    else:
        z = (1.0 / mu_C - 1j * omega_bar) / (sigma_omega * np.sqrt(2.0))
        integral = np.sqrt(np.pi / 2.0) / (mu_C * sigma_omega) * erfcx(z)
        Re_phi = float(np.real(integral))
    return 2.0 * cm.r0 ** 2 * (1.0 - Re_phi) + 2.0 * cm.v_drift ** 2 * mu_C ** 2


def compute_AB(cfg: SoaringConfig) -> tuple[float, float, float, float]:
    """Return ``(A, B, rho, mean_T)`` of Eq. ``eq:msd-closed``.

    ``A = (v_xy <tau_T>)^2``, ``B = v_xy^2 Var(tau_T) + Sigma_S +
    Sigma_C``, ``rho = exp(-sigma_theta^2/2)``, and ``mean_T`` is the
    mean cycle duration ``<tau_T> + <tau_S> + <tau_C>``.
    """
    v_xy = cfg.v_xy
    mu_T = cfg.transition.params["mu"]
    tau_0_T = cfg.transition.params["tau_0"]
    mean_T_phase = lomax_mean(mu_T, tau_0_T)
    var_T_phase = lomax_var(mu_T, tau_0_T)
    mean_S_phase = lomax_mean(cfg.search.params["mu"], cfg.search.params["tau_0"])
    mean_C_phase = cfg.climb.params["tau_mean"]
    A = (v_xy * mean_T_phase) ** 2
    B = v_xy ** 2 * var_T_phase + compute_Sigma_S(cfg) + compute_Sigma_C(cfg)
    rho = float(np.exp(-cfg.angular.sigma_theta ** 2 / 2.0))
    return A, B, rho, mean_T_phase + mean_S_phase + mean_C_phase


def Heff_theory_N(N: np.ndarray, cfg: SoaringConfig) -> np.ndarray:
    """Eq. ``eq:Heff-convex`` evaluated at cycle count ``N`` (returns
    ``H_eff``, not ``2 H_eff``)."""
    A, B, rho, _ = compute_AB(cfg)
    GN = G_N(N, rho)
    GpN = G_N_prime(N, rho)
    s_G = N * GpN / GN
    w = A * GN / (A * GN + B * N)
    return 0.5 * (1.0 + w * (s_G - 1.0))
