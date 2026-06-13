"""Transport observables: ensemble-averaged MSD and Hurst-exponent fits.

The MSD estimator used throughout this codebase is the **pure
ensemble-averaged** (EA) MSD::

    ⟨δ²(Δ)⟩_EA = ⟨ |r_m(Δ) − r_m(0)|² ⟩_m,

i.e. for each lag Δ we take a single pair (origin at t = 0, endpoint at
t = Δ) from each realisation m and average across the ensemble. No
time-averaging inside a single trajectory is performed in the
production scripts: for CTRWs with α<1 the time-averaged MSD does not
converge to the EA-MSD (weak ergodicity breaking), so EA is the
correct estimator for the subdiffusive regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "HurstFit",
    "msd_ensemble",
    "msd_ensemble_percentiles",
    "fit_hurst",
]


@dataclass(frozen=True)
class HurstFit:
    """Result of a Hurst-exponent fit on log(MSD) vs log(Δ).

    Attributes
    ----------
    hurst : float
        Fitted Hurst exponent. Defined by MSD ∝ Δ^{2H}; so the slope of
        log(MSD) vs log(Δ) equals 2H.
    slope : float
        Fitted log-log slope (i.e. 2 × hurst).
    intercept : float
        Fitted log-log intercept (log amplitude).
    fit_range : tuple[float, float]
        The (min, max) lag in seconds actually used for the fit.
    n_points : int
        Number of lag values used.
    slope_err : float
        Standard error of the fitted log-log slope from the ordinary
        least-squares regression (``NaN`` if fewer than 3 points). The
        Hurst-exponent standard error is ``slope_err / 2``.
    """

    hurst: float
    slope: float
    intercept: float
    fit_range: tuple[float, float]
    n_points: int
    slope_err: float = float("nan")

    @property
    def hurst_err(self) -> float:
        """Standard error of the Hurst exponent, ``slope_err / 2``."""
        return self.slope_err / 2.0


def msd_ensemble(ensemble: np.ndarray) -> np.ndarray:
    r"""Pure ensemble-averaged MSD.

    For each lag ``k`` the MSD is the mean over trajectories of the
    squared displacement from the trajectory's own origin:

    .. math::
        \langle\delta^2(k)\rangle_{\mathrm{EA}}
        = \frac{1}{M}\sum_{m=1}^{M} \bigl\lvert r_m(k) - r_m(0)\bigr\rvert^2 .

    No internal time-average is performed: each trajectory contributes
    exactly one pair ``(0, k)`` to the lag-``k`` estimate. This is the
    appropriate estimator for non-ergodic processes such as a CTRW with
    α<1 (where TA-MSD and EA-MSD do not coincide).

    Parameters
    ----------
    ensemble : ndarray, shape (M, N, d)
        ``M`` trajectories of length ``N`` in dimension ``d``. The
        origin of each trajectory is ``ensemble[m, 0, :]``.

    Returns
    -------
    ndarray, shape (N,)
        EA-MSD at each lag ``k = 0, 1, …, N-1``. ``out[0] = 0`` by
        construction.
    """
    if ensemble.ndim != 3:
        raise ValueError(
            "ensemble must have shape (n_trajectories, n_steps, d), "
            f"got shape {ensemble.shape}"
        )
    ensemble = np.asarray(ensemble, dtype=float)
    disp = ensemble - ensemble[:, 0:1, :]      # (M, N, d)
    sq = np.sum(disp * disp, axis=2)            # (M, N)
    return sq.mean(axis=0)                       # (N,)


def _squared_displacements(ensemble: np.ndarray) -> np.ndarray:
    """Return the (M, N) array of squared displacements from each
    trajectory's own origin. Shared by the MSD point estimate and its
    error estimators."""
    if ensemble.ndim != 3:
        raise ValueError(
            "ensemble must have shape (n_trajectories, n_steps, d), "
            f"got shape {ensemble.shape}"
        )
    ensemble = np.asarray(ensemble, dtype=float)
    disp = ensemble - ensemble[:, 0:1, :]
    return np.sum(disp * disp, axis=2)


def msd_ensemble_percentiles(
    ensemble: np.ndarray,
    q_lo: float = 5.0,
    q_hi: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""EA-MSD with the direct sample-percentile band.

    Returns ``(msd, lo, hi)`` where ``lo``/``hi`` are the ``q_lo``-th /
    ``q_hi``-th percentiles (default 5--95) of the per-trajectory squared
    displacements ``|r_m(Δ)-r_m(0)|^2`` at each lag --- the same ``M``
    samples whose mean is the EA-MSD. No resampling is involved: the
    band is computed directly with ``np.percentile`` over the ensemble.

    This band describes the spread of individual flights around the
    mean, not the standard error of the mean. For the heavy-tailed
    classes (``mu_T < 4``) the distribution is strongly skewed, so the
    mean typically runs in the upper part of the band.
    """
    if not (0.0 <= q_lo < q_hi <= 100.0):
        raise ValueError(f"require 0 <= q_lo < q_hi <= 100, got ({q_lo}, {q_hi})")
    sq = _squared_displacements(ensemble)
    msd = sq.mean(axis=0)
    lo, hi = np.percentile(sq, [q_lo, q_hi], axis=0)
    return msd, lo, hi


def fit_hurst(
    lags: np.ndarray,
    msd: np.ndarray,
    lag_range: tuple[float, float],
) -> HurstFit:
    """Fit a power-law to the MSD over a specified lag range.

    Parameters
    ----------
    lags : ndarray
        Time lags (in seconds, or any consistent unit). Must exclude
        ``Δ = 0``.
    msd : ndarray
        MSD values at the corresponding lags. Same length as ``lags``.
    lag_range : tuple[float, float]
        Inclusive (lag_min, lag_max) over which to perform the log-log
        linear fit.

    Returns
    -------
    HurstFit
        Fitted parameters and bookkeeping.
    """
    lags = np.asarray(lags, dtype=float)
    msd = np.asarray(msd, dtype=float)
    if lags.shape != msd.shape:
        raise ValueError(
            f"lags and msd must have same shape, got {lags.shape} vs {msd.shape}"
        )
    if np.any(lags <= 0):
        raise ValueError("fit range must exclude zero and negative lags")

    lag_min, lag_max = lag_range
    if lag_min <= 0 or lag_max <= lag_min:
        raise ValueError(
            f"invalid lag_range {lag_range!r}: require 0 < lag_min < lag_max"
        )

    mask = (lags >= lag_min) & (lags <= lag_max) & (msd > 0)
    if mask.sum() < 2:
        raise ValueError(
            f"Not enough points in lag range {lag_range!r} for a linear fit "
            f"(got {mask.sum()})."
        )

    log_lags = np.log(lags[mask])
    log_msd = np.log(msd[mask])
    slope, intercept = np.polyfit(log_lags, log_msd, 1)

    # Standard error of the OLS slope from the residuals (unweighted fit):
    # SE = sqrt( (Σresid² / (n-2)) / Σ(x-x̄)² ). NaN for fewer than 3 points.
    n = log_lags.size
    if n > 2:
        resid = log_msd - (slope * log_lags + intercept)
        sxx = float(np.sum((log_lags - log_lags.mean()) ** 2))
        slope_err = (
            float(np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx))
            if sxx > 0.0
            else float("nan")
        )
    else:
        slope_err = float("nan")

    return HurstFit(
        hurst=slope / 2.0,
        slope=slope,
        intercept=intercept,
        fit_range=(lag_min, lag_max),
        n_points=int(mask.sum()),
        slope_err=slope_err,
    )
