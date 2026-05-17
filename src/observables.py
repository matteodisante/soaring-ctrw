"""Transport observables: time-averaged MSD and Hurst-exponent fits.

The functions here follow the same conventions used in Vilpellet et al.
The time-averaged MSD of a single trajectory is

    δ²(Δ) = (1 / (N − k)) · Σ_{t=0}^{N−k−1} |r(t+k) − r(t)|²,

with Δ = k·dt. We average further over an ensemble of trajectories by
taking the sample mean of δ²(Δ) at each lag.

The efficient implementation below uses the Calandrini algorithm, in
which the cross-correlation term is computed via FFT, yielding an
O(N log N) algorithm per trajectory [see Kehr, Kutner & Binder,
Phys. Rev. B 23, 4931 (1981); or the discussion in Sec. 3.2 of
Metzler et al., Phys. Chem. Chem. Phys. 16, 24128 (2014)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "HurstFit",
    "msd_time_averaged",
    "msd_ensemble",
    "fit_hurst",
    "velocity_autocorrelation",
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
    """

    hurst: float
    slope: float
    intercept: float
    fit_range: tuple[float, float]
    n_points: int


def msd_time_averaged(trajectory: np.ndarray) -> np.ndarray:
    """Time-averaged MSD for a single trajectory.

    Parameters
    ----------
    trajectory : ndarray, shape (N, d)
        Regularly sampled positions in *d* dimensions (usually d = 2).

    Returns
    -------
    ndarray, shape (N,)
        ``msd[k]`` = time-averaged squared displacement at lag ``k`` (in
        sample units). ``msd[0] = 0`` by construction.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be 2-D of shape (N, d), got shape {trajectory.shape}"
        )
    n, d = trajectory.shape

    # --- S1 term: (1/(N-k)) Σ [r²(t) + r²(t+k)] ---------------------------
    # Computed recursively in O(N).
    r2 = np.sum(trajectory**2, axis=1)
    s1 = np.empty(n, dtype=float)
    total = 2.0 * r2.sum()
    s1[0] = total / n
    for k in range(1, n):
        total -= r2[k - 1] + r2[n - k]
        s1[k] = total / (n - k)

    # --- S2 term: (1/(N-k)) Σ r(t) · r(t+k) --------------------------------
    # Computed via FFT: autocorrelation of each coordinate summed.
    s2 = np.zeros(n, dtype=float)
    size_fft = 2 * n  # zero-pad to avoid circular convolution
    for axis in range(d):
        x = trajectory[:, axis]
        f = np.fft.fft(x, n=size_fft)
        auto = np.fft.ifft(f * np.conjugate(f)).real[:n]
        s2 += auto / (n - np.arange(n))

    msd = s1 - 2.0 * s2
    msd[0] = 0.0  # clean up numerical noise at Δ=0
    return msd


def msd_ensemble(ensemble: np.ndarray) -> np.ndarray:
    """Ensemble-averaged, time-averaged MSD.

    Parameters
    ----------
    ensemble : ndarray, shape (M, N, d)
        ``M`` trajectories of length ``N`` in dimension ``d``.

    Returns
    -------
    ndarray, shape (N,)
        Mean of ``msd_time_averaged`` over the ensemble.
    """
    if ensemble.ndim != 3:
        raise ValueError(
            "ensemble must have shape (n_trajectories, n_steps, d), "
            f"got shape {ensemble.shape}"
        )
    per_traj = np.stack([msd_time_averaged(traj) for traj in ensemble], axis=0)
    return per_traj.mean(axis=0)


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

    return HurstFit(
        hurst=slope / 2.0,
        slope=slope,
        intercept=intercept,
        fit_range=(lag_min, lag_max),
        n_points=int(mask.sum()),
    )
