"""SM diagnostic: convergence of the per-cycle variance estimator.

For the near-critical classes (``mu_T < 4``) the fourth moment of the
Lomax transition duration diverges, so the sample estimator of
``Var(tau_T)`` --- and hence of the background amplitude
``B = v_xy^2 Var(tau_T) + Sigma_S + Sigma_C`` of Eq. (eq:msd-closed) ---
has infinite variance and converges slowly with the ensemble size ``M``.
This is the sampling effect discussed in the manuscript (Appendix D and
the Conclusions): it affects sailplanes (``mu_T = 2.62``) most strongly,
but also paragliders (``mu_T = 3.93``), while hang gliders
(``mu_T = 4.79 > 4``) have a finite fourth moment and converge normally.

The figure shows the running estimate of ``v_xy^2 Var(tau_T)``, the
dominant term of ``B`` (>97% of it for all three classes), normalised by
its analytic value, versus ``M``, over several independent seeds. The
spread of the curves at fixed ``M`` is the finite-sample uncertainty on
``B``; it shrinks slowly (heavy tail) for ``mu_T < 4`` and quickly for
``mu_T > 4``.

Output (under ``outputs/``):
    figures/variance_convergence.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from soaring_ctrw.distributions import LomaxTail
from soaring_ctrw.model import SoaringConfig
from soaring_ctrw.paths import CONFIGS_DIR, FIGURES_DIR

AIRCRAFT_ORDER = ("paragliders", "hang_gliders", "sailplanes")
AIRCRAFT_LABELS = {
    "paragliders": "paragliders",
    "hang_gliders": "hang gliders",
    "sailplanes": "sailplanes",
}
PANEL_LETTERS = ("a", "b", "c")


def _running_variance(samples: np.ndarray, Ms: np.ndarray) -> np.ndarray:
    """Cumulative (prefix) sample variance of ``samples`` evaluated at the
    sample sizes ``Ms`` (uses the population formula S2/M - (S1/M)^2)."""
    c1 = np.cumsum(samples)
    c2 = np.cumsum(samples ** 2)
    idx = Ms - 1
    mean = c1[idx] / Ms
    return c2[idx] / Ms - mean ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft", nargs="+", default=list(AIRCRAFT_ORDER),
        choices=list(AIRCRAFT_ORDER),
    )
    parser.add_argument("--n-max", type=int, default=200_000,
                        help="Largest ensemble size M shown.")
    parser.add_argument("--n-seeds", type=int, default=8,
                        help="Independent seeds (curves) per aircraft.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    Ms = np.unique(np.geomspace(50, args.n_max, 80).astype(int))
    Ms = Ms[Ms >= 2]

    n = len(args.aircraft)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.2),
                             constrained_layout=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ac, letter in zip(axes, args.aircraft, PANEL_LETTERS):
        cfg = SoaringConfig.from_yaml(CONFIGS_DIR / f"{ac}.yaml")
        mu_T = float(cfg.transition.params["mu"])
        tau0_T = float(cfg.transition.params["tau_0"])
        v_xy = float(cfg.v_xy)
        dist = LomaxTail(mu=mu_T, tau_0=tau0_T)
        analytic = v_xy ** 2 * dist.variance  # finite for mu_T > 2

        for s in range(args.n_seeds):
            rng = np.random.default_rng(args.seed + s)
            tau = dist.sample(args.n_max, rng)
            est = v_xy ** 2 * _running_variance(tau, Ms)
            ax.plot(Ms, est / analytic, lw=1.0, alpha=0.6)

        ax.axhline(1.0, color="k", ls="--", lw=1.2,
                   label=r"analytic $v_{xy}^2\,\mathrm{Var}(\tau^T)$")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 2.2)
        ax.set_xlabel(r"ensemble size $M$")
        if ax is axes[0]:
            ax.set_ylabel(r"$\hat{B}_{\mathrm{T}}(M)\,/\,$analytic")
        tail = "finite 4th moment" if mu_T > 4 else "divergent 4th moment"
        ax.set_title(
            rf"({letter}) {AIRCRAFT_LABELS[ac]}: $\mu_T={mu_T:g}$ ({tail})",
            fontsize=10, loc="left",
        )
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        r"Convergence of the per-cycle variance estimator "
        r"$v_{xy}^2\,\mathrm{Var}(\tau^T)$ vs ensemble size",
        fontsize=11,
    )
    out = args.figures_dir / "variance_convergence.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
