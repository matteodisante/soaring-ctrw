# soaring-ctrw

A cycle-based Continuous-Time Random Walk (CTRW) model with angular
persistence for cross-country soaring flights.

This repository accompanies a manuscript investigating how the
universal sub-ballistic transport scaling (Hurst exponent
*H* ≈ 0.88) reported by Vilpellet, Darmon & Benzaquen
[[arXiv:2601.01293](https://arxiv.org/abs/2601.01293)] can be reproduced
by a minimal 2-D CTRW in which elementary steps correspond to complete
transition→search→climb cycles and successive heading directions are
correlated through a Gaussian random walk on the circle. Two manuscript
versions are bundled:

- `paper/paper.pdf` — long version (22 pages) with full derivation,
  Monte Carlo benchmarks and phase diagram.
- `paper/paper_prl.pdf` — letter version (7 pages) targeting PRL
  formatting, with the same scientific content compressed.

## Scientific motivation

The empirical paper of Vilpellet et al. (2026) reports a robust
*H* ≈ 0.88 across paragliders, hang gliders and sailplanes despite
large differences in characteristic speed, glide ratio and heavy-tail
exponents of the phase durations. The paper adopts throughout the
*survival convention* $P(\tau>t)\sim t^{-\mu}$ used by Vilpellet's
Hill estimator. In this convention the coupled Lévy-walk prediction
for the MSD is $\delta^2\sim\Delta^{3-\mu_T}$, valid only for
$1<\mu_T<2$; the density-convention form $\Delta^{4-\mu_T}$ valid
for $2<\mu_T<3$ that appears in Vilpellet's Discussion is the same
physical scaling relabelled via $\mu_{\rm density}=\mu_{\rm survival}+1$,
not a separate prediction. Vilpellet's Hill-estimator values
μ_T ≈ 3.93, 4.79, 2.62 all sit *above* 2 in survival convention:
transition durations have finite second moment, the Lévy-walk
asymptotic regime never sets in, and the naive prediction collapses
to plain diffusion (*H* = 1/2), incompatible with the observed
*H* ≈ 0.88. The discrepancy points to two missing ingredients:

1. **Two-dimensional directional decorrelation** between successive
   ballistic segments.
2. **Pre-asymptotic crossover** between ballistic ($N\!\ll\!N_p$) and
   diffusive ($N\!\gg\!N_p$) regimes over the finite observation
   window 10¹–10³ s.

This repository implements the minimal model that makes both effects
explicit and quantifies their joint role.

## Model summary

Each soaring cycle *n* consists of a transition, a search, and a climb
phase with durations $(\tau^T_n,\tau^S_n,\tau^C_n)$ drawn from the
prescribed heavy-tailed (Pareto/Mittag-Leffler) and exponential
distributions. The transition contributes a persistent ballistic step

$$\mathbf{x}_n^T = \mathbf{x}_{n-1}^T + v_{xy}\, \tau^{\mathrm{T}}_n \hat{\mathbf{e}}(\theta_n),
\qquad \theta_n = \theta_{n-1} + \eta_n,\quad \eta_n \sim \mathcal{N}(0, \sigma_\theta^2).$$

Search and climb phases each contribute a non-trivial intra-phase
displacement, calibrated against Fig. 3 of Vilpellet et al.:

- **Search** — subordinated Lévy walk: ballistic legs at speed $v_c^S$
  with exponential durations $\sigma_0$, alternating with stationary
  Mittag-Leffler waits of exponent $\alpha_S\!\in\!(0,1)$ and scale
  $\tau_w^S$. Reproduces ballistic → sub-diffusive
  $\Delta^{\alpha_S}$ crossover.
- **Climb** — 2-D harmonic oscillator (pilot circling a thermal at
  radius $r_0$ with mean turn period $\bar T_{\rm turn}$, dispersion
  $\sigma_T$) plus a linear orographic drift of magnitude
  $v_{\rm drift}$. Gaussian damping of the period dispersion smears
  the anti-correlation dip when $\sigma_T\bar\omega_0 \gtrsim 1$.

Per-cycle leg directions are i.i.d. uniform on the circle, independent
of the transition heading. The angular diffusivity $\sigma_\theta^2$
is the single phenomenological parameter introduced beyond what the
data fix; it sets the persistence length
$N_p = 2/\sigma_\theta^2$ in cycles.

Analytical predictions are worked out in `docs/model.md` and the
companion manuscripts.

## Repository layout

```
soaring-ctrw/
├── src/soaring_ctrw/
│   ├── distributions.py   # Pareto, Exponential, Mittag-Leffler samplers
│   ├── model.py           # SoaringConfig, SearchMotionConfig, ClimbMotionConfig
│   ├── simulation.py      # simulate_single, simulate_ensemble, intra-phase generators
│   └── observables.py     # time-averaged MSD, Hurst-exponent fit
├── configs/
│   ├── paragliders.yaml
│   ├── hang_gliders.yaml
│   └── sailplanes.yaml
├── scripts/
│   ├── run_simulation.py         # single-aircraft MSD
│   ├── scan_phase_diagram.py     # (μ_T, σ_θ) phase diagram of H_eff
│   ├── plot_all_aircraft.py      # full set of paper figures
│   ├── plot_final_calibrated.py  # final MSD + local slope
│   ├── compare_to_empirical.py   # diagnostic local-slope comparison
│   └── diagnose_local_slope.py   # diagnostic tool
├── tests/
│   └── test_*.py
├── paper/
│   ├── paper.tex,     paper.pdf       # long version
│   ├── paper_prl.tex, paper_prl.pdf   # PRL-format letter
│   ├── references.bib
│   └── figures/*.pdf
└── docs/
    └── model.md
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# run a single-aircraft simulation and plot MSD
python scripts/run_simulation.py --config configs/paragliders.yaml

# produce all paper figures
python scripts/plot_all_aircraft.py

# run unit tests
pytest
```

## Status

The angular diffusivity σ_θ = 0.35 rad is *not* fitted aircraft by
aircraft: it is set analytically by the phase diagram, on which the
$H_{\rm eff}=0.88$ iso-contour of the bare-cycle MSD becomes nearly
horizontal in $(\mu_T,\sigma_\theta)$ and crosses σ_θ ≃ 0.35
essentially independently of μ_T. With σ_θ frozen at this universal
value and the hang-glider / sailplane Pareto scales tuned only to
match the paraglider mean cycle duration
$\langle T\rangle \approx 415$ s (the condition that collapses
$\delta^2/v_{xy}^2$ onto a single master curve), the full Monte
Carlo produces a tight cluster of fitted Hurst exponents on the
observation window 10 s < Δ < 5×10³ s:

| aircraft      | μ_T (transition) | $\langle T\rangle$ (s) | H fitted |
|---------------|:----------------:|:----------------------:|:--------:|
| paragliders   |       3.93       |         415.2          |  0.934   |
| hang gliders  |       4.79       |         414.9          |  0.930   |
| sailplanes    |       2.62       |         415.3          |  0.937   |

The aircraft-to-aircraft spread is $|\Delta H|\lesssim 0.01$ across
a factor ≈ 3.6 in horizontal speed and a factor ≈ 1.8 in the
transition tail exponent — this is the universality the paper aims
to explain. The cluster sits ≈ 0.05 above the empirical *H* ≈ 0.88;
the discussion section of the paper documents this slight
over-prediction and identifies it as the limitation set by treating
the cycle-step model with $\sigma_\theta$ pinned analytically rather
than fitted.

The model reproduces both Fig. 1 (total MSD) and Fig. 3 (conditional
per-phase MSD) of Vilpellet et al. (2026):

- **Transition**: persistent ballistic at all lags, by construction.
- **Search**: subordinated Lévy walk (Fogedby 1994; Magdziarz–Weron
  2007) — alternating ballistic tactical-repositioning legs (mean
  duration σ_0, speed v_c^S) with Mittag-Leffler tight-circling
  waits (scale τ_w^S, exponent α_S = 0.6 for paragliders/hang
  gliders, 0.4 for sailplanes). The pilot is never at rest — the
  "waiting times" represent localised circling at a radius well
  below the GPS resolution, which appears as a pause in the
  xy-projection. Reproduces ballistic → sub-diffusive Δ^α_S
  crossover asymptotically.
- **Climb**: 2-D harmonic oscillator (pilot circling a thermal at
  radius r_0 ≈ 50/58/130 m with mean turn period
  $\bar T_{\rm turn}$ and dispersion σ_T) plus a linear orographic
  drift v_drift. Reproduces the three regimes of Fig. 3 of
  Vilpellet et al.: ballistic at short Δ, anti-correlation /
  saturation at intermediate Δ, quasi-ballistic drift tail.

The universal collapse $\delta^2(\Delta)/v_{xy}^2 = $ universal of
the rescaled MSDs (inset of Fig. 1 of Vilpellet) is verified in the
paper figures.

The companion manuscripts are in `paper/paper.pdf` (long) and
`paper/paper_prl.pdf` (letter).

## References

- J. Vilpellet, A. Darmon, M. Benzaquen, *From Random Walks to Thermal
  Rides: Universal Anomalous Transport in Soaring Flights*,
  [arXiv:2601.01293](https://arxiv.org/abs/2601.01293) (2026).
- R. Metzler, J. Klafter, *The Random Walk's Guide to Anomalous
  Diffusion: A Fractional Dynamics Approach*, Phys. Rep. 339, 1 (2000).
- V. Zaburdaev, S. Denisov, J. Klafter, *Lévy Walks*, Rev. Mod. Phys.
  87, 483 (2015).
- O. Bénichou, C. Loverdo, M. Moreau, R. Voituriez, *Intermittent
  Search Strategies*, Rev. Mod. Phys. 83, 81 (2011).
- H. C. Fogedby, *Lévy flights in random environments*,
  Phys. Rev. Lett. 73, 2517 (1994).
- M. Magdziarz, A. Weron, *Competition between subdiffusion and
  Lévy flights: A Monte Carlo approach*, Phys. Rev. E 75, 056702
  (2007).
