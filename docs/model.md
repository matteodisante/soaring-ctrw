# Model description

This document specifies the cycle-based Continuous-Time Random Walk
(CTRW) model implemented in this repository and derives its main
analytical properties. We follow the notation of Metzler & Klafter
(Phys. Rep. **339**, 1, 2000) and Zaburdaev, Denisov & Klafter
(Rev. Mod. Phys. **87**, 483, 2015). The companion manuscripts
`paper/paper.pdf` (long version, 22 pp.) and `paper/paper_prl.pdf`
(letter version, 7 pp.) contain the full derivation, calibration and
phase-diagram analysis; this note is the technical companion targeted
at the implementation in `src/soaring_ctrw/`.

## 1. Model definition

We treat a cross-country soaring flight as a sequence of complete
transition–search–climb cycles indexed by $n \in \mathbb{N}$. The
horizontal position at the end of cycle $n$ decomposes as
$\mathbf{x}_n = \mathbf{x}_{n-1} + \Delta\mathbf{x}^{\mathrm{T}}_n
 + \Delta\mathbf{x}^{\mathrm{S}}_n + \Delta\mathbf{x}^{\mathrm{C}}_n$,
with one contribution per phase (transition, search, climb). The cycle
duration is
$T_n = \tau^{\mathrm{T}}_n + \tau^{\mathrm{S}}_n + \tau^{\mathrm{C}}_n$,
the elapsed physical time after $n$ cycles is
$t_n = \sum_{k=1}^{n} T_k$, and the cycle counter is
$N(t) = \max\{n : t_n \le t\}$.

**Assumptions (explicit).**

1. Phase durations are independent across phases and i.i.d. across
   cycles in the duration triple
   $(\tau^{\mathrm{T}}_n,\tau^{\mathrm{S}}_n,\tau^{\mathrm{C}}_n)$.
2. Transition duration: Pareto tail,
   $P(\tau^{\mathrm{T}} > \tau) = (\tau_{\min}^{\mathrm{T}}/\tau)^{\mu_{\mathrm{T}}}$
   for $\tau \ge \tau_{\min}^{\mathrm{T}}$, with $\mu_{\mathrm{T}}$ the
   tail exponent reported in Fig. 4c of Vilpellet et al. (2026).
3. Search duration: Pareto tail with exponent $\mu_{\mathrm{S}}$
   (Fig. 4f).
4. Climb duration: exponential with mean $\bar{\tau}_{\mathrm{C}}$
   (Fig. 4i is consistent with a thin tail).
5. Heading dynamics — the key modelling choice:
   $\theta_n = \theta_{n-1} + \eta_n$,
   $\eta_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\theta^2)$,
   with $\theta_0$ deterministic. This is the only phenomenological
   parameter introduced beyond what the data fix.

The transition step is the persistent ballistic kernel
$\Delta\mathbf{x}^{\mathrm{T}}_n = v_{xy}\,\tau^{\mathrm{T}}_n
 \,\hat{\mathbf{e}}(\theta_n)$ with
$\hat{\mathbf{e}}(\theta) \equiv (\cos\theta,\sin\theta)$. The search
and climb intra-phase contributions are detailed in §4 below.

Assumption 5 sets the persistence length in cycles,

$$
N_p = \frac{2}{\sigma_\theta^2},
$$

i.e. the number of cycles after which the heading correlator
$\langle \cos(\theta_{n+k}-\theta_n)\rangle = \exp(-k\sigma_\theta^2/2)$
drops to $1/\mathrm{e}$.

## 2. Closed-form transition MSD after $N$ cycles

Setting aside the search and climb increments (which contribute a
short-lag plateau but no large-scale persistence), and using
$\langle \hat{\mathbf{e}}(\theta_j)\cdot\hat{\mathbf{e}}(\theta_i)\rangle
 = \rho^{|j-i|}$ with $\rho \equiv \mathrm{e}^{-\sigma_\theta^2/2}$,

$$
\langle |\mathbf{X}_N^{\mathrm{T}}|^2\rangle
 = N\,\langle\ell^2\rangle
 + 2\,\langle\ell\rangle^2 \sum_{k=1}^{N-1} (N-k)\,\rho^k,
\qquad \ell_n = v_{xy}\,\tau^{\mathrm{T}}_n.
$$

Using $\sum_{k=1}^{N-1}(N-k)\rho^k = N\rho/(1-\rho) -
\rho(1-\rho^N)/(1-\rho)^2$,

$$
\boxed{\;\langle |\mathbf{X}_N^{\mathrm{T}}|^2\rangle
 = N\,\langle\ell^2\rangle
 + \langle\ell\rangle^2 \left[
     \frac{2N\rho}{1-\rho} - \frac{2\rho\,(1-\rho^N)}{(1-\rho)^2}
   \right].\;}
$$

Two limiting regimes follow directly:

- **Ballistic, $N \ll N_p$:** Taylor-expanding $\rho = 1 - \sigma_\theta^2/2 + O(\sigma_\theta^4)$ and $\rho^N = 1 - N\sigma_\theta^2/2 + O((N\sigma_\theta^2)^2)$,
  $\langle |\mathbf{X}_N^{\mathrm{T}}|^2\rangle \simeq \langle\ell\rangle^2 N^2
   \bigl(1 - \tfrac{1}{3}N\sigma_\theta^2 + \dots\bigr) \propto N^2$,
  i.e. ballistic motion of a persistent walker.
- **Diffusive, $N \gg N_p$:** $\rho^N \to 0$ and
  $\langle |\mathbf{X}_N^{\mathrm{T}}|^2\rangle \simeq N\left[\langle\ell^2\rangle
   + \tfrac{2\rho}{1-\rho}\langle\ell\rangle^2\right] \propto N$,
  with an effective diffusivity rescaled by the persistence factor
  $(1+\rho)/(1-\rho)$.

A *single power-law fit* across the crossover window $N \sim N_p$
returns an effective exponent that interpolates smoothly between $1/2$
and $1$. This is the mechanism by which the observed
$H \approx 0.88$ of Vilpellet et al. (2026) can arise as a
pre-asymptotic crossover regime rather than as a Hurst exponent
of an asymptotic anomalous law.

### Heavy-tailed step lengths

For $\mu_{\mathrm{T}} \le 3$, $\langle\ell^2\rangle$ diverges and the
above derivation breaks down. The asymptotic law is then the Lévy-walk
scaling $\langle|\mathbf{X}_N^{\mathrm{T}}|^2\rangle \sim N^{3-\mu_{\mathrm{T}}}$,
valid only for $1 < \mu_{\mathrm{T}} < 2$; for $\mu_{\mathrm{T}} > 2$
(the regime measured for paragliders, hang gliders and sailplanes,
$\mu_{\mathrm{T}} = 3.93,\,4.79,\,2.62$), the second moment is finite
and the cycle-counter renewal recovers normal diffusion at large $N$.
The Monte Carlo of `simulate_ensemble` captures both cases without
analytical distinction, since the empirical MSD is computed directly
from the simulated trajectories.

## 3. Mapping cycle count to physical time

For $\mu_{\mathrm{T}} > 2$ the cycle mean
$\langle T\rangle = \langle \tau^{\mathrm{T}}\rangle +
 \langle \tau^{\mathrm{S}}\rangle + \langle \tau^{\mathrm{C}}\rangle$
is finite, the renewal theorem gives
$N(t) \sim t/\langle T\rangle$ almost surely, and the MSD inherits
the cycle-level scaling via $N \leftrightarrow t/\langle T\rangle$.
The transition-time fraction

$$
\eta_{\mathrm{T}} = \frac{\langle \tau^{\mathrm{T}}\rangle}
 {\langle \tau^{\mathrm{T}}\rangle + \langle \tau^{\mathrm{S}}\rangle
  + \langle \tau^{\mathrm{C}}\rangle}
$$

controls the long-time *effective* speed: in the fully persistent
limit $\sigma_\theta \to 0$ the long-time MSD is
$\delta^2(\Delta) \sim (v_{xy}\eta_{\mathrm{T}})^2\,\Delta^2$.

For $\mu_{\mathrm{T}} \le 1$ (not realised for any aircraft in the
paper) the transition mean diverges, the cycle-count process is
anomalous, and one must resort to stable-law subordination; this
regime is not pursued here.

## 4. Intra-phase contributions

The minimal model with horizontally-stationary search and climb
captures the long-time crossover but misses two pre-asymptotic
features visible in Fig. 3 of Vilpellet et al. (2026): the
sub-diffusive scaling $\delta^2_S \propto \Delta^{\alpha_S}$ in the
search phase ($\alpha_S \approx 0.6$ for paragliders/hang gliders,
$0.4$ for sailplanes), and the ballistic-then-saturating-then-drift
structure in the climb phase. The current implementation reproduces
both via dedicated intra-phase generators implemented in
`src/soaring_ctrw/simulation.py`.

### 4.1 Search: subordinated Lévy walk

Inside each search phase of total duration $\tau_n^S$, the walker
alternates ballistic legs and stationary waits (Fogedby 1994;
Magdziarz–Weron 2007):

- Leg duration: $\sigma_k \sim \mathrm{Exp}(\sigma_0)$, walker moves
  at constant speed $v_c^S$ in direction $\psi_k$ (Gaussian random
  walk on the circle with per-leg increment variance
  $\sigma_\psi^2$).
- Waiting time: $\xi_k \sim \mathrm{ML}(\alpha_S,\,\tau_w^S)$ with
  $\alpha_S \in (0,1)$, walker stationary in the $xy$-projection
  (this represents tight in-place circling at a radius below the
  GPS resolution).

The two scales $\sigma_0$ and $\tau_w^S$ separate three regimes in
the conditional MSD:

- $\Delta \ll \sigma_0$: ballistic, $\delta^2_S \simeq (v_c^S)^2 \Delta^2$.
- $\sigma_0 \ll \Delta \ll \tau_w^S$: transient (slightly sub-ballistic).
- $\Delta \gg \tau_w^S$: sub-diffusive, $\delta^2_S \propto \Delta^{\alpha_S}$
  by the Metzler–Klafter renewal theorem applied to the subordinator.

The class is implemented as `SearchMotionConfig` with attributes
`(v_c_S, sigma_0, tau_0_S, alpha_S, sigma_psi_S)`. (The
implementation field `tau_0_S` corresponds to the $\tau_w^S$ symbol
of the PRL paper: it is the Mittag-Leffler scale of the
**intra-phase** waiting times, *not* to be confused with the Pareto
scale $\tau_{\min}^S$ of the cycle-duration scheduler.)

### 4.2 Climb: harmonic oscillator with Gaussian-damped period and drift

During each climb of duration $\tau_n^C$, the pilot circles a thermal
core at radius $r_0$ with period $T_{\mathrm{turn}}$, plus a slow
linear orographic drift:

$$
\mathbf{x}^C_n(c) = r_0\bigl[\cos(\omega_n c+\phi_0) - \cos\phi_0,\;
                                \sin(\omega_n c+\phi_0) - \sin\phi_0\bigr]
                 + v_{\mathrm{drift}}\,c\,\hat{\mathbf{e}}(\phi_n^{\mathrm{drift}}),
$$

with $\omega_n = 2\pi/T_{\mathrm{turn},n}$. The period
$T_{\mathrm{turn},n}$ is drawn per cycle from a Gaussian with mean
$\bar{T}_{\mathrm{turn}}$ and standard deviation $\sigma_T$,
positivity-clipped at $0.2\,\bar{T}_{\mathrm{turn}}$. The drift
direction $\phi_n^{\mathrm{drift}}$ is uniform per cycle and
independent of the heading. After ensemble-averaging over $\phi_0$
(uniform on the circle) and $T_{\mathrm{turn},n}$ (Gaussian), the
conditional MSD reads

$$
\delta^2_C(\Delta) =
2 r_0^2\!\left[1
 - \mathrm{e}^{-\sigma_T^2 \bar{\omega}_0^2 \Delta^2/2}
   \cos(\bar{\omega}_0 \Delta)\right]
+ v_{\mathrm{drift}}^2\,\Delta^2,
\qquad \bar{\omega}_0 = 2\pi/\bar{T}_{\mathrm{turn}}.
$$

The Gaussian damping factor $\exp(-\sigma_T^2\bar{\omega}_0^2\Delta^2/2)$
is the dispersion in the per-cycle turn frequency, and is what
prevents periodic anti-correlation dips from surviving ensemble
averaging when $\sigma_T\bar{\omega}_0 \gtrsim 1$ (paragliders, hang
gliders). Without dispersion ($\sigma_T \to 0$) one recovers the
clean oscillatory form $2 r_0^2[1 - \cos(\bar{\omega}_0\Delta)] +
v_{\mathrm{drift}}^2\Delta^2$ used in the sailplane fit, which keeps
the marked anti-correlation dip at $\Delta \approx \pi/\bar{\omega}_0$.

The class is implemented as `ClimbMotionConfig` with attributes
`(thermal_radius, turn_period, turn_period_std, v_drift)`.

## 5. Connection to Green–Kubo

The MSD in two dimensions admits the Green–Kubo representation

$$
\delta^2(\Delta) = 2 \int_0^{\Delta} (\Delta - t)\,C_{\mathbf{v}}(t)\,\mathrm{d}t,
$$

where $C_{\mathbf{v}}(t) = \langle \mathbf{v}(t')\cdot \mathbf{v}(t'+t)\rangle$
is the velocity autocorrelation function. In our model, during the
transition phase $\mathbf{v}(t) = v_{xy}\hat{\mathbf{e}}(\theta_n)$,
during the search phase $\mathbf{v}(t)$ is the leg velocity (with
piecewise pauses), and during the climb phase $\mathbf{v}(t)$ is the
sum of the rotational tangent velocity $r_0\omega_0$ and the drift
$v_{\mathrm{drift}}$. The transition contribution to $C_{\mathbf{v}}$
factorises as the product of (i) the probability that both times
$t'$ and $t' + t$ fall within transition phases (intermittency
factor) and (ii) the heading correlator
$\langle \cos\theta(t'+t)\cos\theta(t')\rangle$, which decays on the
timescale $t_p = N_p \langle T\rangle$. Both factors are short- to
intermediate-ranged, and their convolution produces the
pre-asymptotic power-law window observed in the data. Measuring
$C_{\mathbf{v}}(t)$ directly on empirical trajectories — as proposed
by Vilpellet et al. (2026) in their first future direction — is the
natural discriminating test for the present model.

## 6. Calibration strategy

Given $\mu_{\mathrm{T}}, \mu_{\mathrm{S}}, \bar{\tau}_{\mathrm{C}},
v_{xy}$ measured per aircraft, we fix $\tau_{\min}^{\mathrm{T}}$ and
$\tau_{\min}^{\mathrm{S}}$ so that the **mean cycle duration**
$\langle T\rangle$ matches the paragliders value (≈ 415 s; this is
the alignment condition that produces the universal MSD collapse of
Vilpellet et al. Fig. 1 inset). The single free parameter
$\sigma_\theta$ is fixed once and for all to $0.35$ rad, on the
$(\mu_{\mathrm{T}}, \sigma_\theta)$ iso-line $H_{\mathrm{eff}} = 0.88$
of the phase diagram — see `scripts/scan_phase_diagram.py`. The
intra-phase scales $(\sigma_0, \tau_w^S, \alpha_S, v_c^S, r_0,
T_{\mathrm{turn}}, \sigma_T, v_{\mathrm{drift}})$ are then chosen
consistently with the per-phase MSDs of Vilpellet Fig. 3.

## 7. Limitations and extensions

1. **Coloured heading noise.** Replacing the i.i.d. Gaussian
   increments $\eta_n$ by a fractional process would introduce a
   second persistence timescale and allow direct comparison with
   the temporal fractional dynamics framework
   (time-fractional Fokker–Planck) of the seminar from which this
   project originates. Note that an Ornstein–Uhlenbeck process on
   the unbounded heading $\theta$ is unphysical (it pulls
   $\theta\to 0$ and breaks rotational invariance); a workable
   colour would have to act on the increments $\eta_n$ instead.
2. **Wind advection.** Adding a global drift
   $\mathbf{v}_{\mathrm{wind}}$ with optional spatial heterogeneity
   would separate air-relative from ground-relative motion, which
   is the paper's third future direction.
3. **Adaptive $\sigma_\theta$ by phase or skill.** Letting
   $\sigma_\theta$ depend on search outcomes would reproduce the
   "learning" signature of higher-skill pilots without adding free
   parameters.

All three extensions are compatible with the present code layout:
each amounts to a new dataclass replacement of the relevant
`*Config` block.
