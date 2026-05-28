# The model

A flight is a renewal sequence of soaring **cycles**. Each cycle has three
phases — *transition* (T), *search* (S), *climb* (C) — with i.i.d.
phase-duration schedulers:

- **Transition** and **search**: Lomax survival $P(\tau > t) = (1 + t/\tau_0)^{-\mu}$
  — see {class}`~soaring_ctrw.distributions.LomaxTail`.
- **Climb**: exponential, $P(\tau > t) = e^{-t/\mu_C^{\mathrm{eff}}}$
  — see {class}`~soaring_ctrw.distributions.Exponential`.

The transition phase carries the heading from cycle to cycle through a Gaussian
random walk on the circle, $\theta_n = \theta_{n-1} + \eta_n$,
$\eta_n \sim \mathcal{N}(0, \sigma_\theta^2)$, with $\theta_0$ uniform on
$[0, 2\pi)$ per trajectory — see {class}`~soaring_ctrw.model.AngularConfig`.
The dispersion $\sigma_\theta$ is the single phenomenological parameter,
calibrated per aircraft against the universal $H \approx 0.88$.

Intra-phase dynamics:

- **Search** is a local CTRW with physical-duration stopping — ballistic legs
  interleaved with Mittag-Leffler turning waits
  ({class}`~soaring_ctrw.distributions.MittagLeffler`). Parameters live in
  {class}`~soaring_ctrw.model.SearchMotionConfig`.
- **Climb** is circular motion at radius $r_0$ with a slow orographic drift —
  see {class}`~soaring_ctrw.model.ClimbMotionConfig`.

## From parameters to observables

```{list-table}
:header-rows: 1
:widths: 30 70

* - Step
  - API
* - Load + assemble parameters
  - {class}`~soaring_ctrw.model.SoaringConfig` (`from_yaml`, `bare`); each
    {class}`~soaring_ctrw.model.PhaseConfig` builds a sampler via `build()`.
* - Inject the calibrated $\sigma_\theta$
  - {func}`~soaring_ctrw.calibration.load_calibrated_config`
* - Simulate trajectories
  - {func}`~soaring_ctrw.simulation.simulate_single` →
    {class}`~soaring_ctrw.simulation.CycleTrajectory`;
    {func}`~soaring_ctrw.simulation.simulate_ensemble` for a population;
    {func}`~soaring_ctrw.simulation.interpolate_trajectory` to a uniform grid.
* - Measure the MSD and Hurst exponent
  - {func}`~soaring_ctrw.observables.msd_ensemble`,
    {func}`~soaring_ctrw.observables.fit_hurst` →
    {class}`~soaring_ctrw.observables.HurstFit`
```

The full signatures, attributes and per-method docstrings are in the
{doc}`api`; the dependency relationships are drawn in {doc}`architecture`.
