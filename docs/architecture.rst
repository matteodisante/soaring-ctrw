Architecture
============

The project separates a small, reusable **library** (``src/soaring_ctrw``)
from the **scripts** that orchestrate it. Conceptually the library is layered:

- **Leaf utilities** — :mod:`~soaring_ctrw.paths` (repo-relative output
  locations) and :mod:`~soaring_ctrw.distributions` (waiting-time samplers)
  depend on nothing else in the package.
- **Parameter layer** — :mod:`~soaring_ctrw.model` builds the frozen-dataclass
  configs and assembles the samplers from :mod:`~soaring_ctrw.distributions`.
- **Dynamics** — :mod:`~soaring_ctrw.simulation` turns a config into
  trajectories.
- **Analysis** — :mod:`~soaring_ctrw.observables` computes the MSD and Hurst
  exponent from trajectories.
- **Bridges** — :mod:`~soaring_ctrw.calibration` reads/writes the per-aircraft
  calibration YAML and injects the calibrated ``sigma_theta`` into a config;
  :mod:`~soaring_ctrw.cache` provides manifest-based NPZ caching for the
  Monte-Carlo scripts.

The scripts sit on top and depend downward into the library only.

The graph below is regenerated from the actual ``import`` statements on every
documentation build, so it is always an accurate picture of the current
dependencies (blue = library modules, yellow = scripts).

.. graphviz:: _generated/architecture.dot
   :align: center

Per-script import maps appear on each script's page under :doc:`scripts`.
