Scripts
=======

Each command-line script under ``scripts/`` orchestrates the library to
produce one part of the manuscript. The pages below are generated on every
build: the narrative ("what the script does and why") is the script's own
**module docstring**, the **import map** is derived from its actual imports,
and the **command-line interface** is captured from the live ``--help``
output. A new script dropped into ``scripts/`` gets a page automatically.

For the order in which to run them, see :doc:`pipeline`.

.. toctree::
   :glob:
   :maxdepth: 1

   _generated/scripts/*
