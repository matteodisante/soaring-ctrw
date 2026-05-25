"""Repo-relative output paths.

All scripts write their artefacts under ``outputs/`` at the repository
root, organised as

    outputs/figures/   -- ``.pdf`` and ``.png`` figures
    outputs/data/      -- ``.npz`` numerical caches

Every script accepts an ``--output-dir`` argument that overrides these
defaults; the helpers below are only the conventional locations.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["REPO_ROOT", "OUTPUTS_DIR", "FIGURES_DIR", "DATA_DIR", "CONFIGS_DIR"]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATA_DIR = OUTPUTS_DIR / "data"
CONFIGS_DIR = REPO_ROOT / "configs"
