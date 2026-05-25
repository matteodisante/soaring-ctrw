"""Lightweight dataset cache with manifest-based invalidation.

Every script that produces Monte Carlo data goes through three steps:

1.  Build a ``manifest`` dict that describes the *requested* run:
    script name, CLI args that affect the numerical outcome, the
    SHA-256 hash of the resolved YAML config(s), the random seed.

2.  Compare it against the manifest of the existing dataset (if any)
    via ``decide_action``. The policy is:

      * No cache on disk         -> ``"regenerate"``.
      * Cache matches request    -> ``"reuse"`` (or prompt the user in
                                    interactive mode).
      * Cache differs            -> ``"regenerate"`` (never prompt:
                                    different numerics must be re-run).

3.  Either ``load_dataset`` the cached arrays, or compute them and call
    ``save_dataset`` (which writes the ``.npz`` and the ``.json``
    manifest atomically next to each other).

The on-disk layout is one canonical slot per ``(script, aircraft)``::

    outputs/data/<script_slug>/<aircraft>.npz
    outputs/data/<script_slug>/<aircraft>.json

Re-running overwrites the slot in place; the manifest is what tells
``decide_action`` whether the existing file is still valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

__all__ = [
    "CACHE_MODES",
    "CacheDecision",
    "build_manifest",
    "config_hash",
    "decide_action",
    "derived_seed",
    "load_dataset",
    "save_dataset",
    "add_cache_args",
    "slot_paths",
]


def derived_seed(base_seed: int, *tag_parts: str) -> int:
    """Deterministic 64-bit seed from ``(base_seed, *tag_parts)``.

    Used to give each (aircraft, ...) slot an independent RNG stream
    whose state is independent of execution order, so that the cached
    output of one slot can be regenerated identically without first
    re-running the others.
    """
    h = hashlib.sha256(f"{base_seed}|{'|'.join(tag_parts)}".encode()).digest()
    return int.from_bytes(h[:8], "big")

CacheMode = Literal["auto", "reuse", "rerun", "require", "off"]
CACHE_MODES: tuple[str, ...] = ("auto", "reuse", "rerun", "require", "off")


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def config_hash(config_path: Path) -> str:
    """SHA-256 of the raw YAML bytes.

    The hash is taken on the file content (not on the parsed object) so
    that any user edit to the YAML — including comments — triggers a
    re-run. That is the conservative choice and matches the user's
    expectation that "changing numerical settings forces a rerun".
    """
    return _sha256_bytes(Path(config_path).read_bytes())


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    script: str,
    params: dict[str, Any],
    config_paths: dict[str, Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest dict for the *requested* run.

    ``params`` must contain only values that affect the numerical
    outcome (n_trajectories, dt, ranges, seed, ...). Output paths,
    figure styling, log directory, etc. must NOT be in it.

    ``config_paths`` maps a logical name (e.g. ``"aircraft"``) to a
    YAML path; its SHA-256 is added to ``configs`` in the manifest.
    """
    configs = {}
    if config_paths:
        for name, path in config_paths.items():
            configs[name] = {
                "path": str(path),
                "sha256": config_hash(path),
            }

    manifest = {
        "script": script,
        "params": _normalize(params),
        "configs": configs,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
    }
    if extra:
        manifest["extra"] = _normalize(extra)
    return manifest


def _normalize(obj: Any) -> Any:
    """Convert obj into a JSON-serialisable, comparison-stable form.

    Tuples become lists, numpy scalars become Python scalars, numpy
    arrays become lists. Dict keys are stringified. This is what gets
    compared by ``_params_equal`` so the conversion must be applied to
    both sides identically.
    """
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Disk layout
# ---------------------------------------------------------------------------


def slot_paths(data_dir: Path, script_slug: str, slot: str) -> tuple[Path, Path]:
    """Return the ``(npz_path, manifest_path)`` for one cache slot."""
    sub = data_dir / script_slug
    return sub / f"{slot}.npz", sub / f"{slot}.json"


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


@dataclass
class CacheDecision:
    action: Literal["reuse", "regenerate"]
    reason: str
    diff: list[str]  # human-readable param differences (empty on full match)


def _params_equal(a: dict, b: dict) -> list[str]:
    """Return a list of human-readable differences between manifests.

    Compares ``params`` and ``configs`` (the parts that determine the
    numerical outcome). Ignores ``created_at`` and ``git_commit``.
    """
    diffs: list[str] = []
    pa, pb = a.get("params", {}), b.get("params", {})
    keys = sorted(set(pa) | set(pb))
    for k in keys:
        if pa.get(k) != pb.get(k):
            diffs.append(f"params.{k}: cached={pa.get(k)!r}  requested={pb.get(k)!r}")
    ca, cb = a.get("configs", {}), b.get("configs", {})
    keys = sorted(set(ca) | set(cb))
    for k in keys:
        sa = (ca.get(k) or {}).get("sha256")
        sb = (cb.get(k) or {}).get("sha256")
        if sa != sb:
            diffs.append(f"configs.{k}.sha256: cached={sa}  requested={sb}")
    return diffs


def _confirm_reuse(slot_label: str, cached_when: str | None) -> bool:
    """Interactive Y/n prompt. Returns True for reuse, False for rerun.

    On non-TTY (CI, pipes) defaults to reuse — matches ``--cache reuse``.
    """
    when = f" ({cached_when})" if cached_when else ""
    if not sys.stdin.isatty():
        print(f"  cache hit{when}: reusing (non-interactive).")
        return True
    try:
        ans = input(f"  cache hit{when}. Reuse cached data? [Y/n] ").strip().lower()
    except EOFError:
        return True
    return ans in ("", "y", "yes")


def decide_action(
    *,
    npz_path: Path,
    manifest_path: Path,
    requested_manifest: dict[str, Any],
    mode: CacheMode,
    slot_label: str = "",
) -> CacheDecision:
    """Decide whether to ``reuse`` or ``regenerate`` the dataset.

    Policy:

    * ``rerun``   -> always regenerate.
    * ``off``     -> always regenerate; caller MUST skip ``save_dataset``.
    * ``require`` -> reuse if present and matching; otherwise raise.
    * ``reuse``   -> reuse if present and matching; otherwise regenerate.
    * ``auto``    -> like ``reuse`` but, on a clean match, prompt the
                    user interactively. On mismatch, regenerate silently
                    (the user said: different numerics force a re-run).
    """
    if mode == "rerun" or mode == "off":
        return CacheDecision("regenerate", f"cache mode = {mode}", [])

    if not (npz_path.exists() and manifest_path.exists()):
        if mode == "require":
            raise FileNotFoundError(
                f"--cache require: no cached dataset at {npz_path}"
            )
        return CacheDecision("regenerate", "no cache on disk", [])

    try:
        cached = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        if mode == "require":
            raise RuntimeError(f"cache manifest unreadable: {exc}") from exc
        return CacheDecision("regenerate", f"manifest unreadable: {exc}", [])

    diffs = _params_equal(cached, requested_manifest)
    if diffs:
        if mode == "require":
            raise RuntimeError(
                "--cache require: cached parameters differ:\n  "
                + "\n  ".join(diffs)
            )
        return CacheDecision("regenerate", "params differ", diffs)

    # Match.
    if mode == "auto":
        if _confirm_reuse(slot_label, cached.get("created_at")):
            return CacheDecision("reuse", "user accepted cache", [])
        return CacheDecision("regenerate", "user requested re-run", [])
    return CacheDecision("reuse", f"cache matches (mode={mode})", [])


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_dataset(
    *,
    npz_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    """Atomically write ``arrays`` to NPZ and ``manifest`` to JSON."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    tmp_json = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    # Open as file handle so np.savez doesn't auto-append ".npz".
    with open(tmp_npz, "wb") as f:
        np.savez(f, **arrays)
    tmp_json.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    os.replace(tmp_npz, npz_path)
    os.replace(tmp_json, manifest_path)


def load_dataset(
    npz_path: Path, manifest_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load arrays and manifest from a cache slot."""
    manifest = json.loads(manifest_path.read_text())
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return arrays, manifest


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def add_cache_args(parser, *, default: CacheMode = "auto") -> None:
    """Attach the standard ``--cache`` argument to an argparse parser."""
    parser.add_argument(
        "--cache",
        choices=CACHE_MODES,
        default=default,
        help=(
            "Cache policy: "
            "'auto' = prompt on match, regen on mismatch (default); "
            "'reuse' = reuse if compatible, else regen; "
            "'rerun' = always regen and overwrite; "
            "'require' = only plot from cache, error if missing/mismatched; "
            "'off' = regen and DO NOT save."
        ),
    )
