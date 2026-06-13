"""Generate LaTeX \\newcommand macros from code-computed calibration values.

Reads ``outputs/data/calibration/<aircraft>.yaml`` for each aircraft and
writes ``outputs/data/paper_macros.tex``.  The paper includes this file via::

    \\InputIfFileExists{../../soaring-ctrw/outputs/data/paper_macros.tex}{}%
      {\\newcommand{\\HfitPara}{0.881}...}

so that every numeric quantity in the manuscript is driven by the code output,
not by a value typed by hand.

**Prerequisites**:

1. Run ``scripts/estimate_sigma_theta.py --write`` for all aircraft to
   populate the ``sigma_theta`` section of the calibration YAMLs.
2. Run ``scripts/plot_msd_all_aircraft.py --write-hfit`` to populate the
   ``h_fit`` section.

Usage::

    python scripts/write_paper_macros.py [--out PATH]

Options::

    --out PATH    Path for the output .tex file
                  (default: outputs/data/paper_macros.tex)
    --aircraft    Subset of aircraft to include
                  (default: paragliders hang_gliders sailplanes)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when run from repo root.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from soaring_ctrw.calibration import read_calibration
from soaring_ctrw.paths import DATA_DIR

# Mapping from aircraft key → short TeX suffix used in macro names
# (must be unique, no underscores or spaces, valid in TeX command names)
AIRCRAFT_SUFFIXES = {
    "paragliders":  "Para",
    "hang_gliders": "Hang",
    "sailplanes":   "Sail",
}
AIRCRAFT_ORDER = ("paragliders", "hang_gliders", "sailplanes")

DEFAULT_OUT = DATA_DIR / "paper_macros.tex"


def _safe(val: object, fmt: str = ".3f") -> str:
    """Format a float; return 'NaN' if missing or non-finite."""
    try:
        v = float(val)
        import math
        if not math.isfinite(v):
            return "NaN"
        return format(v, fmt)
    except (TypeError, ValueError):
        return "NaN"


def build_macros(aircraft_list: list[str]) -> list[tuple[str, str, str]]:
    """Return a list of (macro_name, value_str, comment) tuples.

    Each entry will be written as::

        \\newcommand{\\MacroName}{value_str}  % comment
    """
    rows: list[tuple[str, str, str]] = []

    for aircraft in aircraft_list:
        suf = AIRCRAFT_SUFFIXES.get(aircraft)
        if suf is None:
            print(f"  WARNING: unknown aircraft key {aircraft!r}, skipping.",
                  file=sys.stderr)
            continue

        cal = read_calibration(aircraft)
        if not cal:
            print(f"  WARNING: no calibration YAML for {aircraft!r}.", file=sys.stderr)

        # ---- sigma_theta* -----------------------------------------------
        sigma_sec = cal.get("sigma_theta", {})
        sigma_val = sigma_sec.get("value", None)
        sigma_se  = sigma_sec.get("value_se", None)

        rows.append((
            f"SigmaStar{suf}",
            _safe(sigma_val, ".3f"),
            f"{aircraft}: calibrated sigma_theta* (rad)",
        ))
        rows.append((
            f"SigmaStar{suf}SE",
            _safe(sigma_se, ".3f"),
            f"{aircraft}: sigma_theta* replica s.e. (rad)",
        ))

        # ---- H_fit -------------------------------------------------------
        hfit_sec = cal.get("h_fit", {})
        h_val  = hfit_sec.get("h_fit", None)
        h_se   = hfit_sec.get("h_fit_replica_se", None)

        rows.append((
            f"Hfit{suf}",
            _safe(h_val, ".3f"),
            f"{aircraft}: fitted H (EA-MSD log-log)",
        ))
        rows.append((
            f"Hfit{suf}SE",
            _safe(h_se, ".3f"),
            f"{aircraft}: H_fit replica s.e.",
        ))

        # ---- critical time t_c and memory length n_c ---------------------
        crit_sec = cal.get("critical_time", {})
        tc_val = crit_sec.get("t_c", None)
        nc_val = crit_sec.get("n_c", None)

        rows.append((
            f"Tc{suf}",
            _safe(tc_val, ".0f"),
            f"{aircraft}: crossover time t_c (s, integer)",
        ))
        rows.append((
            f"Nc{suf}",
            _safe(nc_val, ".1f"),
            f"{aircraft}: directional-memory length n_c (cycles)",
        ))
        # Kilo variants for compact display (e.g. "4.9 x 10^3 s")
        try:
            import math
            tc_kilo = float(tc_val) / 1000.0
            tc_kilo_str = _safe(tc_kilo, ".1f") if math.isfinite(tc_kilo) else "NaN"
        except (TypeError, ValueError):
            tc_kilo_str = "NaN"
        rows.append((
            f"Tc{suf}Kilo",
            tc_kilo_str,
            f"{aircraft}: t_c / 10^3 s (one decimal place)",
        ))

        # ---- bare-cycle sigma_theta* -------------------------------------
        by_mode = sigma_sec.get("by_mode", {})
        bare_val = by_mode.get("bare", None)
        by_mode_se = sigma_sec.get("by_mode_se", {})
        bare_se = by_mode_se.get("bare", None)

        rows.append((
            f"SigmaStarBare{suf}",
            _safe(bare_val, ".3f"),
            f"{aircraft}: bare-cycle calibrated sigma_theta* (rad)",
        ))
        rows.append((
            f"SigmaStarBare{suf}SE",
            _safe(bare_se, ".3f"),
            f"{aircraft}: bare-cycle sigma_theta* replica s.e. (rad)",
        ))

    # ---- Cross-class summary: max replica SE (used in equation display)
    h_se_vals: list[float] = []
    for aircraft in aircraft_list:
        suf = AIRCRAFT_SUFFIXES.get(aircraft)
        if suf is None:
            continue
        cal = read_calibration(aircraft)
        h_se = cal.get("h_fit", {}).get("h_fit_replica_se", None)
        try:
            v = float(h_se)
            import math
            if math.isfinite(v):
                h_se_vals.append(v)
        except (TypeError, ValueError):
            pass
    se_all = max(h_se_vals) if h_se_vals else float("nan")
    rows.append((
        "HfitSEAll",
        _safe(se_all, ".3f"),
        "max H_fit replica s.e. across aircraft classes (for equation display)",
    ))

    return rows


def write_tex(rows: list[tuple[str, str, str]], out_path: Path) -> None:
    """Write the \\newcommand block to ``out_path``."""
    lines: list[str] = []
    lines.append(
        "% AUTO-GENERATED by scripts/write_paper_macros.py — do not edit.\n"
        "% Re-generate with:\n"
        "%   python scripts/estimate_sigma_theta.py --write  (all aircraft)\n"
        "%   python scripts/plot_msd_all_aircraft.py --write-hfit\n"
        "%   python scripts/write_paper_macros.py\n"
        "%"
    )
    prev_aircraft = None
    for macro, val, comment in rows:
        # Insert a blank comment line between aircraft blocks
        suf = comment.split(":")[0].strip()
        if suf != prev_aircraft:
            lines.append(f"% ---- {suf}")
            prev_aircraft = suf
        lines.append(
            rf"\newcommand{{\{macro}}}{{{val}}}  % {comment}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output .tex file (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--aircraft",
        nargs="+",
        default=list(AIRCRAFT_ORDER),
        choices=list(AIRCRAFT_ORDER),
        help="Aircraft classes to include.",
    )
    args = parser.parse_args()

    print(f"Reading calibration YAMLs for: {', '.join(args.aircraft)}")
    rows = build_macros(args.aircraft)

    write_tex(rows, args.out)
    print(f"Written {len(rows)} macros → {args.out}")

    # Pretty-print for inspection
    print()
    for macro, val, comment in rows:
        print(f"  \\{macro:<22} {val:<8}  % {comment}")


if __name__ == "__main__":
    main()
