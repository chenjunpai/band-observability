"""29 -- reclassify every result already in the repository.  No dynamics.

WHY THIS RUNS FIRST
-------------------
`nolab.metrics.sync_rate` marks a run "converged" whenever the error halved at
some point and a log-linear fit through the initial transient has a negative
slope; it never checks where the error ends up (see nolab/verdict.py).  In
the idealised setting a genuinely synchronising observer reaches 1e-5, and the
runs that carry the largest reported rates in this repository plateau at 0.2-0.5
instead.  So every table that reads `converged` or ranks by `rate` is currently
reporting the slope of a transient in runs that never synchronised.

This script rewrites the verdict for every stored run using the plateau, and
prints what changed.  It touches nothing: it reads results/*/results.json and
writes results/29_reclassify/.

Run it before rerunning any physics.  Roughly half of the open questions in
NEXT_STEPS_INDEX are answered by reclassification alone.

HOW IT WORKS
------------
It walks each results file recursively and picks up every dict that looks like a
trial: one carrying an error curve (`err`) or, failing that, a final error
(`final`).  Curve-carrying rows (currently only 27_rate_stability) get the
tail-median verdict; the rest get the weaker single-sample verdict, and any row
whose plateau lands within a factor 3 of a threshold -- the measured plateau
oscillation -- is flagged `needs_curve`, which is the input list for
scripts/33_fig1_curves.py.

Files with no per-run error at all (12_reynolds, 21_shell_gain_fix, 26) store
only aggregated rates and cannot be reclassified; they are listed as such, and
that is itself the finding: they have to be rerun if their conclusions are to
survive.

    python scripts/29_reclassify.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab.harness import save
from nolab.verdict import (outcome, verdict_from_final, is_ambiguous,
                               floor_from_conditions, SYNC_TOL, PARTIAL_TOL)

OUT = ROOT / "results" / "29_reclassify"
RESULTS = ROOT / "results"
EXTRA_FILES = [ROOT / "exp14" / "zeroshot.json"]

# keys that identify a run, in the order we want them in the label
LABEL_KEYS = ["method", "layout", "family", "kind", "config", "name",
              "nu", "m", "n", "n_requested", "n_stripes", "gap_width", "rho",
              "width_factor", "floor", "denom_floor", "noise", "nu_err",
              "m_assim", "M", "loc", "mu", "seed", "sensor_seed", "init_seed"]


def _label(chain):
    """Build a readable label from the dict and its ancestors."""
    merged = {}
    for d in chain:
        for k in LABEL_KEYS:
            if k in d and not isinstance(d[k], (list, dict)):
                merged.setdefault(k, d[k])
    parts = []
    for k in LABEL_KEYS:
        if k in merged:
            v = merged[k]
            parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(parts) or "?"


def _conditions(chain):
    """Pull run conditions out of the dict chain so sync_tol can be set to the
    physical floor rather than the idealised 1e-2."""
    c = dict(noise=0.0, nu_err=0.0, dt_obs=None)
    for d in chain:
        for k in ("noise", "obs_noise"):
            if isinstance(d.get(k), (int, float)):
                c["noise"] = max(c["noise"], float(d[k]))
        if isinstance(d.get("nu_err"), (int, float)):
            c["nu_err"] = max(c["nu_err"], abs(float(d["nu_err"])))
        if isinstance(d.get("dt_obs"), (int, float)):
            c["dt_obs"] = float(d["dt_obs"])
        if isinstance(d.get("m_assim"), (int, float)) and d["m_assim"] > 1:
            # dt is 0.002 throughout this repository
            c["dt_obs"] = max(c["dt_obs"] or 0.0, float(d["m_assim"]) * 0.002)
    return c


def _is_trial(d):
    if not isinstance(d, dict):
        return False
    if isinstance(d.get("err"), list) and len(d["err"]) >= 4:
        return True
    return "final" in d and isinstance(d.get("final"), (int, float, type(None)))


def walk(node, chain=(), found=None):
    """Collect (chain, dict) for every trial-looking dict."""
    found = [] if found is None else found
    if isinstance(node, dict):
        chain2 = chain + (node,)
        if _is_trial(node):
            found.append(chain2)
        for v in node.values():
            if isinstance(v, (dict, list)):
                walk(v, chain2, found)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                walk(v, chain, found)
    return found


def reclassify_file(path):
    data = json.loads(Path(path).read_text())
    rows, changed, ambiguous = [], 0, 0
    for chain in walk(data):
        d = chain[-1]
        cond = _conditions(chain)
        floor = floor_from_conditions(**cond)
        sync_tol = max(SYNC_TOL, 3.0 * floor)
        if isinstance(d.get("err"), list) and len(d["err"]) >= 4:
            v = outcome(d.get("ts"), d["err"], sync_tol=sync_tol)
            source = "curve"
        else:
            v = verdict_from_final(d.get("final"), sync_tol=sync_tol,
                                   diverged=bool(d.get("diverged", False)))
            source = "final"
        old_conv = bool(d.get("converged", False))
        new_conv = v["verdict"] == "SYNC"
        amb = (source == "final") and is_ambiguous(v["plateau_rel"],
                                                   sync_tol, PARTIAL_TOL)
        changed += int(old_conv != new_conv)
        ambiguous += int(amb)
        rows.append(dict(label=_label(chain), source=source,
                         verdict=v["verdict"], plateau=v["plateau"],
                         plateau_rel=v["plateau_rel"],
                         sync_tol=sync_tol, conditions=cond,
                         rate_old=d.get("rate"), status_old=d.get("status"),
                         converged_old=old_conv, converged_new=new_conv,
                         flipped=bool(old_conv != new_conv),
                         needs_curve=bool(amb),
                         rate_is_meaningful=v["rate_is_meaningful"]))
    return rows, changed, ambiguous


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(p for p in RESULTS.glob("*/results.json")
                   if "29_reclassify" not in str(p))
    files += [p for p in EXTRA_FILES if p.exists()]
    if not files:
        raise SystemExit(f"no results found under {RESULTS}")

    all_rows, per_file, unusable = {}, {}, []
    for p in files:
        name = p.parent.name if p.name == "results.json" else p.stem
        try:
            rows, changed, amb = reclassify_file(p)
        except Exception as exc:                      # keep going on odd files
            unusable.append(dict(file=str(p.relative_to(ROOT)),
                                 reason=f"unreadable: {exc}"))
            continue
        if not rows:
            unusable.append(dict(
                file=str(p.relative_to(ROOT)),
                reason=("no per-run error stored (only aggregated rates); "
                        "cannot be reclassified without rerunning")))
            continue
        all_rows[name] = rows
        c = Counter(r["verdict"] for r in rows)
        per_file[name] = dict(n=len(rows), verdicts=dict(c), flipped=changed,
                              needs_curve=amb,
                              curve_backed=sum(r["source"] == "curve"
                                               for r in rows))
        print(f"\n=== {name}  ({len(rows)} runs, "
              f"{'curves' if rows[0]['source'] == 'curve' else 'final only'}) ===")
        print("   verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
        print(f"   converged flag flips: {changed}/{len(rows)}"
              f"   ambiguous (need curve rerun): {amb}")
        flips = [r for r in rows if r["flipped"] and r["converged_old"]]
        for r in sorted(flips, key=lambda r: -(r["rate_old"] or 0))[:8]:
            print(f"     was converged rate={r['rate_old']:.2f} -> "
                  f"{r['verdict']} plateau={r['plateau_rel']:.2e}   {r['label']}")

    if unusable:
        print("\n=== cannot be reclassified (no per-run error stored) ===")
        for u in unusable:
            print(f"   {u['file']}: {u['reason']}")

    # the headline number
    tot = sum(v["n"] for v in per_file.values())
    flip = sum(v["flipped"] for v in per_file.values())
    amb = sum(v["needs_curve"] for v in per_file.values())
    print(f"\n{flip}/{tot} runs change verdict; {amb} need a curve rerun "
          f"(scripts/33_fig1_curves.py)")

    save(dict(per_file=per_file, rows=all_rows, unusable=unusable),
         dict(sync_tol_idealised=SYNC_TOL, partial_tol=PARTIAL_TOL,
              n_runs=tot, n_flipped=flip, n_needs_curve=amb,
              criterion=("verdict from the plateau (tail median of the error "
                         "curve, or the single stored final error), not from "
                         "the slope of the initial transient"),
              supersedes=("the `converged` flag of nolab.metrics.sync_rate in "
                          "every results file listed here")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
