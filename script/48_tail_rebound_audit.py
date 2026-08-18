"""48 -- the tail-half rebound audit.  Applies the diagnostic that RESULTS.md
section 0 already prescribes to every run in results/44_truth_ensemble.  No
dynamics, no re-run: it reads the stored error curves (~2 s).

    python scripts/48_tail_rebound_audit.py

WHAT THIS TESTS
---------------
`verdict.outcome()` calls the median of the tail 25% of the error curve the
"plateau".  RESULTS.md section 0 already warns that on a curve which has not
flattened, that number is "the error at about 0.85T", not a plateau, and
prescribes the fix: compare the median of the FIRST half of the tail (A) with
the median of the SECOND half (B).

  B/A < 1   still descending -- the verdict may be pessimistic ("not yet")
  B/A ~ 1   genuinely flat   -- the verdict means what it says
  B/A > 1   RISING           -- the tail median is reading a transient dip and
                               the verdict is optimistic

The third case was never checked.  It is the one that matters here, because an
exactly blind subspace (scripts/47) is not damped by the nudging term at all:
its content is set by the flow, so the total error dips when the flow happens to
drain it and rebounds when the flow refills it.  Non-monotone tails are the
dynamical signature of the rank deficiency, and they are what a below-threshold
layout should look like.

RESULT (T = 16 ladder, nu = 5e-3, K_c = 5, threshold p* = m* = 11)
------------------------------------------------------------------
    family    q   runs   median B/A   rebound (>1.3)
    stripes  10     24       1.30         12/24     <- below threshold
    stripes  11     24       0.76          0/24
    stripes  12     24       0.28          0/24
    lattice  12      8       1.23          1/8
    lattice  13      8       1.86          7/8
    lattice  14      8       0.98          2/8
    lattice  15     24       0.40          0/24

Of the 120 runs, 51 are scored SYNC.  Exactly ONE of those 51 has a rising tail,
and it is the run the necessity claim rests on:

    stripes p=10, truth 1, mu=5, init_seed=0
        tail first half   2.337e-03
        tail second half  6.001e-03      B/A = 2.57  (the largest in the file)
        minimum at        t = 13.70      not at T = 16
        final            7.313e-03

Its sibling seed (mu=5, seed 1) has B/A = 1.94, and all 8 runs of that row
rebound (1.53 to 2.57).  The row scores SYNC because bestmu_median = 7.36e-3 is
the mean of one SYNC (3.13e-3) and one PARTIAL (1.16e-2) -- with two init seeds
the "median" aggregation is a coin flip.  For contrast, p = 11 truth 1 at the
same mu goes 3.61e-3 -> 1.00e-3 (B/A = 0.28) with its minimum at t = 15.98.

So the sub-threshold crossing is a transient dip caught by the tail median, not
a slow convergence.  Necessity is not broken and does not need to be softened.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results" / "44_truth_ensemble" / "results.json"
REBOUND = float(os.environ.get("REBOUND_THRESHOLD", 1.3))


def tail_split(err, frac=0.25):
    """(A, B, B/A) medians of the two halves of the last `frac` of the curve."""
    rel = np.asarray(err, float) / float(err[0])
    tail = rel[int((1.0 - frac) * len(rel)):]
    h = len(tail) // 2
    A = float(np.median(tail[:h]))
    B = float(np.median(tail[h:]))
    return A, B, B / A


def audit(path):
    d = json.load(open(path))
    per_row, contaminated, rows_out = {}, [], []
    for r in d["rows"]:
        for pr in r["per_run"]:
            rel = np.asarray(pr["err"], float) / pr["err"][0]
            ts = np.asarray(pr["ts"], float)
            A, B, ratio = tail_split(pr["err"])
            rec = dict(family=r["family"], q=r["q"], truth_seed=r["truth_seed"],
                       mu=pr["mu"], init_seed=pr["init_seed"],
                       verdict=pr["verdict"], delta_K=r.get("delta_K"),
                       tail_A=A, tail_B=B, tail_ratio=ratio,
                       t_min=float(ts[rel.argmin()]), t_end=float(ts[-1]),
                       rel_min=float(rel.min()), rel_final=float(rel[-1]),
                       rebound=bool(ratio > REBOUND))
            rows_out.append(rec)
            per_row.setdefault((r["family"], r["q"]), []).append(ratio)
            if pr["verdict"] == "SYNC" and ratio > REBOUND:
                contaminated.append(rec)

    print(f"{'family':>8} {'q':>3} {'runs':>5} {'median B/A':>11} {'rebound':>9}")
    summary = {}
    for k in sorted(per_row):
        v = np.array(per_row[k])
        summary[f"{k[0]}_{k[1]}"] = dict(runs=len(v), median_ratio=float(np.median(v)),
                                         n_rebound=int((v > REBOUND).sum()))
        print(f"{k[0]:>8} {k[1]:>3} {len(v):>5} {np.median(v):>11.2f} "
              f"{str(int((v > REBOUND).sum())) + '/' + str(len(v)):>9}")

    n_sync = sum(1 for r in rows_out if r["verdict"] == "SYNC")
    print(f"\n{len(rows_out)} runs, {n_sync} scored SYNC.")
    print(f"SYNC verdicts whose error is RISING at the end (B/A > {REBOUND}):")
    if not contaminated:
        print("  none")
    for c in contaminated:
        print(f"  {c['family']} p={c['q']} truth{c['truth_seed']} mu={c['mu']} "
              f"seed={c['init_seed']}: A={c['tail_A']:.3e} B={c['tail_B']:.3e} "
              f"B/A={c['tail_ratio']:.2f} t_min={c['t_min']:.2f}/{c['t_end']:.0f}")
    return dict(meta=dict(source=str(path), rebound_threshold=REBOUND,
                          n_runs=len(rows_out), n_sync=n_sync,
                          n_sync_contaminated=len(contaminated),
                          note=("B/A is median(second half of tail 25%) / "
                                "median(first half); >1 means the error is "
                                "rising and the plateau median is reading a dip")),
                by_family_q=summary, contaminated=contaminated, rows=rows_out)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    out = audit(src)
    dest = ROOT / "results" / "48_tail_rebound_audit"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dest / 'results.json'}")
