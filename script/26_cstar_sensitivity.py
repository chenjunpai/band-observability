"""26 -- how much of N_c(nu) survives the choice of c*?  (no simulation needed)

THE ISSUE
---------
Throughout v2/v3, N_c is "the first K whose mean rate crosses c* = 0.75".  That
is an iso-rate contour of a continuous quantity, not a synchronisation
threshold: in `results/12_reynolds` at nu = 1.5e-2 the K = 2 observation (25
modes) already has mean rate 0.25 > 0, so the system synchronises at a fifth of
the reported "critical" count.  Calling it the critical observation count
invites exactly the objection the paper cannot afford, since the count is the
quantity the whole configuration story is about.

Two things follow and both are cheap:
  * report N_c for a range of c* and show the ORDERING in nu is unchanged (that
    is the defensible claim), or discover that it is not;
  * note the K grid.  Script 12 sweeps K in {2, 3, 4, 5, 6, 8, 10}: K = 7
    (225 modes) is missing, so the reported 169 -> 289 step at nu = 2.5e-3 is a
    property of the grid, not of the flow.  That gap must be filled by rerunning
    12 with K = 7 before any staircase claim is made; this script flags every
    N_c whose predecessor level was not tested.

    python scripts/26_cstar_sensitivity.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import save

OUT = ROOT / "results" / "26_cstar_sensitivity"
SRC = ROOT / "results" / "12_reynolds" / "results.json"
C_STARS = [float(x) for x in os.environ.get(
    "C_STARS", "0.25,0.5,0.75,1.0,1.25").split(",")]


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run scripts/12_reynolds.py first")
    data = json.loads(SRC.read_text())["rows"]
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for nu_key, rec in data.items():
        per_K = {int(k): v for k, v in rec["per_K"].items()}
        Ks = sorted(per_K)
        means = {K: float(np.mean(per_K[K]["rates"])) for K in Ks}
        entry = dict(nu=nu_key, grashof=rec.get("grashof"),
                     K_grid=Ks, N_grid=rec.get("N_grid"),
                     rate_at_smallest_K=means[Ks[0]],
                     synchronises_below_Nc=bool(means[Ks[0]] > 0.05),
                     per_cstar={})
        for c in C_STARS:
            hit = next((K for K in Ks if means[K] >= c), None)
            if hit is None:
                entry["per_cstar"][str(c)] = dict(K_c=None, N_c=None,
                                                  gap_in_K_grid=None)
                continue
            i = Ks.index(hit)
            prev = Ks[i - 1] if i > 0 else None
            gap = bool(prev is not None and hit - prev > 1)
            entry["per_cstar"][str(c)] = dict(
                K_c=hit, N_c=per_K[hit]["ndof"], mean_rate=means[hit],
                gap_in_K_grid=gap,
                untested_K=(list(range(prev + 1, hit)) if gap else []))
        rows.append(entry)
        line = "  ".join(f"c*={c}:N_c={entry['per_cstar'][str(c)]['N_c']}"
                         + ("*" if entry["per_cstar"][str(c)].get("gap_in_K_grid")
                            else "")
                         for c in C_STARS)
        print(f"nu={nu_key}  {line}   (rate at K={Ks[0]}: "
              f"{means[Ks[0]]:.2f})")

    print("\n* = the level below N_c was never tested (K grid gap); that step "
          "is a grid artifact until K is filled in.")
    save(rows, dict(source=str(SRC), c_stars=C_STARS,
                    question=("is the N_c ordering in nu robust to the rate "
                              "threshold, and which steps are K-grid artifacts?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
