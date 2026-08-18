"""39 -- close the m grid at nu = 2.5e-3 so "offset = 7" is a measurement.

`results/32_lattice_aliasing_fix` sweeps m = 13..18, 20 and its extension adds
22, 24, 26.  It reports the first SYNC at m = 22 and therefore an offset of 7
above m* = 15.  But m = 19 and m = 21 were never run, so the measurement only
bounds the offset to {5, 6, 7}.  PAPER_DATA_APPENDIX quotes 7 as a number.

This fills 19 and 21 (and 23, 25 for a monotone check).  It is the cheapest
open item in the repository: four to six lattice runs.

The same gap does not exist at the other two viscosities -- their m grids are
consecutive -- so only nu = 2.5e-3 is run here by default.

    python scripts/39_lattice_offset_fill.py             # ~25 min
    MS=19,21 python scripts/39_lattice_offset_fill.py    # ~12 min
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save, lattice_m, delta_K
from nolab.observations import covering_radius
from nolab.verdict import outcome, best_verdict

OUT = ROOT / "results" / "39_lattice_offset_fill"
NU = float(os.environ.get("NU", 2.5e-3))
K_C = int(os.environ.get("K_C", 7))
N_GRID = int(os.environ.get("N_GRID", 128))
MS = [int(x) for x in os.environ.get("MS", "19,21,23,25").split(",")]
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]
INIT_SEEDS = [int(x) for x in os.environ.get("INIT_SEEDS", "0,1").split(",")]
T = float(os.environ.get("T", 8.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=N_GRID, dt=dt, T_spin=30.0)
    g = flow.g
    m_star = 2 * K_C + 1
    print(f"nu={NU:g}  K_c={K_C}  m*={m_star}  N={N_GRID}")
    rows = []
    for m in MS:
        ix, iy, meta = lattice_m(g.N, m)
        h = float(covering_radius(g, ix, iy))
        dK, dmax = delta_K(g, ix, iy, K_C, 0.0)
        per_run, verdicts = [], []
        for mu in MUS:
            obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
            for s in INIT_SEEDS:
                r = trial(flow, w, obs, mu, T=T, init_seed=s)
                v = outcome(r["ts"], r["err"])
                per_run.append(dict(mu=mu, init_seed=s, verdict=v["verdict"],
                                    plateau_rel=v["plateau_rel"],
                                    diverged=r["diverged"], final=r["final"],
                                    ts=r["ts"], err=r["err"]))
                verdicts.append(v["verdict"])
        rels = [x["plateau_rel"] for x in per_run
                if x["plateau_rel"] is not None]
        best = float(min(rels)) if rels else None
        bv = best_verdict(verdicts)
        rows.append(dict(nu=NU, N_grid=N_GRID, K_c=K_C, m=m, m_star=m_star,
                         n=int(meta["n_actual"]), h=h, delta_K=dK,
                         delta_K_max_eig=dmax, delta_K_K=K_C,
                         delta_K_floor=0.0,
                         best_verdict=bv, best_plateau_rel=best,
                         offset=m - m_star, per_run=per_run))
        print(f"   m={m:3d} n={meta['n_actual']:4d} h={h:.3f} dK={dK:+.5f} "
              f"plateau={best:.2e} {bv}  (offset {m-m_star})")

    prev = ROOT / "results" / "32_lattice_aliasing_fix"
    known = []
    for p in (prev / "results.json", prev / "extension_nu25" / "results.json"):
        if p.exists():
            for r in json.loads(p.read_text())["rows"]:
                if abs(r.get("nu", 0) - NU) < 1e-12:
                    known.append((r["m"], r["best_verdict"]))
    allm = sorted(set([(r["m"], r["best_verdict"]) for r in rows] + known))
    first = next((m for m, v in allm if v == "SYNC"), None)
    print(f"\n merged m ladder: {[(m, v) for m, v in allm]}")
    if first is None:
        print(" no SYNC yet -- extend MS upward before quoting an offset")
    else:
        print(f" first SYNC at m={first}  ->  offset = {first - m_star} "
              f"above m* = {m_star}")
    save(rows, dict(nu=NU, K_c=K_C, m_star=m_star, N_grid=N_GRID, T=T, dt=dt,
                    mus=MUS, init_seeds=INIT_SEEDS, floor=FLOOR,
                    merged_ladder=[[m, v] for m, v in allm],
                    first_sync=first,
                    offset_above_m_star=(None if first is None
                                         else first - m_star),
                    fills="m=19,21 (and 23,25) missing from 32_lattice_"
                          "aliasing_fix, which bounded the offset only to "
                          "{5,6,7}"),
         str(OUT))
    print(f"written to {OUT/'results.json'}")


if __name__ == "__main__":
    main()
