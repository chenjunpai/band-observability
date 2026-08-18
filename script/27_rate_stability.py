"""27 -- is the mu-to-mu scatter physics, or is it the fit?

THE OBSERVATION THAT NEEDS EXPLAINING
-------------------------------------
From `results/17_config_ablation`, uniform n = 400, seed 0:

    mu = 10   rate 1.87  (converged)
    mu = 20   rate 0.13  (weak)
    mu = 50   rate 1.96  (converged)
    mu = 100  rate 0.56  (converged)

A factor of 15 between neighbouring mu, non-monotone, on the same layout and the
same truth.  Either the observer really is that sensitive to mu -- which would
be a headline result in its own right and must be investigated -- or
`sync_rate`'s single fitting window is picking up a plateau on some runs.  The
v3 tables cannot distinguish the two, because scripts 17 and 19 store only the
scalar rate: the error curves were discarded.  Every table in the paper that
takes a max over mu inherits this uncertainty.

WHAT THIS SCRIPT DOES
---------------------
Re-runs the suspicious sweeps at higher temporal resolution, KEEPS the full
error curves in the results file, and refits each one under four protocols
(early window, late window, tighter floor, Theil-Sen).  A run whose four
estimates agree within 1.5x is trustworthy; one that does not is flagged, and
its curve is available for plotting.

Run this before touching anything else: if the estimator is unstable, every
downstream ranking in the paper is provisional.

    python scripts/27_rate_stability.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save, GENERATORS
from nolab import sync_rate_multi

OUT = ROOT / "results" / "27_rate_stability"
NU = float(os.environ.get("NU", 5e-3))
T = float(os.environ.get("T", 8.0))
MUS = [float(x) for x in os.environ.get(
    "MUS", "5,10,15,20,30,50,75,100,150,200").split(",")]
CASES = [("uniform", 400, 0), ("uniform", 400, 1), ("uniform", 300, 0),
         ("uniform", 300, 1), ("uniform", 784, 0)]
INIT_SEEDS = [int(x) for x in os.environ.get("INIT_SEEDS", "0,1,2").split(",")]


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    g = flow.g
    rows = []
    for layout, n, seed in CASES:
        ix, iy = GENERATORS[layout](g.N, n, seed=seed)
        for mu in MUS:
            for init in INIT_SEEDS:
                obs = FixedPointObs(g, ix, iy, denom_floor=0.25)
                r = trial(flow, w, obs, mu, T=T, record_every=2,
                          init_seed=init)
                m = sync_rate_multi(r["ts"], r["err"])
                rows.append(dict(layout=layout, n=n, sensor_seed=seed,
                                 init_seed=init, mu=mu,
                                 rate_v2=r["rate"], status_v2=r["status"],
                                 rate_median=m["rate"],
                                 rate_spread=m["rate_spread"],
                                 rates_by_protocol=m["rates"],
                                 fit_stable=m["stable"],
                                 ts=r["ts"], err=r["err"]))
            sel = [x for x in rows if x["n"] == n and x["sensor_seed"] == seed
                   and x["mu"] == mu]
            print(f"  {layout} n={n} s={seed} mu={mu:<6g} "
                  f"v2={np.mean([x['rate_v2'] for x in sel]):.3f} "
                  f"median={np.mean([x['rate_median'] for x in sel]):.3f} "
                  f"init_spread={np.ptp([x['rate_median'] for x in sel]):.3f} "
                  f"{'' if all(x['fit_stable'] for x in sel) else 'FIT UNSTABLE'}")

    unstable = [dict(layout=r["layout"], n=r["n"], mu=r["mu"],
                     seed=r["sensor_seed"], init=r["init_seed"],
                     spread=r["rate_spread"])
                for r in rows if not r["fit_stable"]]
    print(f"\n{len(unstable)}/{len(rows)} runs have protocol-dependent rates")
    save(rows, dict(nu=NU, T=T, dt=dt, mus=MUS, cases=CASES,
                    init_seeds=INIT_SEEDS, n_unstable=len(unstable),
                    unstable=unstable,
                    question=("is the mu-to-mu scatter in 17 physical or a "
                              "fitting-window artifact, and how much of it is "
                              "observer-initial-condition variance?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
