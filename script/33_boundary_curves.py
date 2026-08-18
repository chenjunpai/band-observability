"""33 -- the Fig 1 dataset: boundary layouts rerun with the error curves kept.

WHY A RERUN IS NEEDED AT ALL
----------------------------
Every results file in the repository except 27 stores only `final`, the error at
the single instant t = T.  On a plateau that oscillates by a factor of ~3 (see
the curves in results/27_rate_stability), a single sample can land on either
side of a verdict threshold.  `scripts/29_reclassify.py` reclassifies from
`final` and flags every row whose plateau is within a factor 2 of a threshold as
`needs_curve`; those flags are what this script resolves.

Rather than rerunning everything -- most rows are nowhere near a threshold and
their verdicts are already safe -- this reruns the layouts that Fig 1 and the
counterexample pairs actually rest on, and keeps the whole curve.

THE PAIRS THIS HAS TO NAIL DOWN
-------------------------------
POSITIONING_v4 §1.1 and §1.2 both rest on rows that turned out to be
misclassified ("uniform n=300 converges at rate 0.79", "uniform n=400 converges
at rate 1.96" -- both plateau at 5.8e-2 and 1.2e-1 and do not converge).  Their
replacements, from results/19_anisotropy_ablation_fix, are:

  same coverage radius, opposite outcome
    uniform  n=784-equivalent n=400   h = 0.483   delta_K = -0.050   plateau 1.2e-1
    corridor gw = 0.4, seed 1         h = 0.483   delta_K = +0.069   plateau 8.0e-5

  smaller coverage radius, worse outcome (both at n = 784)
    stripes_exact p = 8               h = 0.396   delta_K = -0.029   plateau 4.0e-1
    corridor gw = 0.4, seed 1         h = 0.483   delta_K = +0.069   plateau 8.0e-5

Both pairs live in the band where a single `final` sample is not enough, so both
have to be re-measured from curves before they can carry a claim in the paper.

WHAT IT PRODUCES
----------------
For each layout: h, max corridor width, delta_K at the canonical setting
(K = 5, floor 0, width_factor 1) and at the dynamics setting (floor 0.25), and
per (mu, observer-init seed) the full curve plus the plateau verdict.  That is
everything Fig 1 needs: plateau vs delta_K as the main panel, plateau vs h as
the inset that fails to collapse.

Observer-initial-condition seeds are swept explicitly.  In results/27 the same
(layout, mu) gave rates of 0.057 / 4.73 / 25.98 across three observer inits --
a factor of 450 -- so the init seed is a bigger source of variance than the
sensor seed near the boundary, and it has never been reported.

    python scripts/33_boundary_curves.py        # ~90 min with the defaults
    MUS=10,50 INIT_SEEDS=0 python scripts/33_boundary_curves.py   # ~25 min
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save, GENERATORS
from nolab.observations import covering_radius
from nolab import (stripes_exact, corridor, lattice_m, delta_K,
                       corridor_diagnostics)
from nolab.verdict import outcome, best_verdict

OUT = ROOT / "results" / "33_boundary_curves"
NU = float(os.environ.get("NU", 5e-3))
T = float(os.environ.get("T", 8.0))
N_SENS = int(os.environ.get("N_SENS", 784))
K_BAND = int(os.environ.get("K_BAND", 5))
FLOOR = float(os.environ.get("FLOOR", 0.25))
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]
SENSOR_SEEDS = [int(x) for x in os.environ.get("SENSOR_SEEDS", "0,1").split(",")]
INIT_SEEDS = [int(x) for x in os.environ.get("INIT_SEEDS", "0,1").split(",")]

# The set is deliberately small and deliberately spans the transition: three
# clear failures, three clear successes, and everything in the ambiguous band
# between delta_K = -0.10 and +0.05 where the single-sample verdicts are unsafe.
SPEC = [
    ("uniform",  dict(n=250)),          # dK -0.099   expected FAIL
    ("uniform",  dict(n=300)),          # dK -0.086   boundary
    ("uniform",  dict(n=400)),          # dK -0.050   boundary  <- §1.1 pair
    ("uniform",  dict(n=500)),          # dK +0.035   boundary
    ("uniform",  dict(n=784)),          # dK +0.157   control, clear SYNC
    ("stripes",  dict(n_stripes=8)),    # dK -0.029   FAIL       <- §1.2 pair
    ("stripes",  dict(n_stripes=10)),   # dK -0.002   just below p* = 11
    ("stripes",  dict(n_stripes=11)),   # dK +0.206   at p*
    ("stripes",  dict(n_stripes=12)),   # dK +0.212   clear SYNC
    ("corridor", dict(gap_width=0.4)),  # dK +0.069   SYNC       <- both pairs
    ("corridor", dict(gap_width=0.8)),  # dK +0.014   boundary
    ("corridor", dict(gap_width=1.2)),  # dK -0.215   FAIL
    ("lattice",  dict(m=10)),           # dK -0.001   FAIL
    ("lattice",  dict(m=12)),           # dK +0.001   boundary
]


def build(g, kind, spec, seed):
    if kind == "uniform":
        ix, iy = GENERATORS["uniform"](g.N, spec["n"], seed=seed)
        return ix, iy, f"uniform_n{spec['n']}"
    if kind == "stripes":
        ix, iy, _ = stripes_exact(g.N, N_SENS, spec["n_stripes"], seed=seed,
                                  jitter=True)
        return ix, iy, f"stripes_p{spec['n_stripes']}"
    if kind == "corridor":
        ix, iy, _ = corridor(g.N, N_SENS, spec["gap_width"], seed=seed)
        return ix, iy, f"corridor_gw{spec['gap_width']}"
    if kind == "lattice":
        ix, iy, _ = lattice_m(g.N, spec["m"])
        return ix, iy, f"lattice_m{spec['m']}"
    raise ValueError(kind)


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    g = flow.g

    rows = []
    print(f"{'layout':<18}{'ss':>3}{'h':>7}{'corr':>7}{'dK(f0)':>10}"
          f"{'plateau':>10}  verdict   init spread")
    for kind, spec in SPEC:
        for ss in SENSOR_SEEDS:
            ix, iy, label = build(g, kind, spec, ss)
            if kind == "lattice" and ss != SENSOR_SEEDS[0]:
                continue                       # lattice is deterministic
            h = float(covering_radius(g, ix, iy))
            corr = corridor_diagnostics(g, ix, iy)["max_corridor"]
            dK0, dmax = delta_K(g, ix, iy, K_BAND, 0.0)
            dKf, _ = delta_K(g, ix, iy, K_BAND, FLOOR)
            per_run, verdicts = [], []
            for mu in MUS:
                for init in INIT_SEEDS:
                    obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
                    r = trial(flow, w, obs, mu, T=T, record_every=2,
                              init_seed=init)
                    v = outcome(r["ts"], r["err"])
                    verdicts.append(v["verdict"])
                    per_run.append(dict(mu=mu, init_seed=init,
                                        verdict=v["verdict"],
                                        plateau=v["plateau"],
                                        plateau_rel=v["plateau_rel"],
                                        e0=v["e0"], rate=r["rate"],
                                        rate_is_meaningful=v["rate_is_meaningful"],
                                        converged_old=r["converged"],
                                        status_old=r["status"],
                                        diverged=r["diverged"],
                                        final=r["final"],
                                        ts=r["ts"], err=r["err"]))
            rels = [x["plateau_rel"] for x in per_run
                    if x["plateau_rel"] is not None]
            best = float(min(rels)) if rels else None
            bv = best_verdict(verdicts)
            # how much of the scatter is the observer initial condition alone?
            spread = {}
            for mu in MUS:
                vals = [x["plateau_rel"] for x in per_run
                        if x["mu"] == mu and x["plateau_rel"] is not None]
                if len(vals) >= 2:
                    spread[str(mu)] = float(max(vals) / max(min(vals), 1e-30))
            max_spread = max(spread.values()) if spread else None
            rows.append(dict(layout=label, kind=kind, spec=spec,
                             sensor_seed=ss, h=h, max_corridor=corr,
                             delta_K_floor0=dK0, delta_K_floor=dKf,
                             delta_K_max_eig=dmax, K=K_BAND,
                             best_verdict=bv, best_plateau_rel=best,
                             init_spread_by_mu=spread,
                             max_init_spread=max_spread, per_run=per_run))
            print(f"{label:<18}{ss:>3}{h:>7.3f}{corr:>7.2f}{dK0:>+10.4f}"
                  f"{'n/a' if best is None else '%.1e' % best:>10}  "
                  f"{bv:<9} "
                  f"{'n/a' if max_spread is None else '%.1fx' % max_spread}")

    # the two claims Fig 1 is supposed to make, checked on this data alone
    def rank_violations(key):
        pts = [(r[key], r["best_plateau_rel"], r["layout"]) for r in rows
               if r["best_plateau_rel"] is not None]
        bad = []
        for i in range(len(pts)):
            for j in range(len(pts)):
                # a better (larger delta_K / smaller h) layout doing worse
                better = pts[i][0] > pts[j][0] if key != "h" else pts[i][0] < pts[j][0]
                if better and pts[i][1] > 3.0 * pts[j][1]:
                    bad.append((pts[i][2], pts[j][2]))
        return bad

    v_dK = rank_violations("delta_K_floor0")
    v_h = rank_violations("h")
    print(f"\nordering violations (a layout that should be better plateauing "
          f">3x worse):")
    print(f"   by delta_K(floor 0): {len(v_dK)}")
    print(f"   by coverage radius h: {len(v_h)}")
    if v_dK:
        print("   delta_K violations:", sorted(set(v_dK))[:6])

    save(rows, dict(nu=NU, T=T, dt=dt, n_sensors=N_SENS, mus=MUS,
                    sensor_seeds=SENSOR_SEEDS, init_seeds=INIT_SEEDS,
                    K_band=K_BAND, dynamics_floor=FLOOR,
                    canonical_delta_K="K=5, width_factor=1, denom_floor=0",
                    verdict_criterion="nolab.verdict.outcome (plateau)",
                    ordering_violations=dict(delta_K=len(v_dK), h=len(v_h)),
                    question=("does the plateau collapse on delta_K or on h, "
                              "measured from curves rather than from a single "
                              "final sample?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
