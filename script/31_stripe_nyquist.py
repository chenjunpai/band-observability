"""31 -- the strongest test in the project: directional Nyquist at FIXED sensor
count, across three viscosities.

THE CLAIM
---------
A stripe layout with p bands samples the torus with period 2*pi/p in y, so modes
k_y and k_y + p are indistinguishable to it.  The observation operator is
therefore singular on the determining band |k| <= K_c(nu) whenever p <= 2*K_c,
and non-singular as soon as p >= 2*K_c + 1.  The prediction is

    synchronisation requires  p >= p* = 2 K_c(nu) + 1

with NO free parameter, and p* MOVES with viscosity (9, 11, 15 for
K_c = 4, 5, 7) while any fixed coverage radius does not.

WHY THIS BEATS THE LATTICE VERSION (scripts/25)
---------------------------------------------------
In a lattice family m, n = m^2 and h = pi/m are all locked together, so a
lattice threshold is compatible with "you need more sensors" and with "you need
smaller h".  In the stripe family the sensor count is held at n = 784 (with
`jitter_mode="rowshift"`; the old per-point jitter dropped 20% of the sensors,
giving 588-783) and only the band pitch changes.  Measured in
results/19_anisotropy_ablation_fix at
nu = 5e-3 (K_c = 5, p* = 11), with the plateau criterion of nolab.verdict:

    p =  3  h = 0.987   plateau 1.3     FAIL     (mu = 200 diverges to 1e3)
    p =  4  h = 0.743   plateau 1.0     FAIL     (mu = 50  diverges to 1e11)
    p =  6  h = 0.501   plateau 5.9e-1  FAIL
    p =  8  h = 0.396   plateau 4.0e-1  FAIL
    p = 12  h = 0.264   plateau 5.9e-3  SYNC
    p = 16  h = 0.220   plateau 1.8e-5  SYNC

6/6 with the sensor count fixed and h sweeping monotonically from 0.99 to 0.22.
Note p = 6 and p = 12 occupy the SAME number of grid rows (12, at thickness 2
and 1): what decides the outcome is the band pitch, not the row count and not
the coverage.  Note also that p = 6 sits at h = 0.501, comfortably below the
h_c ~ 0.6 that POSITIONING_v3 claimed, and fails -- that was the pre-registered
decision point of 19-fix, and h lost it.

The geometry alone already shows the step (scripts/30, nu = 5e-3, K = 5,
floor 0, width_factor 1):

    p =  8   delta_K = -0.0294
    p =  9   delta_K = -0.0084
    p = 10   delta_K = -0.0021
    p = 11   delta_K = +0.2055     <- p* = 2*K_c + 1
    p = 12   delta_K = +0.2121

delta_K jumps by two orders of magnitude between p = 10 and p = 11.  This script
tests whether the dynamics follow that jump, and whether the jump moves with nu.

WHAT IS BEING RISKED
--------------------
Written down before the run, per nu:
  * every p <  p* fails (plateau > 0.15)
  * every p >= p* + 2 synchronises (plateau < 1e-2)
  * p = p* and p* + 1 are allowed to be PARTIAL: in the lattice family the
    threshold is necessary but not sufficient, full synchronisation arriving
    about 3 pitches above it, and the same is expected here.
If instead the threshold sits at a fixed h across the three viscosities, the
coverage-radius story is right after all and this script is the cleanest
possible evidence for it.  Either way it is publishable; the current draft
asserts one of them without having run it.

RESOLUTION NOTE
---------------
This script runs at N = 128.  CAREFUL: results/12_reynolds measures K_c at
N = 256 for nu <= 5e-3 (those states are under-resolved at N = 128 by the
repository's own resolution_ok criterion).  So for nu = 5e-3 and 2.5e-3 the
quoted K_c comes from an N = 256 run while the stripe test here is at N = 128
-- a cross-resolution point that MUST be stated in the paper, and K_c at
N = 128 should be re-measured to confirm the threshold.

    python scripts/31_stripe_nyquist.py            # ~70 min
    MUS=10,50 SEEDS=0 python scripts/31_stripe_nyquist.py   # ~20 min, smoke
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save
from nolab.observations import covering_radius
from nolab import stripes_exact, delta_K, corridor_diagnostics
from nolab.verdict import outcome, best_verdict, VERDICT_ORDER

OUT = ROOT / "results" / "31_stripe_nyquist"
N_GRID = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
T = float(os.environ.get("T", 8.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))       # dynamics floor
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]

# nu -> (K_c from results/12_reynolds, band counts to sweep around p* = 2K_c+1)
# K_c values are the ones AFTER the K = 7, 9 refill of script 12:
#   nu 1.5e-2 -> N_c 81  -> K_c 4     nu 5.0e-3 -> N_c 121 -> K_c 5
#   nu 2.5e-3 -> N_c 225 -> K_c 7
CASES = {
    1.5e-2: (4, [5, 6, 7, 8, 9, 10, 12]),
    5.0e-3: (5, [6, 8, 9, 10, 11, 12, 14]),
    2.5e-3: (7, [10, 12, 13, 14, 15, 16, 18]),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    rows = []
    for nu, (K_c, ps) in sorted(CASES.items(), reverse=True):
        flow, w = get_truth(nu, N=N_GRID, dt=dt, T_spin=30.0)
        g = flow.g
        p_star = 2 * K_c + 1
        print(f"\nnu={nu:g}  K_c={K_c}  predicted p* = {p_star}  "
              f"(n = {N_SENS} fixed)")
        for p in ps:
            try:
                ix, iy, meta = stripes_exact(g.N, N_SENS, p, seed=0, jitter=True)
            except ValueError as exc:
                print(f"   p={p}: skipped ({exc})")
                continue
            h = float(covering_radius(g, ix, iy))
            dK0, dmax = delta_K(g, ix, iy, K_c, 0.0)
            dKf, _ = delta_K(g, ix, iy, K_c, FLOOR)
            corr = corridor_diagnostics(g, ix, iy)["max_corridor"]
            per_run, verdicts = [], []
            for seed in SEEDS:
                ix, iy, meta = stripes_exact(g.N, N_SENS, p, seed=seed,
                                             jitter=True)
                for mu in MUS:
                    obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
                    r = trial(flow, w, obs, mu, T=T, init_seed=seed)
                    v = outcome(r["ts"], r["err"])
                    per_run.append(dict(seed=seed, mu=mu, verdict=v["verdict"],
                                        plateau=v["plateau"],
                                        plateau_rel=v["plateau_rel"],
                                        rate=r["rate"],
                                        rate_is_meaningful=v["rate_is_meaningful"],
                                        status_old=r["status"],
                                        converged_old=r["converged"],
                                        diverged=r["diverged"],
                                        final=r["final"],
                                        ts=r["ts"], err=r["err"]))
                    verdicts.append(v["verdict"])
            bv = best_verdict(verdicts)
            rels = [x["plateau_rel"] for x in per_run
                    if x["plateau_rel"] is not None]
            best_plateau = float(min(rels)) if rels else None
            predicted = "pass" if p >= p_star else "fail"
            observed = "pass" if bv == "SYNC" else (
                "partial" if bv == "PARTIAL" else "fail")
            holds = ((predicted == "fail" and observed == "fail")
                     or (predicted == "pass" and observed in ("pass", "partial")))
            rows.append(dict(nu=nu, N_grid=N_GRID, K_c=K_c, p=p,
                             p_star=p_star, n_sensors=N_SENS,
                             n_actual=int(meta["n_actual"]),
                             thickness=int(meta["thickness"]),
                             rows_occupied=int(p * meta["thickness"]),
                             gap=float(meta["gap"]), h=h, max_corridor=corr,
                             delta_K_floor0=dK0, delta_K_floor=dKf,
                             delta_K_max_eig=dmax,
                             best_verdict=bv, best_plateau_rel=best_plateau,
                             predicted=predicted, observed=observed,
                             prediction_holds=bool(holds), per_run=per_run))
            flag = "OK " if holds else "!! "
            print(f"  {flag}p={p:3d} rows={p*meta['thickness']:3d} h={h:.3f} "
                  f"corr={corr:.2f} dK={dK0:+.4f} "
                  f"plateau={best_plateau if best_plateau is None else '%.1e' % best_plateau} "
                  f"{bv:<8} predicted={predicted}")

    hold = sum(r["prediction_holds"] for r in rows)
    print(f"\nprediction p >= 2*K_c+1 holds in {hold}/{len(rows)} cases")
    below = [r for r in rows if r["p"] < r["p_star"]]
    nfail = sum(r["best_verdict"] in ("FAIL", "DIVERGED") for r in below)
    print(f"necessity: {nfail}/{len(below)} of the p < p* cases fail outright")
    save(rows, dict(N_grid=N_GRID, n_sensors=N_SENS, T=T, dt=dt, mus=MUS,
                    seeds=SEEDS, floor=FLOOR,
                    cases={str(k): v for k, v in CASES.items()},
                    verdict_criterion="nolab.verdict.outcome (plateau)",
                    prediction=("sync requires p >= 2*K_c+1; the sensor count "
                                "is held fixed so the threshold cannot be a "
                                "count or a coverage-radius effect"),
                    prediction_holds=f"{hold}/{len(rows)}"),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
