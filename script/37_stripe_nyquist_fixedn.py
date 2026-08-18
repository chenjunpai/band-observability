"""37 -- stripe Nyquist, rerun with the sensor count actually held fixed.

WHAT WAS WRONG WITH 31
----------------------
31 is the paper's strongest experiment and its whole point is that the sensor
count is held at n = 784 so that the threshold cannot be a counting effect.  It
called `stripes_exact(..., jitter=True)`, whose per-point +-1 jitter collides on
the grid; the deduplication then deletes the collisions.  Realised counts in
`results/31_stripe_nyquist`:

    p        5    6    7    8    9   10   11   12   13   14   15   16   18
    n      668  686  588  615  651  668  684  686  709  718  742  746  783

n rises with p, and the outcome rises with p, so the alternative explanation the
experiment exists to exclude is exactly the one that co-varies with the answer.
`jitter_mode="rowshift"` (see nolab/configs_fix.py) shifts each occupied row by
a uniform offset instead: it breaks row-to-row alignment just as well, creates
no collisions, and gives n_actual == 784 for every p and seed.  `strict_count`
makes the script abort rather than quietly ship a different number.

THREE OTHER CHANGES
-------------------
1.  Scoring is reported at two levels, because 31 conflated them and RESULTS.md
    §2.2 quotes the wrong one:
      necessity      no p < p* reaches SYNC.        31 satisfies this 21/21.
      pre-registered every p < p* is FAIL.          31 satisfies this 16/21;
                                                    the 5 misses are PARTIAL.
    Both go in the table.  The abstract may only use the first.
2.  Initial-condition spread.  `results/33` records `max_init_spread` up to
    9.2x and does not report it; near the SYNC/PARTIAL line (1e-2) a 9x spread
    decides the verdict.  Two init seeds per sensor seed, and the spread is
    stored so Fig 1 can carry error bars.
3.  delta_K is recorded with its full identity (K, width_factor, denom_floor,
    jitter_mode).  The +0.206 at p = 11 previously reported is a point-jitter
    number; the collision-free value is +0.073.  Quoting either without the mode
    is not reproducible.

RESOLUTION
----------
Runs at N = 128 by default like 31, but nu = 5e-3 and 2.5e-3 are UNDER-RESOLVED
at N = 128 by the repository's own criterion, and K_c for those two comes from
N = 256 runs in 12_reynolds.  Run `scripts/36_resolution_audit.py` first and set
K_C_OVERRIDE / N_GRID from its Part B / Part C output.  Do not publish this
table until 36 has passed.

    python scripts/37_stripe_nyquist_fixedn.py                  # ~80 min
    NUS=0.005 PS=8,10,11,12 python scripts/37_stripe_nyquist_fixedn.py  # ~25 min
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
from nolab.verdict import outcome, best_verdict

OUT = ROOT / "results" / "37_stripe_nyquist_fixedn"
N_GRID = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
T = float(os.environ.get("T", 8.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))          # dynamics floor
DK_FLOOR = float(os.environ.get("DK_FLOOR", 0.0))     # diagnostic floor (exp30)
JITTER_MODE = os.environ.get("JITTER_MODE", "rowshift")
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]
SENSOR_SEEDS = [int(x) for x in os.environ.get("SENSOR_SEEDS", "0,1").split(",")]
INIT_SEEDS = [int(x) for x in os.environ.get("INIT_SEEDS", "0,1").split(",")]

# nu -> (K_c, band counts).  K_c MUST come from a run at N_GRID; see script 36.
CASES = {
    1.5e-2: (4, [5, 6, 7, 8, 9, 10, 12]),
    5.0e-3: (5, [6, 8, 9, 10, 11, 12, 14]),
    2.5e-3: (7, [10, 12, 13, 14, 15, 16, 18]),
}
if os.environ.get("NUS"):
    keep = {float(x) for x in os.environ["NUS"].split(",")}
    CASES = {k: v for k, v in CASES.items() if k in keep}
if os.environ.get("PS"):
    ps = [int(x) for x in os.environ["PS"].split(",")]
    CASES = {k: (v[0], ps) for k, v in CASES.items()}
if os.environ.get("K_C_OVERRIDE"):                    # "0.005:5,0.0025:6"
    for tok in os.environ["K_C_OVERRIDE"].split(","):
        nu_s, kc_s = tok.split(":")
        nu = float(nu_s)
        if nu in CASES:
            CASES[nu] = (int(kc_s), CASES[nu][1])


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
        print(f"\nnu={nu:g}  K_c={K_c}  p*={p_star}  N={N_GRID}  "
              f"n={N_SENS} (strict)  jitter={JITTER_MODE}")
        for p in ps:
            try:
                ix0, iy0, meta0 = stripes_exact(
                    g.N, N_SENS, p, seed=SENSOR_SEEDS[0], jitter=True,
                    jitter_mode=JITTER_MODE, strict_count=True)
            except (ValueError, RuntimeError) as exc:
                print(f"   p={p}: skipped ({exc})")
                continue
            h = float(covering_radius(g, ix0, iy0))
            dK, dmax = delta_K(g, ix0, iy0, K_c, DK_FLOOR)
            corr = corridor_diagnostics(g, ix0, iy0)["max_corridor"]

            per_run, verdicts, n_seen = [], [], set()
            for s_sens in SENSOR_SEEDS:
                ix, iy, meta = stripes_exact(
                    g.N, N_SENS, p, seed=s_sens, jitter=True,
                    jitter_mode=JITTER_MODE, strict_count=True)
                n_seen.add(int(meta["n_actual"]))
                obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
                for mu in MUS:
                    for s_init in INIT_SEEDS:
                        r = trial(flow, w, obs, mu, T=T, init_seed=s_init)
                        v = outcome(r["ts"], r["err"])
                        per_run.append(dict(
                            sensor_seed=s_sens, init_seed=s_init, mu=mu,
                            verdict=v["verdict"], plateau=v["plateau"],
                            plateau_rel=v["plateau_rel"],
                            rate=(r["rate"] if v["rate_is_meaningful"] else None),
                            diverged=r["diverged"], final=r["final"],
                            ts=r["ts"], err=r["err"]))
                        verdicts.append(v["verdict"])
            assert n_seen == {N_SENS}, f"count not fixed: {n_seen}"

            rels = [x["plateau_rel"] for x in per_run
                    if x["plateau_rel"] is not None]
            best = float(min(rels)) if rels else None
            bv = best_verdict(verdicts)
            # spread over init seeds, at the mu/sensor-seed that did best
            spread = None
            groups = {}
            for x in per_run:
                if x["plateau_rel"] is not None:
                    groups.setdefault((x["sensor_seed"], x["mu"]), []).append(
                        x["plateau_rel"])
            sp = [max(v) / min(v) for v in groups.values() if len(v) > 1 and min(v) > 0]
            spread = float(max(sp)) if sp else None

            necessity_ok = not (p < p_star and bv == "SYNC")
            prereg_ok = ((p < p_star and bv == "FAIL")
                         or (p >= p_star and bv in ("SYNC", "PARTIAL")))
            rows.append(dict(
                nu=nu, N_grid=N_GRID, K_c=K_c, p=p, p_star=p_star,
                n_requested=N_SENS, n_actual=int(meta0["n_actual"]),
                count_exact=bool(meta0["count_exact"]),
                jitter_mode=JITTER_MODE, thickness=int(meta0["thickness"]),
                rows_occupied=int(p * meta0["thickness"]),
                gap=float(meta0["gap"]), h=h, max_corridor=corr,
                delta_K=dK, delta_K_max_eig=dmax, delta_K_K=K_c,
                delta_K_floor=DK_FLOOR, delta_K_width_factor=1.0,
                best_verdict=bv, best_plateau_rel=best,
                max_init_spread=spread,
                necessity_ok=bool(necessity_ok),
                prereg_ok=bool(prereg_ok), per_run=per_run))
            print(f"   p={p:3d} n={meta0['n_actual']} rows={p*meta0['thickness']:3d} "
                  f"h={h:.3f} dK={dK:+.4f} plateau={best:.2e} {bv:<8} "
                  f"spread={'n/a' if spread is None else '%.1fx' % spread}  "
                  f"necessity={'ok' if necessity_ok else 'VIOLATED'} "
                  f"prereg={'ok' if prereg_ok else 'miss'}")

    n_nec = sum(r["necessity_ok"] for r in rows)
    n_pre = sum(r["prereg_ok"] for r in rows)
    print(f"\nnecessity (no SYNC below p*):        {n_nec}/{len(rows)}")
    print(f"pre-registered (every p<p* is FAIL): {n_pre}/{len(rows)}")
    print("Report BOTH.  Only the first may appear in the abstract.")
    save(rows, dict(N_grid=N_GRID, n_sensors=N_SENS, T=T, dt=dt, mus=MUS,
                    sensor_seeds=SENSOR_SEEDS, init_seeds=INIT_SEEDS,
                    dynamics_floor=FLOOR, delta_K_floor=DK_FLOOR,
                    jitter_mode=JITTER_MODE,
                    cases={str(k): v for k, v in CASES.items()},
                    necessity_holds=f"{n_nec}/{len(rows)}",
                    prereg_holds=f"{n_pre}/{len(rows)}",
                    supersedes="31_stripe_nyquist (jitter deleted 3-22% of the "
                               "sensors, so the fixed-count claim did not hold)",
                    K_c_source="MUST be a run at N_grid; see 36_resolution_audit"),
         str(OUT))
    print(f"written to {OUT/'results.json'}")


if __name__ == "__main__":
    main()
