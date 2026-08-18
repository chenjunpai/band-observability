"""41 -- the stripe Nyquist ladder, rerun on layouts without the truncation hole
and aggregated by a rule that is not "best of sixteen draws".

TWO CHANGES FROM 37, BOTH FORCED BY 37's OWN OUTPUT
---------------------------------------------------
1.  THE LAYOUTS.  `results/37_stripe_nyquist_fixedn` holds n at 784 by
    truncating the tail, which removes `deficit` consecutive sensors from the
    last band.  `scripts/40_stripe_deltaK_v2.py` shows what that costs: h is
    non-monotone in p (p=13 h=0.491, p=15 h=0.396, both worse than p=12 h=0.405)
    and delta_K is non-monotone and flips sign one band LATE at nu = 2.5e-3
    (p = 16 instead of p* = 15).  Spread the deficit and delta_K rises
    monotonically with p and flips exactly on p* for K_c = 4, 5 and 7.  The
    dynamics in 37 therefore ran on layouts carrying an unintended directional
    hole, which makes its plateaus pessimistic and its delta_K column unusable
    for ordering the family.

2.  THE AGGREGATION.  37 reports `best_verdict` = the most favourable verdict and
    `best_plateau_rel` = the minimum plateau over all 16 runs
    (4 mu x 2 sensor seeds x 2 observer-init seeds).  The spread over those 16 is
    up to 35x, which is wider than the SYNC/PARTIAL window itself (1e-2 to 0.15,
    a factor of 15).  Taking the minimum is the same selection bias the project
    already retired when it stopped reporting max-over-mu of the rate, and it
    changes conclusions: under the median over the 16 runs, nu = 5e-3 p = 12
    drops from SYNC to PARTIAL and the first SYNC of that viscosity moves from
    p = 12 to p = 14.

    The defensible split is that mu is a DESIGN parameter -- an implementer gets
    to tune it -- whereas the sensor seed and the observer's initial condition
    are nuisance draws nobody gets to choose.  So: optimise over mu, aggregate
    over seeds.  This script reports all three of

        best-of-all     min over the 16 runs                (37's rule, kept for
                                                             comparability)
        bestmu-median   min over mu of the median over seeds (HEADLINE)
        bestmu-worst    min over mu of the worst over seeds  (conservative)

    and flags every p whose verdict differs between them.  On 37's data the
    headline rule changes two verdicts (nu=1.5e-2 p=6 and nu=5e-3 p=9, both
    PARTIAL -> FAIL) and leaves the first SYNC of each viscosity unchanged, which
    incidentally IMPROVES the pre-registered strong rule from 15/21 to 17/21.

WHAT IS BEING TESTED (unchanged from 37, restated so it can be checked)
-----------------------------------------------------------------------
p bands sample y with period 2*pi/p, so k_y and k_y + p are indistinguishable and
the observation operator is singular on |k| <= K_c whenever p <= 2 K_c.  With the
sensor count held at exactly 784 for every p, neither "more sensors" nor "smaller
h" can account for a threshold at p* = 2 K_c + 1.  K_c is taken from the
c* = 0.75 crossing measured AT N = 128 in results/36_resolution_audit part B
(4, 5, 7); K_C_OVERRIDE exists because that crossing is delicate at nu = 2.5e-3
(K = 6 scores 0.734 at N = 128, 0.767 at N = 192, 0.727 at N = 256, i.e. it
straddles 0.75), so the ladder must be reported under K_c = 6 as well.

    python scripts/41_stripe_nyquist_v3.py                     # ~110 min
    NUS=0.005 python scripts/41_stripe_nyquist_v3.py            # ~35 min
    K_C_OVERRIDE=0.0025:6 python scripts/41_stripe_nyquist_v3.py
    PS=10,11 MUS=10 SENSOR_SEEDS=0 T=1 python scripts/41_stripe_nyquist_v3.py  # smoke
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save, delta_K
from nolab.observations import covering_radius
from nolab.stripes_v2 import stripes_v2
from nolab.verdict import outcome, best_verdict, SYNC_TOL, PARTIAL_TOL

OUT = ROOT / "results" / "41_stripe_nyquist_v3"
N_GRID = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
T = float(os.environ.get("T", 8.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))          # dynamics denom_floor
DK_FLOOR = float(os.environ.get("DK_FLOOR", 0.0))     # diagnostic floor (exp30)
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]
SENSOR_SEEDS = [int(x) for x in os.environ.get("SENSOR_SEEDS", "0,1").split(",")]
INIT_SEEDS = [int(x) for x in os.environ.get("INIT_SEEDS", "0,1").split(",")]

CASES = {
    1.5e-2: (4, [5, 6, 7, 8, 9, 10, 12]),
    5.0e-3: (5, [6, 8, 9, 10, 11, 12, 14]),
    2.5e-3: (7, [10, 12, 13, 14, 15, 16, 17, 18]),   # 17 was never tested in 37
}
if os.environ.get("NUS"):
    keep = {float(x) for x in os.environ["NUS"].split(",")}
    CASES = {k: v for k, v in CASES.items() if k in keep}
if os.environ.get("PS"):
    ps = [int(x) for x in os.environ["PS"].split(",")]
    CASES = {k: (v[0], ps) for k, v in CASES.items()}
if os.environ.get("K_C_OVERRIDE"):                    # "0.0025:6"
    for tok in os.environ["K_C_OVERRIDE"].split(","):
        nu_s, kc_s = tok.split(":")
        nu = float(nu_s)
        if nu in CASES:
            CASES[nu] = (int(kc_s), CASES[nu][1])


def verdict_of(rel):
    if rel is None:
        return "NO_DATA"
    return ("SYNC" if rel < SYNC_TOL
            else ("PARTIAL" if rel < PARTIAL_TOL else "FAIL"))


def aggregate(per_run):
    """The three aggregation rules, side by side.

    per_run entries carry `mu` and `plateau_rel`.  `bestmu_median` is the
    headline: optimise over the design parameter mu, aggregate over the nuisance
    seeds with a median.
    """
    by_mu = {}
    for r in per_run:
        if r["plateau_rel"] is not None:
            by_mu.setdefault(r["mu"], []).append(r["plateau_rel"])
    if not by_mu:
        return dict(best_of_all=None, bestmu_median=None, bestmu_worst=None,
                    spread=None, verdicts={}, agree=False)
    allv = [v for l in by_mu.values() for v in l]
    boa = float(min(allv))
    med = float(min(float(np.median(l)) for l in by_mu.values()))
    wor = float(min(float(max(l)) for l in by_mu.values()))
    vs = dict(best_of_all=verdict_of(boa), bestmu_median=verdict_of(med),
              bestmu_worst=verdict_of(wor))
    return dict(best_of_all=boa, bestmu_median=med, bestmu_worst=wor,
                spread=float(max(allv) / max(min(allv), 1e-30)),
                verdicts=vs, agree=bool(len(set(vs.values())) == 1),
                per_mu_median={str(k): float(np.median(v))
                               for k, v in sorted(by_mu.items())})


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
              f"n={N_SENS} exact, deficit spread")
        print(f"   {'p':>3}{'def':>5}{'h':>7}{'dK':>10}"
              f"{'best-all':>11}{'bestmu-med':>12}{'bestmu-wst':>12}  verdict(med)")
        for p in ps:
            try:
                ix0, iy0, meta = stripes_v2(N_GRID, N_SENS, p, seed=SENSOR_SEEDS[0])
            except (ValueError, RuntimeError) as exc:
                print(f"   p={p}: skipped ({exc})")
                continue
            h = float(covering_radius(g, ix0, iy0))
            dK, dmax = delta_K(g, ix0, iy0, K_c, DK_FLOOR)
            per_run = []
            for s_sens in SENSOR_SEEDS:
                ix, iy, _ = stripes_v2(N_GRID, N_SENS, p, seed=s_sens)
                for mu in MUS:
                    for s_init in INIT_SEEDS:
                        obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
                        r = trial(flow, w, obs, mu, T=T, init_seed=s_init)
                        v = outcome(r["ts"], r["err"])
                        per_run.append(dict(
                            sensor_seed=s_sens, mu=mu, init_seed=s_init,
                            verdict=v["verdict"], plateau_rel=v["plateau_rel"],
                            plateau=v["plateau"], rate=r["rate"],
                            rate_is_meaningful=v["rate_is_meaningful"],
                            diverged=r["diverged"], final=r["final"],
                            ts=r["ts"], err=r["err"]))
            agg = aggregate(per_run)
            headline = agg["verdicts"].get("bestmu_median", "NO_DATA")
            predicted = "pass" if p >= p_star else "fail"
            necessity_ok = (p >= p_star) or (headline != "SYNC")
            prereg_ok = ((p < p_star and headline in ("FAIL", "DIVERGED"))
                         or (p >= p_star and headline in ("SYNC", "PARTIAL")))
            rows.append(dict(nu=nu, N_grid=N_GRID, K_c=K_c, p=p,
                             p_star=p_star, n_requested=N_SENS,
                             n_actual=int(meta["n_actual"]),
                             count_exact=True, deficit=int(meta["deficit"]),
                             thickness=int(meta["thickness"]),
                             per_row=int(meta["per_row"]),
                             rows_occupied=int(meta["rows_occupied"]),
                             gap=float(meta["gap"]), h=h,
                             delta_K=dK, delta_K_K=K_c,
                             delta_K_floor=DK_FLOOR, delta_K_width_factor=1.0,
                             delta_K_max_eig=dmax,
                             generator="stripes_v2 (deficit spread)",
                             aggregation=agg, verdict=headline,
                             verdict_best_of_all=agg["verdicts"].get("best_of_all"),
                             verdict_worst=agg["verdicts"].get("bestmu_worst"),
                             aggregation_rules_agree=agg["agree"],
                             predicted=predicted,
                             necessity_ok=bool(necessity_ok),
                             prereg_ok=bool(prereg_ok),
                             per_run=per_run))
            def f(x):
                return "n/a" if x is None else f"{x:.2e}"
            print(f"   {p:>3}{meta['deficit']:>5}{h:>7.3f}{dK:>+10.4f}"
                  f"{f(agg['best_of_all']):>11}{f(agg['bestmu_median']):>12}"
                  f"{f(agg['bestmu_worst']):>12}  {headline}"
                  + ("" if agg["agree"] else "   (rules disagree)")
                  + ("   <- p*" if p == p_star else ""))

    nec = sum(r["necessity_ok"] for r in rows)
    pre = sum(r["prereg_ok"] for r in rows)
    disagree = [dict(nu=r["nu"], p=r["p"], verdicts=r["aggregation"]["verdicts"])
                for r in rows if not r["aggregation_rules_agree"]]
    first_sync = {}
    for r in rows:
        if r["verdict"] == "SYNC":
            k = str(r["nu"])
            first_sync[k] = min(first_sync.get(k, 10 ** 9), r["p"])
    offsets = {k: v - (2 * dict((str(n), c) for n, (c, _) in CASES.items())[k] + 1)
               for k, v in first_sync.items()}
    print(f"\nnecessity (no SYNC below p*): {nec}/{len(rows)}")
    print(f"pre-registered strong rule (every p<p* is FAIL): {pre}/{len(rows)}")
    print(f"first SYNC per nu: {first_sync}   offset above p*: {offsets}")
    print(f"rows where the three aggregation rules disagree: {len(disagree)}")
    for d in disagree:
        print(f"   nu={d['nu']:g} p={d['p']}: {d['verdicts']}")
    save(rows, dict(N_grid=N_GRID, n_sensors=N_SENS, T=T, dt=dt, mus=MUS,
                    sensor_seeds=SENSOR_SEEDS, init_seeds=INIT_SEEDS,
                    dynamics_floor=FLOOR, delta_K_floor=DK_FLOOR,
                    generator="nolab.stripes_v2 (deficit spread, rowshift)",
                    cases={str(k): v for k, v in CASES.items()},
                    K_c_source=("results/36_resolution_audit part B, c*=0.75, "
                                "measured at N=128, the grid used here"),
                    headline_aggregation=("min over mu of the median over "
                                          "(sensor seed, observer init); mu is a "
                                          "design parameter, the seeds are not"),
                    necessity_holds=f"{nec}/{len(rows)}",
                    prereg_holds=f"{pre}/{len(rows)}",
                    first_sync=first_sync, offsets_above_p_star=offsets,
                    aggregation_disagreements=disagree,
                    supersedes=("37_stripe_nyquist_fixedn: its layouts carry a "
                                "hole from tail truncation, and it aggregated by "
                                "the minimum over 16 runs whose spread reaches "
                                "35x")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
