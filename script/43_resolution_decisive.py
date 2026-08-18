"""43 -- is the nu = 2.5e-3 result physics or grid?  The one remaining run that
can change a claim.

WHY THIS IS THE DECIDING RUN
----------------------------
Two facts sit on top of each other and have to be separated.

  * `results/36_resolution_audit` part A: by the repository's own criterion (tail
    enstrophy above 0.8 * 2/3 * N/2 must be under 1e-4 of the total) the N = 128
    truth field is

        nu = 1.5e-2   tail 1.0e-6   OK
        nu = 5.0e-3   tail 2.6e-4   over by 2.6x
        nu = 2.5e-3   tail 1.7e-3   over by 17x        <- worst

  * `results/41_stripe_nyquist_v3`: under the headline aggregation the stripe
    family reaches SYNC at nu = 1.5e-2 (p = 10) and nu = 5e-3 (p = 12), and does
    NOT reach it at nu = 2.5e-3 anywhere in p <= 18.

The one viscosity where sufficiency fails is the one whose grid is 17x outside
tolerance.  Until that coincidence is broken, "sufficiency not reached at
nu = 2.5e-3" is not a statement about the flow.

Necessity is at stake too, in the opposite direction.  At N = 256 the truth
carries more small-scale energy, so a layout could plausibly do WORSE and a
below-threshold p could do better; if any p < p* = 15 reaches SYNC at N = 256,
the 22/22 necessity claim fails at that resolution and the paper has to restrict
itself to nu >= 5e-3.  That is why the ladder here includes p = 12 and 14, not
just the ones above the threshold.

WHAT COMES OUT
--------------
For each p, the verdict at N = 128 and N = 256 under the same three aggregation
rules used in scripts/41, plus delta_K at both resolutions.  Four outcomes, and
the sentence to write for each is fixed in advance:

  A. N=256 reaches SYNC above p* and no p < p* does
     -> "sufficiency at nu = 2.5e-3 is resolution-limited: not reached at
        N = 128, reached at p = <x> at N = 256".  Necessity stands at both
        resolutions.  This is the best case and it REMOVES an open item.
  B. N=256 still reaches no SYNC, necessity holds
     -> "not reached at either resolution up to p = 18"; the failure is not a
        grid artefact, and the offset at this viscosity is simply larger than the
        swept range.  Also fine, and now defensible.
  C. some p < p* reaches SYNC at N = 256
     -> necessity is resolution-dependent.  Restrict every threshold claim to
        nu >= 5e-3 and report this as the boundary of validity.  Do NOT quietly
        drop the viscosity.
  D. verdicts move but stay in the same class (e.g. PARTIAL -> PARTIAL with a
     10x lower plateau)
     -> report the plateau shift as a resolution sensitivity in Limitations; the
        claims stand.

COST
----
Measured on this repository: a T = 8 trial takes ~47 s at N = 128 and ~205 s at
N = 256.  The default grid (5 values of p, 4 mu, 2 observer inits, both
resolutions) is 40 trials at each resolution: ~30 min at N = 128 and ~2.3 h at
N = 256.  The N = 128 half is a consistency check against exp41 and can be
skipped with GRIDS=256 once you trust it.

Lean version, still decisive (drops mu = 5 and 20, keeps one p below the
threshold and two above):

    PS=14,15,18 MUS=10,50 python scripts/43_resolution_decisive.py   # ~50 min

Full:

    python scripts/43_resolution_decisive.py
    GRIDS=256 python scripts/43_resolution_decisive.py               # ~2.3 h
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
from nolab.verdict import outcome, SYNC_TOL, PARTIAL_TOL

OUT = ROOT / "results" / "43_resolution_decisive"
NU = float(os.environ.get("NU", 2.5e-3))
K_C = int(os.environ.get("K_C", 7))          # c*=0.75 crossing at N=128 AND 256
N_SENS = int(os.environ.get("N_SENS", 784))
T = float(os.environ.get("T", 8.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))
GRIDS = [int(x) for x in os.environ.get("GRIDS", "128,256").split(",")]
PS = [int(x) for x in os.environ.get("PS", "12,14,15,16,18").split(",")]
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]
INIT_SEEDS = [int(x) for x in os.environ.get("INIT_SEEDS", "0,1").split(",")]
WANT_DK = os.environ.get("DELTA_K", "1") not in ("0", "no", "false")


def verdict_of(rel):
    if rel is None:
        return "NO_DATA"
    return ("SYNC" if rel < SYNC_TOL
            else ("PARTIAL" if rel < PARTIAL_TOL else "FAIL"))


def aggregate(runs):
    by_mu = {}
    for mu, rel in runs:
        if rel is not None and np.isfinite(rel):
            by_mu.setdefault(float(mu), []).append(float(rel))
    if not by_mu:
        return None
    allv = [v for l in by_mu.values() for v in l]
    return dict(best_of_all=float(min(allv)),
                bestmu_median=float(min(float(np.median(l))
                                        for l in by_mu.values())),
                bestmu_worst=float(min(float(max(l)) for l in by_mu.values())),
                n_runs=len(allv),
                spread=float(max(allv) / max(min(allv), 1e-30)))


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    p_star = 2 * K_C + 1
    rows = []
    print(f"nu={NU:g}  K_c={K_C}  p*={p_star}  n={N_SENS}  T={T}  "
          f"grids={GRIDS}  mus={MUS}  inits={INIT_SEEDS}")
    for N in GRIDS:
        flow, w = get_truth(NU, N=N, dt=dt, T_spin=30.0)
        g = flow.g
        print(f"\nN = {N}")
        print(f"   {'p':>3}{'h':>7}{'dK':>10}{'best-all':>11}"
              f"{'bestmu-med':>12}{'bestmu-wst':>12}  verdict(med)")
        for p in PS:
            try:
                ix, iy, meta = stripes_v2(N, N_SENS, p, seed=0)
            except (ValueError, RuntimeError) as exc:
                print(f"   p={p}: skipped ({exc})")
                continue
            h = float(covering_radius(g, ix, iy))
            dK = None
            if WANT_DK:
                try:
                    dK, _ = delta_K(g, ix, iy, K_C, 0.0)
                except MemoryError:
                    dK = None                 # N=256 x 225 modes is ~0.5 GB
            runs, per_run = [], []
            for mu in MUS:
                for s_init in INIT_SEEDS:
                    obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
                    r = trial(flow, w, obs, mu, T=T, init_seed=s_init)
                    v = outcome(r["ts"], r["err"])
                    runs.append((mu, v["plateau_rel"]))
                    per_run.append(dict(mu=mu, init_seed=s_init,
                                        verdict=v["verdict"],
                                        plateau_rel=v["plateau_rel"],
                                        rate=r["rate"],
                                        rate_is_meaningful=v["rate_is_meaningful"],
                                        diverged=r["diverged"],
                                        final=r["final"],
                                        ts=r["ts"], err=r["err"]))
            agg = aggregate(runs)
            head = verdict_of(agg["bestmu_median"]) if agg else "NO_DATA"
            rows.append(dict(nu=NU, N_grid=N, K_c=K_C, p=p, p_star=p_star,
                             n_actual=int(meta["n_actual"]),
                             deficit=int(meta["deficit"]),
                             thickness=int(meta["thickness"]), h=h,
                             delta_K=dK, delta_K_K=K_C, delta_K_floor=0.0,
                             aggregation=agg, verdict=head,
                             verdict_best_of_all=(verdict_of(agg["best_of_all"])
                                                 if agg else "NO_DATA"),
                             verdict_worst=(verdict_of(agg["bestmu_worst"])
                                            if agg else "NO_DATA"),
                             below_threshold=bool(p < p_star),
                             per_run=per_run))
            print(f"   {p:>3}{h:>7.3f}"
                  f"{('n/a' if dK is None else '%+.4f' % dK):>10}"
                  f"{agg['best_of_all']:>11.2e}{agg['bestmu_median']:>12.2e}"
                  f"{agg['bestmu_worst']:>12.2e}  {head}"
                  + ("   <- p*" if p == p_star else ""))

    # the decision, computed rather than eyeballed
    summary = {}
    for N in GRIDS:
        blk = [r for r in rows if r["N_grid"] == N]
        for rule in ("verdict", "verdict_best_of_all"):
            syncs = [r["p"] for r in blk if r[rule] == "SYNC"]
            below = [r["p"] for r in blk
                     if r["below_threshold"] and r[rule] == "SYNC"]
            summary[f"N{N}_{rule}"] = dict(
                first_sync=(min(syncs) if syncs else None),
                offset=(min(syncs) - (2 * K_C + 1) if syncs else None),
                necessity_ok=bool(not below),
                sync_below_p_star=below)
    print("\nsummary (headline rule = bestmu_median):")
    for k, v in sorted(summary.items()):
        print(f"   {k:<28} first SYNC {v['first_sync']}  offset {v['offset']}  "
              f"necessity {'OK' if v['necessity_ok'] else 'BROKEN ' + str(v['sync_below_p_star'])}")

    a = summary.get(f"N128_verdict", {})
    b = summary.get(f"N256_verdict", {})
    if b:
        if not b["necessity_ok"]:
            call = ("C: necessity is resolution-dependent -- restrict every "
                    "threshold claim to nu >= 5e-3 and report this boundary")
        elif b["first_sync"] is not None and a.get("first_sync") is None:
            call = (f"A: sufficiency at nu={NU:g} is resolution-limited -- not "
                    f"reached at N=128, reached at p={b['first_sync']} at N=256")
        elif b["first_sync"] is None:
            call = ("B: no SYNC at either resolution in the swept range -- the "
                    "failure is not a grid artefact; report 'not reached "
                    f"(p <= {max(PS)})'")
        else:
            call = ("D: both resolutions reach SYNC; compare the plateaus and "
                    "report the shift as a resolution sensitivity")
        print(f"\n=> outcome {call}")
    else:
        call = "N=256 not run; rerun with GRIDS=256"
        print(f"\n=> {call}")

    save(rows, dict(nu=NU, K_c=K_C, p_star=p_star, n_sensors=N_SENS, T=T, dt=dt,
                    grids=GRIDS, ps=PS, mus=MUS, init_seeds=INIT_SEEDS,
                    dynamics_floor=FLOOR,
                    generator="nolab.stripes_v2 (deficit spread, rowshift)",
                    headline_aggregation=("min over mu of the median over "
                                          "observer inits"),
                    summary=summary, outcome=call,
                    tail_enstrophy_context=("results/36_resolution_audit part A: "
                                            "N=128 tail 1.7e-3 at this nu, 17x "
                                            "over the 1e-4 criterion; N=256 "
                                            "4.8e-6, OK"),
                    question=("is the nu=2.5e-3 stripe result -- necessity and "
                              "the absence of SYNC -- a property of the flow or "
                              "of the N=128 grid?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
