"""44 -- do the thresholds survive a different truth trajectory?  The oldest
un-addressed item in the review chain, and the second run that can change a claim.

WHAT HAS NEVER BEEN VARIED
-------------------------
Every dynamics script in this repository calls `get_truth(...)` with the default
seed and then starts every trial from the SAME post-spin-up snapshot:

    19, 21-fix, 23, 24, 27, 28, 31, 32, 33, 34, 41   ->  one truth, one snapshot

The sampling that IS done covers the sensor seed and the observer initial
condition -- both of which are properties of the estimator, not of the flow.  So
every threshold in the paper (p* verified 22/22, the offsets, delta_K's ranking
power) is currently measured on a single realisation of the attractor.

That matters here more than it usually would, because the quantity the theory
says sets the required nudging strength is sup_t ||grad omega||_Linf, and
`results/18_theory_diagnostics_fix` measured it over three truth seeds and found
it varies between them.  A referee asking "is this threshold a property of your
trajectory or of the system?" currently has no data to be answered with.

WHAT THIS RUNS
--------------
nu = 5e-3, the best-resolved of the two main viscosities and the one whose
K_c = 5 is stable across N = 128 / 192 / 256 (results/36 part B), so the
threshold p* = m* = 11 is not itself in doubt.  Three independent truth fields
(seeds 0, 1, 2 -- all three are already cached in truths/), and on each of them
the two ladders across their own threshold:

    stripes p = 8, 10, 11, 12       (p* = 11)
    lattice m = 10, 11, 12, 15      (m* = 11)

with mu in {5, 10, 20, 50} and two observer initial conditions, aggregated by the
headline rule (min over mu of the median over inits).

It also closes a bookkeeping gap for free.  `results/32_lattice_aliasing_fix`
ran ONE observer init per mu, so its lattice verdicts at nu = 5e-3 are effectively
under the optimistic minimum rule no matter what the caption says (see
scripts/42 output, rows marked with *).  The lattice rows here carry two inits,
so the nu = 5e-3 lattice offset finally gets measured under the same rule as
everything else.

WHAT TO CONCLUDE
----------------
  * necessity (no SYNC below the threshold) holds on all three truths
    -> the claim is a property of the system, not of the trajectory.  Report
       "verified on three independent truth fields" and the item is closed.
  * necessity holds but the FIRST SYNC moves by one pitch between truths
    -> expected and harmless: report the offset as a range (e.g. "1-2"), not a
       point value, and say it was measured over three truths.
  * necessity fails on some truth
    -> that is a real finding and the threshold claim must be weakened to
       "necessary for the trajectories tested" with the counterexample shown.
       Do not average it away.
  * the plateau at fixed (p, mu) varies by more than the SYNC/PARTIAL window
    (a factor of 15) across truths
    -> the truth seed is then the dominant source of variance, larger than the
       observer init (measured 2-9x), and every plateau in the paper needs an
       error bar over truths rather than over inits.

The gradient scale sup||grad omega||_Linf is recorded per truth seed so the
variation, if any, can be tied to it.

COST
----
A T = 8 trial takes ~47 s at N = 128 here.  The default grid is
8 layouts x 4 mu x 2 inits x 3 truths = 192 trials, about 2.5 h.  All three truth
fields are already in truths/, so there is no spin-up cost.

Lean version, still decisive for necessity (drops mu = 20, keeps one layout below
and one above each threshold):

    PS=10,11 MS=10,12 MUS=5,10,50 python scripts/44_truth_ensemble.py   # ~55 min

Full:

    python scripts/44_truth_ensemble.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save, delta_K, lattice_m
from nolab.observations import covering_radius
from nolab.stripes_v2 import stripes_v2
from nolab.verdict import outcome, SYNC_TOL, PARTIAL_TOL

OUT = ROOT / "results" / "44_truth_ensemble"
NU = float(os.environ.get("NU", 5e-3))
K_C = int(os.environ.get("K_C", 5))
N_GRID = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
T = float(os.environ.get("T", 8.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))
TRUTH_SEEDS = [int(x) for x in os.environ.get("TRUTH_SEEDS", "0,1,2").split(",")]
PS = [int(x) for x in os.environ.get("PS", "8,10,11,12").split(",")]
MS = [int(x) for x in os.environ.get("MS", "10,11,12,15").split(",")]
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]
INIT_SEEDS = [int(x) for x in os.environ.get("INIT_SEEDS", "0,1").split(",")]


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
                n_runs=len(allv),
                spread=float(max(allv) / max(min(allv), 1e-30)))


def gradient_scale(flow, w, T_stat=6.0, every=25):
    """sup_t ||grad omega||_Linf along this truth, the quantity the mu estimate
    is proportional to.  Recorded so any threshold movement can be attributed."""
    g = flow.g
    ww = w.copy()
    vals = []
    for i in range(int(T_stat / flow.dt)):
        ww = flow.step(ww)
        if i % every == 0:
            gx = g.inv(1j * g.kx * ww)
            gy = g.inv(1j * g.ky * ww)
            vals.append(float(np.sqrt(gx ** 2 + gy ** 2).max()))
    return dict(sup_linf=float(np.max(vals)), mean_linf=float(np.mean(vals)))


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    star = 2 * K_C + 1
    rows, grads = [], {}
    print(f"nu={NU:g}  K_c={K_C}  p*=m*={star}  N={N_GRID}  n={N_SENS}  "
          f"truth seeds {TRUTH_SEEDS}  mus={MUS}  inits={INIT_SEEDS}")

    for ts_seed in TRUTH_SEEDS:
        flow, w = get_truth(NU, N=N_GRID, dt=dt, T_spin=30.0, seed=ts_seed)
        g = flow.g
        grads[str(ts_seed)] = gradient_scale(flow, w)
        print(f"\ntruth seed {ts_seed}   sup|grad w| = "
              f"{grads[str(ts_seed)]['sup_linf']:.1f}")
        print(f"   {'family':>8}{'p/m':>5}{'n':>6}{'h':>7}{'dK':>10}"
              f"{'best-all':>11}{'bestmu-med':>12}  verdict(med)")
        jobs = ([("stripes", p) for p in PS] + [("lattice", m) for m in MS])
        for fam, q in jobs:
            if fam == "stripes":
                ix, iy, meta = stripes_v2(N_GRID, N_SENS, q, seed=0)
                n_act = int(meta["n_actual"])
            else:
                ix, iy, meta = lattice_m(N_GRID, q)
                n_act = int(meta["n_actual"])
            h = float(covering_radius(g, ix, iy))
            dK, _ = delta_K(g, ix, iy, K_C, 0.0)
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
            rows.append(dict(nu=NU, N_grid=N_GRID, K_c=K_C, family=fam,
                             q=q, threshold=star, truth_seed=ts_seed,
                             n_actual=n_act, h=h, delta_K=dK, delta_K_K=K_C,
                             aggregation=agg, verdict=head,
                             verdict_best_of_all=(verdict_of(agg["best_of_all"])
                                                 if agg else "NO_DATA"),
                             below_threshold=bool(q < star),
                             per_run=per_run))
            print(f"   {fam:>8}{q:>5}{n_act:>6}{h:>7.3f}{dK:>+10.4f}"
                  f"{agg['best_of_all']:>11.2e}{agg['bestmu_median']:>12.2e}"
                  f"  {head}" + ("   <- threshold" if q == star else ""))

    # per-truth necessity and first SYNC, then the agreement across truths
    per_truth = {}
    for ts_seed in TRUTH_SEEDS:
        for fam in ("stripes", "lattice"):
            blk = [r for r in rows if r["truth_seed"] == ts_seed
                   and r["family"] == fam]
            if not blk:
                continue
            syncs = [r["q"] for r in blk if r["verdict"] == "SYNC"]
            below = [r["q"] for r in blk
                     if r["below_threshold"] and r["verdict"] == "SYNC"]
            per_truth[f"{fam}_truth{ts_seed}"] = dict(
                first_sync=(min(syncs) if syncs else None),
                offset=(min(syncs) - star if syncs else None),
                necessity_ok=bool(not below), sync_below=below,
                ladder_tested=sorted(r["q"] for r in blk),
                verdicts={str(r["q"]): r["verdict"] for r in blk})

    print("\nper truth seed:")
    for k, v in per_truth.items():
        print(f"   {k:<20} first SYNC {v['first_sync']}  offset {v['offset']}  "
              f"necessity {'OK' if v['necessity_ok'] else 'BROKEN ' + str(v['sync_below'])}"
              f"   {v['verdicts']}")

    nec_all = all(v["necessity_ok"] for v in per_truth.values())
    offsets = {}
    for fam in ("stripes", "lattice"):
        offsets[fam] = [per_truth[f"{fam}_truth{s}"]["offset"]
                        for s in TRUTH_SEEDS
                        if f"{fam}_truth{s}" in per_truth]
    offsets_n_measured = {fam: sum(1 for v in offsets[fam] if v is not None)
                          for fam in ("stripes", "lattice")}
    # how big is the truth-to-truth variance compared with the init variance?
    truth_spread = {}
    for fam in ("stripes", "lattice"):
        for q in (PS if fam == "stripes" else MS):
            vals = [r["aggregation"]["bestmu_median"] for r in rows
                    if r["family"] == fam and r["q"] == q
                    and r["aggregation"] is not None]
            if len(vals) >= 2:
                truth_spread[f"{fam}_{q}"] = float(max(vals) / max(min(vals), 1e-30))
    worst = max(truth_spread.values()) if truth_spread else None

    print(f"\nnecessity holds on every truth: {nec_all}")
    print(f"offset across truths: {offsets}")
    print(f"truth-to-truth plateau spread per layout: "
          + ", ".join(f"{k} {v:.1f}x" for k, v in sorted(truth_spread.items())))
    if worst is not None:
        print(f"worst {worst:.1f}x  (observer-init spread measured elsewhere is "
              f"2-9x; the SYNC/PARTIAL window is 15x)")
        if worst > 15:
            print("   -> the truth seed is the dominant source of variance; "
                  "every plateau in the paper needs an error bar over truths")

    save(rows, dict(nu=NU, K_c=K_C, threshold=star, N_grid=N_GRID,
                    n_sensors=N_SENS, T=T, dt=dt, mus=MUS,
                    truth_seeds=TRUTH_SEEDS, init_seeds=INIT_SEEDS,
                    ps=PS, ms=MS, dynamics_floor=FLOOR,
                    stripe_generator="nolab.stripes_v2 (deficit spread)",
                    headline_aggregation=("min over mu of the median over "
                                          "observer inits"),
                    gradient_scales=grads, per_truth=per_truth,
                    necessity_on_every_truth=bool(nec_all),
                    offsets_across_truths={k: list(v) for k, v in offsets.items()},
                    offsets_n_measured=offsets_n_measured,
                    truth_spread=truth_spread, worst_truth_spread=worst,
                    also_closes=("the nu=5e-3 lattice rows of "
                                 "32_lattice_aliasing_fix ran one observer init "
                                 "per mu, so its verdicts were effectively under "
                                 "the minimum rule; these carry two inits"),
                    question=("are the aliasing thresholds a property of the "
                              "system or of the single truth trajectory every "
                              "other script uses?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
