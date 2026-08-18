"""42 -- one aggregation rule for every family, and a Fig 1 dataset built only
from layouts that still exist.  No dynamics, ~10 s.

TWO THINGS THIS FIXES, BOTH INTRODUCED BY THE 40/41 REWORK ITSELF
-----------------------------------------------------------------
1.  THE TWO FAMILIES ARE NOW AGGREGATED BY DIFFERENT RULES.  `scripts/41` adopted
    the headline rule "min over mu of the median over (sensor seed, observer
    init)" and reports the stripe first-SYNC under it.  The lattice ladder was
    NOT re-aggregated: `results/39_lattice_offset_fill` still labels each m by
    the minimum over its 8 runs.  That single difference is what moved the
    lattice offset from 6 to 5, because m = 20 is

        min over 8 runs      4.24e-03  -> SYNC
        min over mu of the
        median over inits    1.30e-02  -> PARTIAL

    i.e. under the rule the paper says it uses, the first lattice SYNC at
    nu = 2.5e-3 is m = 21 and the offset is 6, not 5.  `39`'s own
    `meta.merged_ladder` shows the collision directly: it contains both
    `[20, "PARTIAL"]` (from 32) and `[20, "SYNC"]` (from 39).

    A second, quieter version of the same problem: the lattice rows for
    nu = 1.5e-2 and 5e-3 come from `results/32_lattice_aliasing_fix`, which ran
    ONE observer initial condition per mu.  With no seed dimension the median
    rule degenerates to the minimum, so those two offsets (2 and 4) are measured
    under the optimistic rule no matter what the caption says.  This script
    marks them `single_replicate` so the table can say so, and the fix is to
    rerun the four rows around each threshold (m = 11, 12 at nu = 1.5e-2;
    m = 15, 16 at nu = 5e-3) with INIT_SEEDS=0,1.

2.  THE FIG 1 DATASET STILL CONTAINS LAYOUTS THAT NO LONGER EXIST.  `exp30` and
    `exp33` built their stripe rows with the old per-point jitter, so the same
    label denotes a different layout there and in `exp41`:

        stripes_p11   exp30/33: n=676, h=0.310, delta_K=+0.2055
                      exp41   : n=784, h=0.299, delta_K=+0.2610

    Plotting Fig 1 with x from exp30/33 and verdicts from exp41 puts the stripe
    points at the wrong abscissa, and the tau / AUC in PAPER_DATA_APPENDIX are
    computed on that mixture.  This script rebuilds the dataset from exp41's
    stripes plus the NON-stripe rows of exp19 and exp33, restricted to
    nu = 5e-3 so that every delta_K is on the same band (K = 5) -- pooling
    K = 4, 5 and 7 into one axis, as the current draft implicitly does, is not
    meaningful because delta_K is band-dependent.

WHAT IT ALSO SETTLES
--------------------
Whether the stripe family can separate h from delta_K at all.  It cannot, and
that has to be said out loud: fixing the truncation hole made h monotone in p,
so within the fixed family h and delta_K are collinear and h scores as well or
better (Kendall tau against the plateau, per viscosity: h 0.810 / 1.000 / 0.786
versus delta_K 0.810 / 0.905 / 0.786).  The stripe family's job is to fix the
sensor count and show the threshold tracking 2*K_c+1 across viscosity; the
refutation of h has to come from the other two arguments, both of which survive:

  * the same layout, h unchanged, gives different verdicts at different nu
    (stripes p = 10, h = 0.310: SYNC at 1.5e-2, PARTIAL at 5e-3, FAIL at 2.5e-3;
    p = 12, h = 0.264: SYNC, SYNC, FAIL).  A purely geometric quantity cannot
    explain a nu-dependent outcome;
  * the cross-family rows of exp19/exp33, where h ties and the outcome does not.

    python scripts/42_unify_aggregation.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab.harness import save
from nolab.ranking import predictor_quality
from nolab.verdict import SYNC_TOL, PARTIAL_TOL, E_REF_DEFAULT

OUT = ROOT / "results" / "42_unify_aggregation"
RES = ROOT / "results"
FIG1_NU = float(os.environ.get("FIG1_NU", 5e-3))
FIG1_K = int(os.environ.get("FIG1_K", 5))


def verdict_of(rel):
    if rel is None:
        return "NO_DATA"
    return ("SYNC" if rel < SYNC_TOL
            else ("PARTIAL" if rel < PARTIAL_TOL else "FAIL"))


def aggregate(runs):
    """runs = [(mu, plateau_rel), ...].  Returns both rules plus replication."""
    by_mu = {}
    for mu, rel in runs:
        if rel is not None and np.isfinite(rel):
            by_mu.setdefault(float(mu), []).append(float(rel))
    if not by_mu:
        return None
    reps = min(len(v) for v in by_mu.values())
    allv = [v for l in by_mu.values() for v in l]
    return dict(best_of_all=float(min(allv)),
                bestmu_median=float(min(float(np.median(l))
                                        for l in by_mu.values())),
                replicates_per_mu=int(reps),
                single_replicate=bool(reps < 2),
                n_runs=len(allv),
                spread=float(max(allv) / max(min(allv), 1e-30)))


# ------------------------------------------------------------------- lattice

def lattice_ladder():
    """Merge 32 + 39 under both rules, preferring the more replicated source."""
    rows = {}
    p32 = RES / "32_lattice_aliasing_fix" / "results.json"
    if p32.exists():
        for r in json.loads(p32.read_text())["rows"]:
            runs = [(x["mu"], x.get("plateau_rel")) for x in r["per_mu"]]
            agg = aggregate(runs)
            rows[(r["nu"], r["m"])] = dict(
                nu=r["nu"], m=r["m"], m_star=r["m_star"], K_c=r["K_c"],
                n=r["n"], h=r["h"], delta_K=r.get("delta_K_floor0"),
                source="32", **(agg or {}))
    p39 = RES / "39_lattice_offset_fill" / "results.json"
    if p39.exists():
        d = json.loads(p39.read_text())
        nu, m_star, K_c = d["meta"]["nu"], d["meta"]["m_star"], d["meta"]["K_c"]
        for r in d["rows"]:
            runs = [(x["mu"], x.get("plateau_rel")) for x in r["per_run"]]
            agg = aggregate(runs)
            key = (nu, r["m"])
            prev = rows.get(key)
            rec = dict(nu=nu, m=r["m"], m_star=m_star, K_c=K_c, n=r["n"],
                       h=r["h"], delta_K=r.get("delta_K"), source="39",
                       **(agg or {}))
            if prev is not None:
                rec["source"] = "39 (supersedes 32: more replication)"
                rec["superseded_32"] = dict(
                    best_of_all=prev["best_of_all"],
                    bestmu_median=prev["bestmu_median"],
                    n_runs=prev["n_runs"])
            rows[key] = rec
    out = sorted(rows.values(), key=lambda r: (-r["nu"], r["m"]))
    for r in out:
        r["verdict_best_of_all"] = verdict_of(r.get("best_of_all"))
        r["verdict_bestmu_median"] = verdict_of(r.get("bestmu_median"))
    return out


# --------------------------------------------------------------------- stripe

def stripe_ladder():
    p = RES / "41_stripe_nyquist_v3" / "results.json"
    if not p.exists():
        return []
    out = []
    for r in json.loads(p.read_text())["rows"]:
        a = r["aggregation"]
        out.append(dict(nu=r["nu"], p=r["p"], p_star=r["p_star"],
                        K_c=r["K_c"], n=r["n_actual"], h=r["h"],
                        delta_K=r["delta_K"], deficit=r["deficit"],
                        best_of_all=a["best_of_all"],
                        bestmu_median=a["bestmu_median"],
                        n_runs=len(r["per_run"]),
                        replicates_per_mu=len(r["per_run"]) // max(
                            len({x["mu"] for x in r["per_run"]}), 1),
                        single_replicate=False,
                        verdict_best_of_all=verdict_of(a["best_of_all"]),
                        verdict_bestmu_median=verdict_of(a["bestmu_median"]),
                        source="41"))
    return out


def first_sync(rows, key, rule):
    got = {}
    for r in rows:
        if r[f"verdict_{rule}"] == "SYNC":
            k = str(r["nu"])
            got[k] = min(got.get(k, 10 ** 9), r[key])
    return got


# ----------------------------------------------------------------- Fig 1 set

def fig1_rows():
    """nu = FIG1_NU only, delta_K all on band K = FIG1_K, no stale stripe rows."""
    rows = []
    p19 = RES / "19_anisotropy_ablation_fix" / "results.json"
    if p19.exists():
        d = json.loads(p19.read_text())
        for r in d["rows"]:
            if r["family"] == "stripes_exact":
                continue                      # superseded by exp41
            runs = [(x["mu"], (None if x.get("final") is None
                               else x["final"] / E_REF_DEFAULT))
                    for x in r["per_mu"]]
            agg = aggregate(runs)
            if agg is None or r.get("K") != FIG1_K:
                continue
            lab = r["family"] + (f"_n{r['n']}" if r["family"] == "uniform"
                                 else (f"_gw{r.get('gap_width')}"
                                       if r["family"] == "corridor" else ""))
            rows.append(dict(label=lab, family=r["family"], seed=r.get("seed"),
                             h=r["h"], delta_K_floor0=r["delta_K_floor0"],
                             best_plateau_rel=agg["best_of_all"],
                             best_verdict=verdict_of(agg["best_of_all"]),
                             source="19 (final only)", curve_backed=False,
                             resolution_flag=bool(r["family"] == "corridor")))
    p33 = RES / "33_boundary_curves" / "results.json"
    if p33.exists():
        for r in json.loads(p33.read_text())["rows"]:
            if str(r["layout"]).startswith("stripes"):
                continue                      # superseded by exp41
            rows.append(dict(label=r["layout"], family=r.get("kind"),
                             seed=r.get("sensor_seed"), h=r["h"],
                             delta_K_floor0=r["delta_K_floor0"],
                             best_plateau_rel=r["best_plateau_rel"],
                             best_verdict=r["best_verdict"],
                             source="33 (curves)", curve_backed=True,
                             resolution_flag=bool(
                                 str(r["layout"]).startswith("corridor"))))
    for r in stripe_ladder():
        if r["nu"] != FIG1_NU:
            continue
        rows.append(dict(label=f"stripes_p{r['p']}", family="stripes_v2",
                         seed=0, h=r["h"], delta_K_floor0=r["delta_K"],
                         best_plateau_rel=r["best_of_all"],
                         best_verdict=r["verdict_best_of_all"],
                         source="41 (curves, deficit spread)",
                         curve_backed=True, resolution_flag=False))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)

    lat = lattice_ladder()
    print("LATTICE ladder under both rules "
          "(* = only one observer init per mu, so the median rule cannot apply)")
    print(f"   {'nu':>8}{'m':>4}{'m*':>4}{'n':>6}{'runs':>6}"
          f"{'best-all':>11}{'bestmu-med':>12}  {'v(all)':<8}{'v(med)':<8} src")
    for r in lat:
        star = "*" if r.get("single_replicate") else " "
        print(f"   {r['nu']:>8.4g}{r['m']:>4}{r['m_star']:>4}{r['n']:>6}"
              f"{r['n_runs']:>5}{star}{r['best_of_all']:>11.2e}"
              f"{r['bestmu_median']:>12.2e}  "
              f"{r['verdict_best_of_all']:<8}{r['verdict_bestmu_median']:<8}"
              f" {r['source']}")
    fs_all = first_sync(lat, "m", "best_of_all")
    fs_med = first_sync(lat, "m", "bestmu_median")
    mstar = {str(r["nu"]): r["m_star"] for r in lat}
    off_all = {k: v - mstar[k] for k, v in fs_all.items()}
    off_med = {k: v - mstar[k] for k, v in fs_med.items()}
    print(f"   first SYNC  best-of-all {fs_all}   offsets {off_all}")
    print(f"   first SYNC  bestmu-med  {fs_med}   offsets {off_med}")
    print("   -> the paper's headline rule is bestmu-median (scripts/41), so the "
          "lattice offsets to quote are the second line.")

    st = stripe_ladder()
    sfs_all = first_sync(st, "p", "best_of_all")
    sfs_med = first_sync(st, "p", "bestmu_median")
    pstar = {str(r["nu"]): r["p_star"] for r in st}
    print("\nSTRIPE ladder (exp41)")
    print(f"   first SYNC  best-of-all {sfs_all}   offsets "
          f"{ {k: v - pstar[k] for k, v in sfs_all.items()} }")
    print(f"   first SYNC  bestmu-med  {sfs_med}   offsets "
          f"{ {k: v - pstar[k] for k, v in sfs_med.items()} }")
    missing = [k for k in pstar if k not in sfs_med]
    if missing:
        print(f"   no SYNC in the swept range under the headline rule at nu = "
              f"{missing} -- report 'not reached (p <= max tested)', not a number")

    f1 = fig1_rows()
    print(f"\nFIG 1 dataset: nu = {FIG1_NU:g}, delta_K on band K = {FIG1_K}, "
          f"{len(f1)} rows "
          f"({sum(r['curve_backed'] for r in f1)} curve-backed, "
          f"{sum(r['resolution_flag'] for r in f1)} flagged "
          f"resolution-unstable)")
    qd = predictor_quality(f1, "delta_K_floor0", +1)
    qh = predictor_quality(f1, "h", -1)
    for q in (qd, qh):
        print(f"   {q['predictor']:<16} n={q['n']:>3}  tau={q['tau']:+.3f}  "
              f"auc={q['auc']:.3f}  inversions(plain)={q['inversions_tol1']}"
              f"/{q['pairs']}  inversions(3x)={q['inversions_tol3']}")
    noflag = [r for r in f1 if not r["resolution_flag"]]
    print("   with the resolution-unstable corridor rows dropped:")
    for pred, sgn in (("delta_K_floor0", +1), ("h", -1)):
        q = predictor_quality(noflag, pred, sgn)
        print(f"   {pred:<16} n={q['n']:>3}  tau={q['tau']:+.3f}  "
              f"auc={q['auc']:.3f}")

    # can the stripe family separate h from delta_K?  (it cannot; say so)
    sep = {}
    for nu in sorted({r["nu"] for r in st}, reverse=True):
        sub = [dict(delta_K_floor0=r["delta_K"], h=r["h"],
                    best_plateau_rel=r["bestmu_median"],
                    best_verdict=r["verdict_bestmu_median"]) for r in st
               if r["nu"] == nu]
        sep[str(nu)] = dict(
            tau_delta_K=predictor_quality(sub, "delta_K_floor0", +1)["tau"],
            tau_h=predictor_quality(sub, "h", -1)["tau"])
    print("\nwithin the FIXED stripe family, per nu (h is now monotone in p, so "
          "the family cannot separate the two):")
    for k, v in sep.items():
        print(f"   nu={float(k):.4g}  tau(delta_K)={v['tau_delta_K']:+.3f}  "
              f"tau(h)={v['tau_h']:+.3f}")

    save(dict(lattice=lat, stripe=st, fig1=f1),
         dict(rule_headline=("min over mu of the median over (sensor seed, "
                            "observer init)"),
              lattice_first_sync=dict(best_of_all=fs_all, bestmu_median=fs_med),
              lattice_offsets=dict(best_of_all=off_all, bestmu_median=off_med),
              stripe_first_sync=dict(best_of_all=sfs_all, bestmu_median=sfs_med),
              stripe_nu_without_sync=missing,
              fig1_nu=FIG1_NU, fig1_K=FIG1_K,
              fig1_quality=dict(delta_K=qd, h=qh),
              stripe_family_cannot_separate=sep,
              excluded=("stripe rows of exp19/30/33 (old per-point jitter, "
                        "n = 611-676, superseded by exp41)"),
              question=("one aggregation rule for both families, and a Fig 1 "
                        "dataset containing only layouts that still exist")),
         OUT)
    print("\nwrote", OUT / "results.json")


if __name__ == "__main__":
    main()
