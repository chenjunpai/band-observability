"""32 -- redo of scripts/25 with the two defects it shipped with.

WHAT WAS WRONG WITH 25
----------------------
1.  THE WRONG K_c AT THE LOWEST VISCOSITY.  `25_lattice_aliasing.py` hard-codes
    K_c = 8 for nu = 2.5e-3, giving m* = 17.  That value came from the old
    script 12, whose K grid was {2,3,4,5,6,8,10} -- 7 was never tested, so
    N_c had to land on 289 (K = 8).  After the refill, `results/12_reynolds`
    gives N_c = 225, i.e. K_c = 7 and m* = 15.  Every number in the nu = 2.5e-3
    block of 25 -- the prediction column AND the delta_K column, which was
    evaluated at K = 8 -- is therefore mislabelled.

2.  A RESOLUTION MISMATCH.  25 runs nu = 2.5e-3 at N = 256 while K_c came from
    an N = 128 measurement.  Either both should be 128 or script 12 should be
    rerun at 256; mixing them means the predicted threshold and the tested
    threshold are not derived from the same flow.  This script runs at N = 128,
    matching script 12; set N_GRID=256 to check the resolution sensitivity, but
    then quote K_c from an N = 256 rerun of 12, not from the current one.

3.  (shared with everything else) It stored only `final`, so the verdicts had to
    be reconstructed from a single sample.  Here the full curves are kept and
    `nolab.verdict.outcome` is used.

WHAT 25 ACTUALLY SHOWED, REREAD WITH THE PLATEAU CRITERION
----------------------------------------------------------
    nu       K_c  m*   m < m*            plateau at m*, m*+1, ...   first SYNC
    1.5e-2    4    9   6,7,8 all FAIL    4.9e-1, 7.5e-2, 2.0e-4     m = 12
    5.0e-3    5   11   8,9,10 all FAIL   3.2e-1, 1.8e-1, 4.3e-3     m = 14
    2.5e-3    7*  15   12,14 all FAIL    (m=16) 9.7e-2 ... 3.3e-3   m = 20

So m >= 2 K_c + 1 is NECESSARY -- nothing below it synchronises, 9/9 across
three viscosities, and the threshold moves 9 -> 11 -> 15 with nu while the
coverage radius at the threshold does not (0.49, 0.42, 0.28) -- but it is NOT
SUFFICIENT: at m = m* the operator is only barely non-singular (delta_K ~ 1e-4)
and full synchronisation arrives about three pitches higher.  The paper must
claim necessity, and report the offset as the measured gap between the
parameter-free bound and the empirical threshold.

This script fills in the m values 25 skipped around each threshold so that the
offset can be quoted as a number rather than as "about three".

    python scripts/32_lattice_aliasing_fix.py        # ~60 min at N = 128
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
from nolab import lattice_m, delta_K, band_coupling
from nolab.verdict import outcome, best_verdict

OUT = ROOT / "results" / "32_lattice_aliasing_fix"
N_GRID = int(os.environ.get("N_GRID", 128))
T = float(os.environ.get("T", 8.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]

# nu -> (K_c from the REFILLED results/12_reynolds, m values to sweep)
# nu 1.5e-2: N_c = 81  -> K_c = 4 -> m* = 9
# nu 5.0e-3: N_c = 121 -> K_c = 5 -> m* = 11
# nu 2.5e-3: N_c = 225 -> K_c = 7 -> m* = 15      (was 8 / 17 in script 25)
CASES = {
    1.5e-2: (4, [7, 8, 9, 10, 11, 12, 13]),
    5.0e-3: (5, [9, 10, 11, 12, 13, 14, 15]),
    2.5e-3: (7, [13, 14, 15, 16, 17, 18, 20]),
}


def read_Kc():
    """Cross-check the hard-coded K_c against results/12_reynolds, so this
    script cannot silently drift out of date the way 25 did."""
    p = ROOT / "results" / "12_reynolds" / "results.json"
    if not p.exists():
        return {}
    rows = json.loads(p.read_text())["rows"]
    out = {}
    for k, rec in rows.items():
        if rec.get("K_c") is not None:
            out[float(k)] = int(rec["K_c"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)

    measured = read_Kc()
    for nu, (K_c, _) in CASES.items():
        got = measured.get(nu)
        if got is not None and got != K_c:
            print(f"!! K_c mismatch at nu={nu:g}: this script uses {K_c}, "
                  f"results/12_reynolds says {got}.  Fix CASES before trusting "
                  f"the prediction column.")
        elif got is not None:
            print(f"   K_c({nu:g}) = {K_c} confirmed against results/12_reynolds")

    rows = []
    for nu, (K_c, ms) in sorted(CASES.items(), reverse=True):
        flow, w = get_truth(nu, N=N_GRID, dt=dt, T_spin=30.0)
        g = flow.g
        m_star = 2 * K_c + 1
        print(f"\nnu={nu:g}  K_c={K_c}  m* = {m_star}  N = {N_GRID}")
        for m in ms:
            ix, iy, meta = lattice_m(g.N, m)
            h = float(covering_radius(g, ix, iy))
            dK0, dmax = delta_K(g, ix, iy, K_c, 0.0)
            dKf, _ = delta_K(g, ix, iy, K_c, FLOOR)
            leak = band_coupling(g, ix, iy, K_c, FLOOR)["leak_fraction"]
            per_run, verdicts = [], []
            for mu in MUS:
                obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
                r = trial(flow, w, obs, mu, T=T)
                v = outcome(r["ts"], r["err"])
                verdicts.append(v["verdict"])
                per_run.append(dict(mu=mu, verdict=v["verdict"],
                                    plateau_rel=v["plateau_rel"],
                                    rate=r["rate"],
                                    rate_is_meaningful=v["rate_is_meaningful"],
                                    converged_old=r["converged"],
                                    status_old=r["status"],
                                    diverged=r["diverged"], final=r["final"],
                                    ts=r["ts"], err=r["err"]))
            bv = best_verdict(verdicts)
            rels = [x["plateau_rel"] for x in per_run
                    if x["plateau_rel"] is not None]
            best = float(min(rels)) if rels else None
            necessity_ok = (m >= m_star) or (bv in ("FAIL", "DIVERGED"))
            rows.append(dict(nu=nu, N_grid=N_GRID, K_c=K_c, m=m,
                             m_star=m_star, n=int(meta["n_actual"]),
                             nyquist_K=int(meta["nyquist_K"]), h=h,
                             delta_K_floor0=dK0, delta_K_floor=dKf,
                             delta_K_max_eig=dmax, leak_fraction=leak,
                             best_verdict=bv, best_plateau_rel=best,
                             above_threshold=bool(m >= m_star),
                             necessity_ok=bool(necessity_ok),
                             per_mu=per_run))
            mark = " " if necessity_ok else "!!"
            star = "*" if m == m_star else " "
            print(f"  {mark}m={m:3d}{star} n={meta['n_actual']:4d} h={h:.3f} "
                  f"dK={dK0:+.5f} plateau="
                  f"{'n/a' if best is None else '%.1e' % best} {bv}")

        block = [r for r in rows if r["nu"] == nu]
        syncs = [r["m"] for r in block if r["best_verdict"] == "SYNC"]
        if syncs:
            print(f"   first SYNC at m = {min(syncs)}; "
                  f"offset above m* = {min(syncs) - m_star}")
        else:
            print(f"   no SYNC in the swept range -- extend `ms` for nu={nu:g}")

    below = [r for r in rows if not r["above_threshold"]]
    print(f"\nnecessity (nothing synchronises below m*): "
          f"{sum(r['necessity_ok'] for r in below)}/{len(below)}")
    offsets = {}
    for nu in CASES:
        block = [r for r in rows if r["nu"] == nu
                 and r["best_verdict"] == "SYNC"]
        if block:
            offsets[str(nu)] = min(r["m"] for r in block) - (2 * CASES[nu][0] + 1)
    print("offset of the empirical threshold above m* per nu:", offsets)

    save(rows, dict(N_grid=N_GRID, T=T, dt=dt, mus=MUS, floor=FLOOR,
                    cases={str(k): v for k, v in CASES.items()},
                    K_c_source="results/12_reynolds (K grid refilled with 7, 9)",
                    verdict_criterion="nolab.verdict.outcome (plateau)",
                    offsets_above_m_star=offsets,
                    supersedes=("25_lattice_aliasing: wrong K_c at nu=2.5e-3 "
                                "(8 instead of 7), N=256 tested against an "
                                "N=128 K_c, and verdicts from a single final "
                                "sample"),
                    claim=("m >= 2K_c+1 is necessary for synchronisation and "
                           "moves with nu; it is not sufficient, and the "
                           "measured offset is reported rather than hidden")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
