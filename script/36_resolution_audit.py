"""36 -- resolution audit: the truth fields the main experiments run on.

THE PROBLEM
-----------
`scripts/12_reynolds.py` escalates the grid for low viscosity:

    N = 256 if nu <= 5e-3 else n_grid     # "under-resolved at N=128"

so K_c(nu) -- the quantity every threshold claim is built on -- is measured at
N = 256 for nu = 5e-3 and nu = 2.5e-3.  Every experiment that TESTS those
thresholds (19, 23, 24, 28, 30, 31, 32, 33, 34) runs at N = 128.  That is the
cross-resolution mismatch that `31_stripe_nyquist.py`'s docstring and
`32_lattice_aliasing_fix`'s meta both claim to have eliminated -- and both state
it with the resolutions the wrong way round.

Worse, by the repository's own `resolution_ok` criterion (enstrophy above
0.8 * 2/3 * N/2 must be under 1e-4 of the total), the N = 128 truth fields at
the two main viscosities do not pass:

    nu = 1.5e-2, N = 128   tail 1.0e-6   OK
    nu = 6.5e-3, N = 128   tail 6.6e-5   OK
    nu = 5.0e-3, N = 128   tail 2.6e-4   FAILS by 2.6x   <- the main viscosity
    nu = 2.5e-3, N = 128   tail 1.7e-3   FAILS by 17x
    nu = 5.0e-3, N = 256   tail 1.9e-7   OK

WHAT THIS SCRIPT DECIDES
------------------------
Part A  tail enstrophy of every cached truth, at N = 128, 192, 256, per nu.
        Output: the table above, generated rather than asserted.

Part B  K_c at the SAME resolution as the dynamics runs.  Re-measures the
        SpectralObs rate ladder at N = 128 for nu = 5e-3 and 2.5e-3 and reports
        K_c(c*) for c* in {0.25, 0.5, 0.75, 1.0, 1.25}, so the paper can quote a
        K_c that was measured where it is used.  (Spot-checked at nu = 5e-3,
        N = 128, 3 snapshots: K=4 mean 0.614, K=5 mean 0.794, so the c* = 0.75
        crossing is still K_c = 5 and p* = 11 is unchanged.  nu = 2.5e-3 has not
        been checked and is the one that can move.)

Part C  verdict invariance.  Reruns four decisive layouts at N = 256 and checks
        that the SYNC/PARTIAL/FAIL verdict is unchanged.  If it is, the paper
        can keep N = 128 everywhere and cite this script; if it is not, the main
        table has to move to N = 256.

DECISION GATE
-------------
    Part B changes K_c            -> p*/m* change; rerun 31/32 with the new K_c.
    Part C changes any verdict    -> N = 128 is not adequate; rerun 31/32/33
                                     at N = 256 (or state the limitation and
                                     restrict the claims to nu >= 6.5e-3).
    Neither                       -> keep N = 128, cite this script, and delete
                                     the escalation in 12_reynolds so that K_c
                                     and the dynamics share a grid.

    python scripts/36_resolution_audit.py                    # ~90 min
    PART=A python scripts/36_resolution_audit.py             # ~1 min
    PART=AB NUS=0.0025 python scripts/36_resolution_audit.py # ~40 min
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import (Grid, get_truth, SpectralObs, FixedPointObs, trial, save,
                   stripes_exact, lattice_m, corridor, delta_K)
from nolab.observations import covering_radius
from nolab.verdict import outcome

OUT = ROOT / "results" / "36_resolution_audit"
PART = os.environ.get("PART", "ABC").upper()
NUS = [float(x) for x in os.environ.get("NUS", "0.015,0.005,0.0025").split(",")]
GRIDS = [int(x) for x in os.environ.get("GRIDS", "128,192,256").split(",")]
MU_SPEC = float(os.environ.get("MU_SPEC", 50.0))
T = float(os.environ.get("T", 8.0))
M_SNAP = int(os.environ.get("M_SNAP", 3))
DECORR = float(os.environ.get("DECORR", 4.0))
C_STARS = (0.25, 0.5, 0.75, 1.0, 1.25)
TAIL_TOL = 1e-4


def tail_enstrophy(flow, wh):
    """Same definition as 12_reynolds.resolution_ok, factored out so both the
    audit and the experiment quote one number."""
    g = flow.g
    E = np.abs(wh) ** 2
    kmag = np.sqrt(g.k2)
    hi = kmag > (2.0 / 3.0) * (g.N / 2) * 0.8
    return float(E[hi].sum() / max(E.sum(), 1e-30))


def part_A(dt, t_spin):
    print("\nA. tail enstrophy of the truth field (pass: < %.0e)" % TAIL_TOL)
    print(f"   {'nu':>9}{'N':>6}{'tail':>12}  verdict")
    rows = []
    for nu in NUS:
        for N in GRIDS:
            flow, w = get_truth(nu, N=N, dt=dt, T_spin=t_spin)
            t = tail_enstrophy(flow, w)
            ok = t < TAIL_TOL
            print(f"   {nu:>9.4g}{N:>6}{t:>12.2e}  "
                  f"{'OK' if ok else 'UNDER-RESOLVED'}")
            rows.append(dict(part="A", nu=nu, N=N, tail=t, resolved=bool(ok)))
    return rows


def part_B(dt, t_spin):
    print("\nB. K_c measured AT THE RESOLUTION THE DYNAMICS RUNS ON")
    rows = []
    for nu in NUS:
        for N in GRIDS:
            if N != 128 and nu > 5e-3:
                continue                       # 128 is already resolved there
            flow, w = get_truth(nu, N=N, dt=dt, T_spin=t_spin)
            snaps = [w.copy()]
            for _ in range(M_SNAP - 1):
                for _ in range(int(DECORR / flow.dt)):
                    w = flow.step(w)
                snaps.append(w.copy())
            Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            per_K = {}
            for K in Ks:
                rs = [trial(flow, ww, SpectralObs(flow.g, K), MU_SPEC, T=T)
                      for ww in snaps]
                vs = [outcome(r["ts"], r["err"]) for r in rs]
                per_K[K] = dict(
                    ndof=(2 * K + 1) ** 2,
                    rate_mean=float(np.mean([r["rate"] for r in rs])),
                    rates=[float(r["rate"]) for r in rs],
                    verdicts=[v["verdict"] for v in vs],
                    plateau_rel=[v["plateau_rel"] for v in vs],
                    all_sync=bool(all(v["verdict"] == "SYNC" for v in vs)))
                print(f"   nu={nu:.2e} N={N} K={K:2d} rate={per_K[K]['rate_mean']:.3f} "
                      f"{'/'.join(v[:4] for v in per_K[K]['verdicts'])}")
            kc = {}
            for c in C_STARS:
                hit = next((K for K in Ks if per_K[K]["rate_mean"] >= c), None)
                kc[c] = dict(K_c=hit,
                             N_c=(None if hit is None else (2 * hit + 1) ** 2),
                             p_star=(None if hit is None else 2 * hit + 1))
            # a criterion that does NOT need c*: first K whose runs are all SYNC
            kc["all_sync"] = next((K for K in Ks if per_K[K]["all_sync"]), None)
            print(f"   -> nu={nu:.2e} N={N}: K_c by c* = "
                  + ", ".join(f"{c}:{kc[c]['K_c']}" for c in C_STARS)
                  + f" | first all-SYNC K = {kc['all_sync']}")
            rows.append(dict(part="B", nu=nu, N=N, per_K=per_K,
                             K_c_by_cstar={str(k): v for k, v in kc.items()
                                           if k != "all_sync"},
                             K_c_all_sync=kc["all_sync"]))
    return rows


DECISIVE = [
    # (label, builder)  -- the four rows the thresholds actually rest on
    ("stripes_p10", lambda N, n: stripes_exact(N, n, 10, seed=0, jitter=True,
                                               jitter_mode="rowshift",
                                               strict_count=True)[:2]),
    ("stripes_p12", lambda N, n: stripes_exact(N, n, 12, seed=0, jitter=True,
                                               jitter_mode="rowshift",
                                               strict_count=True)[:2]),
    ("lattice_m12", lambda N, n: lattice_m(N, 12)[:2]),
    ("corridor_gw0.8", lambda N, n: corridor(N, n, 0.8, seed=0)[:2]),
]


def part_C(dt, t_spin, nu=5e-3, n_sens=784, mus=(5.0, 10.0, 20.0, 50.0)):
    print(f"\nC. verdict invariance in N at nu={nu:g} "
          f"(the claim is that N=128 and N=256 agree)")
    rows = []
    for label, build in DECISIVE:
        for N in (128, 256):
            flow, w = get_truth(nu, N=N, dt=dt, T_spin=t_spin)
            g = flow.g
            ix, iy = build(N, n_sens)
            d0, _ = delta_K(g, ix, iy, 5, 0.0)
            best, bv = None, "NO_DATA"
            for mu in mus:
                r = trial(flow, w, FixedPointObs(g, ix, iy, denom_floor=0.25),
                          mu, T=T)
                v = outcome(r["ts"], r["err"])
                if v["plateau_rel"] is not None and (best is None
                                                     or v["plateau_rel"] < best):
                    best, bv = v["plateau_rel"], v["verdict"]
            print(f"   {label:16} N={N:4}  n={len(ix):4}  h={covering_radius(g, ix, iy):.3f} "
                  f"dK={d0:+.4f}  plateau={best:.2e}  {bv}")
            rows.append(dict(part="C", nu=nu, N=N, layout=label,
                             n_actual=int(len(ix)), delta_K_floor0=d0,
                             best_plateau_rel=best, best_verdict=bv))
    # agreement check
    by = {}
    for r in rows:
        by.setdefault(r["layout"], {})[r["N"]] = r["best_verdict"]
    agree = {k: (v.get(128) == v.get(256)) for k, v in by.items()}
    print("   verdict agreement 128 vs 256: "
          + ", ".join(f"{k}:{'same' if v else 'DIFFERENT'}"
                      for k, v in agree.items()))
    return rows, agree


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    cal = json.loads(pcal.read_text()) if pcal.exists() else None
    dt = cal["recommendations"]["DT"] if cal else 0.002
    t_spin = cal["recommendations"]["T_SPIN_default"] if cal else 30.0
    rows, agree = [], None
    if "A" in PART:
        rows += part_A(dt, t_spin)
    if "B" in PART:
        rows += part_B(dt, t_spin)
    if "C" in PART:
        rc, agree = part_C(dt, t_spin)
        rows += rc
    save(rows, dict(nus=NUS, grids=GRIDS, T=T, dt=dt, mu_spectral=MU_SPEC,
                    M_snap=M_SNAP, decorr=DECORR, tail_tol=TAIL_TOL,
                    part=PART, verdict_agreement=agree,
                    question="is N=128 adequate at nu<=5e-3, and is K_c "
                             "measured on the grid it is used on?"),
         str(OUT))
    print(f"\nwritten to {OUT/'results.json'}")


if __name__ == "__main__":
    main()
