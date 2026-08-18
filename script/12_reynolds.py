"""Scaling law N_c vs Grashof/Reynolds, with no early break (replaces
exp04; P1-7).

v1: `if r['converged']: break` stopped each scan at the first crossing, the
Reynolds dictionary was hard-coded, and the exponent was quoted as
"Re^0.68" / "0.49-0.69" depending on the estimator.  Here every K is run,
the exponents are fitted with a bootstrap CI, and monotonicity is checked.

    python 12_reynolds.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import (KolmogorovFlow, SpectralObs, get_truth, trial, save,
                   first_consistent_K)

OUT = ROOT / "results" / "12_reynolds"
NU_LIST = [float(x) for x in os.environ.get(
    "NU_LIST", "1.5e-2,1.2e-2,1e-2,8e-3,6.5e-3,5e-3,3.75e-3,2.5e-3").split(",")]
MU = float(os.environ.get("MU", 50.0))
T = float(os.environ.get("T", 8.0))
M_SNAP = int(os.environ.get("M_SNAP", 4))
C_STAR = float(os.environ.get("C_STAR", 0.75))


def resolution_ok(flow, wh, frac=1e-4):
    """Fraction of enstrophy in the top third of the resolved band."""
    g = flow.g
    E = np.abs(wh) ** 2
    kmag = np.sqrt(g.k2)
    hi = kmag > (2.0 / 3.0) * (g.N / 2) * 0.8
    tail = float(E[hi].sum() / max(E.sum(), 1e-30))
    return bool(tail < frac), tail


def main():
    cal = None
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    if pcal.exists():
        cal = json.loads(pcal.read_text())
    dt = (cal["recommendations"]["DT"] if cal else 0.004)
    n_grid = int((cal["recommendations"]["N_GRID"] if cal else 128))
    decorr = (cal["recommendations"]["DECORR"] if cal else 8.0)
    t_spin = (cal["recommendations"]["T_SPIN_default"] if cal else 30.0)

    per_nu = {}
    for nu in NU_LIST:
        # low-viscosity states are under-resolved at N=128; escalate to 256.
        N = 256 if nu <= 5e-3 else n_grid
        flow, w = get_truth(nu, N=N, dt=dt, T_spin=t_spin, cfl=True)
        # safety: re-check CFL once more before generating snapshots/trials
        dc = flow.dt_cfl(w)
        if dc < flow.dt:
            flow.set_dt(0.5 * dc)
        ok, tail = resolution_ok(flow, w)
        snaps = [w.copy()]
        for _ in range(M_SNAP - 1):
            for _ in range(int(decorr / flow.dt)):
                w = flow.step(w)
            snaps.append(w.copy())
        kmax = 10 if nu >= 1e-2 else (12 if nu >= 5e-3 else 16)
        K_list = sorted(set([2, 3, 4, 5, 6, 7, 8, 9, kmax]))
        per_K = {}
        for K in K_list:
            rates = []
            for ww in snaps:
                r = trial(flow, ww, SpectralObs(flow.g, K), MU, T=T)
                rates.append(r["rate"])
            per_K[K] = dict(ndof=(2 * K + 1) ** 2,
                            rate_mean=float(np.mean(rates)),
                            rates=rates)
            print(f"  nu={nu:.1e} K={K} mean={np.mean(rates):.4f}")
        K0, ndof0, mono = first_consistent_K(per_K, C_STAR)
        per_nu[f"{nu:.1e}"] = dict(
            N_c=ndof0, K_c=K0, monotonic=bool(mono),
            Re=flow.Re(), grashof=flow.grashof(),
            N_grid=N, resolved=bool(ok), tail_enstrophy=tail,
            per_K={str(k): v for k, v in per_K.items()})
        print(f"  nu={nu:.1e}: N_c={ndof0} (K={K0}, mono={mono}, "
              f"N={N}, resolved={ok}, tail={tail:.1e})")

    # fit log10 N_c = a + b * log10 Grashof (and Re); all points + resolved
    def fit_slope(pairs, label):
        if len(pairs) < 3:
            return None
        xs = np.log10([p[0] for p in pairs])
        ys = np.log10([p[1] for p in pairs])
        a, b = np.polyfit(xs, ys, 1)
        rng = np.random.default_rng(0)
        boots = []
        idx = np.arange(len(pairs))
        for _ in range(4000):
            ii = rng.choice(idx, size=idx.size, replace=True)
            if len(np.unique(ii)) >= 3:
                aa, bb = np.polyfit(xs[ii], ys[ii], 1)
                boots.append(bb)
        ci = [float(np.percentile(boots, 5)),
              float(np.percentile(boots, 95))]
        print(f"  fit N_c ~ {label}^{b:.3f} (n={len(pairs)}) CI={ci}")
        return dict(slope=float(b), intercept=float(a), slope_ci=ci,
                    n=len(pairs))

    fit = {}
    for tag in ("grashof", "Re"):
        allp = [(v[tag], v["N_c"]) for v in per_nu.values()
                if v["N_c"] is not None]
        resp = [(v[tag], v["N_c"]) for v in per_nu.values()
                if v["N_c"] is not None and v["resolved"]]
        fit[tag] = dict(all=fit_slope(allp, tag),
                        resolved=fit_slope(resp, tag))

    meta = dict(nu_list=NU_LIST, mu=MU, T=T, M_snap=M_SNAP, c_star=C_STAR,
                dt=dt, N_grid=n_grid, decorr=decorr, T_spin=t_spin,
                fit=fit,
                question="N_c scaling law, full scans, bootstrap CI")
    save(per_nu, meta, OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
