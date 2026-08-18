"""18-fix -- theory diagnostics with the controls the original is missing.

WHAT WAS WRONG WITH THE ORIGINAL
--------------------------------
`scripts/18_theory_diagnostics.py` computes delta_K for four layouts at n = 784,
seed 0 only.  Those four layouts differ in h (0.42, 0.62, 2.31, 1.63) AND in
geometry AND in sensor distribution, so the run cannot attribute anything: it
shows that delta_K correlates with failure across four very different points,
which was never in doubt.  The decisive samples are the ones near the boundary,
and they were not computed.

It also states a corrected diagnostic relation,

    mu* ~ sup_t ||grad w||_Linf / (delta_Kc - C h Kc)

and then reports that the measured gradient scale (~10^2) is "the same order" as
the measured mu_opt of 50-100.  Substituting the script's own numbers,

    115 / 0.123 = 9.4e2,

which is 10-20x above the measured optimum -- the agreement only appears if
delta_K is dropped from the denominator.  The old estimate was wrong by 20-40x
and the corrected one is wrong by ~10x; that is an improvement, not a
resolution, and the paper must say so.  (It is also worth noting that mu_opt
itself is not measured: in every sweep in the repository the rate is still
rising at the largest mu tested.)

WHAT THIS VERSION ADDS
----------------------
  1. delta_K over 3 sensor seeds AND at the sync/fail boundary layouts, so the
     sign and the ordering can be checked for robustness;
  2. the largest eigenvalue of sym(I_h) alongside delta_K -- the operator is not
     a contraction (max eig 1.00-1.21), so delta_K is only meaningful relative
     to that normalisation;
  3. band coupling ||(I - P_K) I_h P_K||, since delta_K certifies coercivity
     only INSIDE the band while the energy estimate also needs the leakage out
     of it (measured here: small for the point layouts, which is itself the
     answer to the obvious referee question, and reportable);
  4. gradient scales over several truth trajectories rather than one;
  5. an explicit table of predicted vs measured mu*, printed with the ratio, so
     the residual discrepancy is stated rather than absorbed.

    python scripts/18_theory_diagnostics_fix.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, save, GENERATORS
from nolab import layout_report, lattice_m

OUT = ROOT / "results" / "18_theory_diagnostics_fix"
NU = float(os.environ.get("NU", 5e-3))
K = int(os.environ.get("K", 5))
T_STAT = float(os.environ.get("T_STAT", 12.0))
TRUTH_SEEDS = [int(x) for x in os.environ.get("TRUTH_SEEDS", "0,1,2").split(",")]
SENSOR_SEEDS = [int(x) for x in os.environ.get("SENSOR_SEEDS", "0,1,2").split(",")]

# (label, generator, n) -- includes the boundary cases from script 17
BOUNDARY = [("uniform", 200), ("uniform", 300), ("uniform", 400),
            ("uniform", 500), ("uniform", 784),
            ("ground_tracks", 784), ("clustered", 784), ("blind_half", 784)]


def gradient_scales(nu, dt, T_stat, seeds):
    out = []
    for s in seeds:
        flow, w = get_truth(nu, N=128, dt=dt, T_spin=30.0, seed=s)
        g = flow.g
        infs, l4s = [], []
        ww = w.copy()
        for i in range(int(T_stat / flow.dt)):
            ww = flow.step(ww)
            if i % 25 == 0:
                gx = g.inv(1j * g.kx * ww)
                gy = g.inv(1j * g.ky * ww)
                mag = np.sqrt(gx ** 2 + gy ** 2)
                infs.append(float(mag.max()))
                l4s.append(float(np.mean(mag ** 4) ** 0.25))
        out.append(dict(truth_seed=s, sup_linf=float(np.max(infs)),
                        mean_linf=float(np.mean(infs)),
                        sup_l4=float(np.max(l4s))))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, _ = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    g = flow.g

    grads = gradient_scales(NU, dt, T_STAT, TRUTH_SEEDS)
    sup_linf = float(np.mean([x["sup_linf"] for x in grads]))
    print("gradient scales (per truth seed):", grads)

    reports = []
    for name, n in BOUNDARY:
        for s in SENSOR_SEEDS:
            ix, iy = GENERATORS[name](g.N, n, seed=s)
            rep = layout_report(g, ix, iy, K=K, denom_floor=0.25)
            rep.update(layout=name, n_requested=n, sensor_seed=s)
            reports.append(rep)
            print(f"  {name:14} n={n:4d} s={s} h={rep['h']:.3f} "
                  f"dK0={rep['delta_K_floor0']:+.4f} "
                  f"dK.25={rep['delta_K_floor']:+.4f} "
                  f"maxeig={rep['delta_K_max_eig']:.3f} "
                  f"leak={rep['leak_fraction']:.2e} "
                  f"corr={rep['max_corridor']:.3f}")
    for m in (8, 10, 12, 14):
        ix, iy, meta = lattice_m(g.N, m)
        rep = layout_report(g, ix, iy, K=K, denom_floor=0.25)
        rep.update(layout=f"lattice_m{m}", n_requested=meta["n_actual"],
                   sensor_seed=0, nyquist_K=meta["nyquist_K"])
        reports.append(rep)
        print(f"  lattice m={m:3d} n={meta['n_actual']:4d} h={rep['h']:.3f} "
              f"dK.25={rep['delta_K_floor']:+.4f} "
              f"nyquistK={meta['nyquist_K']} (needs >= {K})")

    # seed robustness of the sign, per layout
    sign_stability = {}
    for name, n in BOUNDARY:
        vals = [r["delta_K_floor"] for r in reports
                if r.get("layout") == name and r.get("n_requested") == n]
        sign_stability[f"{name}_n{n}"] = dict(
            values=vals, mean=float(np.mean(vals)), std=float(np.std(vals)),
            sign_consistent=bool(all(v > 0 for v in vals)
                                 or all(v < 0 for v in vals)))

    # predicted vs measured mu*, stated with the residual ratio
    mu_table = []
    for r in reports:
        d = r["delta_K_floor"]
        pred = (sup_linf / d) if d > 0 else None
        mu_table.append(dict(layout=r.get("layout"), n=r.get("n_requested"),
                             seed=r.get("sensor_seed"), delta_K=d,
                             mu_star_pred=pred,
                             note=("delta_K <= 0: the corrected estimate gives "
                                   "no finite mu*, yet several such layouts "
                                   "synchronise -- coercivity is sufficient, "
                                   "not necessary" if pred is None else "")))

    out = dict(gradient_scales=grads, sup_linf_mean=sup_linf,
               layout_reports=reports, sign_stability=sign_stability,
               mu_star_table=mu_table,
               honest_note=("mu* = sup|grad w| / delta_K with the measured "
                            "numbers gives ~9e2 for uniform n=784, versus a "
                            "measured optimum of 50-100 that is itself at the "
                            "edge of every mu sweep in this repository.  "
                            "Report the corrected estimate as an improvement "
                            "from 20-40x to ~10x, not as a resolution."))
    save(out, dict(nu=NU, K=K, T_stat=T_STAT, dt=dt,
                   truth_seeds=TRUTH_SEEDS, sensor_seeds=SENSOR_SEEDS,
                   supersedes="18_theory_diagnostics (single seed, n=784 only, "
                              "no boundary layouts, no normalisation)"),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
