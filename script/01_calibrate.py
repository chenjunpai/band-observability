"""Calibrate the four hyper-parameters every later experiment depends on:

    A) DT      -- is the synchronisation rate converged in dt?
    B) N_GRID  -- is the rate converged in resolution?
    C) T_SPIN  -- how long does each alpha need to reach the attractor?
    D) DECORR  -- how far apart must snapshots be to be independent?

The old package never answered any of these; it hard-coded dt=0.004,
N=128, T_SPIN=15..30, DECORR=8 and the exp18 alpha sweep ran on states
that were nowhere near the attractor (P1-5).

    python 01_calibrate.py

Writes results/01_calibrate/calibrate.json which the other scripts read.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import KolmogorovFlow, SpectralObs, get_truth, trial, save, \
    equilibration_time

OUT = ROOT / "results" / "01_calibrate"
NU = float(os.environ.get("NU", 5e-3))
MU = float(os.environ.get("MU", 50.0))
T = float(os.environ.get("T", 8.0))


def rate_vs_dt():
    """A: rate at dt=0.004, 0.002, 0.001 (spectral K=6, same physics).

    dt=0.008 is above the advective CFL limit for the turbulent attractor
    and is unusable; the three usable candidates are compared under CFL
    protection so an unstable configuration cannot masquerade as a result.
    """
    out = {}
    for dt in (0.004, 0.002, 0.001):
        flow, w = get_truth(NU, N=128, dt=dt, T_spin=20.0, cfl=True)
        r = trial(flow, w, SpectralObs(flow.g, 6), MU, T=T)
        out[str(dt)] = dict(rate=r["rate"], r2=r["r2"], status=r["status"],
                            final=r["final"])
        print(f"  [A] dt={dt} rate={r['rate']:.4f} status={r['status']}")
    r4, r2, r1 = out["0.004"]["rate"], out["0.002"]["rate"], out["0.001"]["rate"]
    d1 = abs(r2 - r4) / max(r4, 1e-9)
    d2 = abs(r1 - r2) / max(r2, 1e-9)
    best = 0.004 if d1 < 0.02 else (0.002 if d2 < 0.02 else 0.001)
    print(f"  [A] relative changes: dt 4->2 {d1:.3f}, 2->1 {d2:.3f}; "
          f"choose dt={best}")
    out["_summary"] = dict(rel_change_4_2=d1, rel_change_2_1=d2,
                           chosen_dt=best)
    return out


def rate_vs_N():
    """B: rate at N=48,64,96,128 with the physical projection fixed."""
    out = {}
    for N in (48, 64, 96, 128):
        K = max(int(round(6 * N / 128)), 1)
        flow, w = get_truth(NU, N=N, dt=0.004, T_spin=20.0)
        r = trial(flow, w, SpectralObs(flow.g, K), MU, T=T)
        out[str(N)] = dict(rate=r["rate"], K=K, status=r["status"])
        print(f"  [B] N={N} K={K} rate={r['rate']:.4f}")
    rates = [out[str(N)]["rate"] for N in (48, 64, 96, 128)]
    d = max(abs(rates[i] - rates[i - 1]) / max(rates[i - 1], 1e-9)
            for i in range(1, len(rates)))
    chosen = 96 if d < 0.05 else 128
    print(f"  [B] max relative change {d:.3f}; choose N={chosen}")
    out["_summary"] = dict(max_rel_change=d, chosen_N=chosen)
    return out


def spinup_vs_alpha():
    """C: equilibration time for alpha in {0.1, 0.03, 0.01, 0.003}."""
    out = {}
    for alpha in (0.1, 0.03, 0.01, 0.003):
        flow = KolmogorovFlow(N=128, nu=NU, alpha=alpha, dt=0.004)
        w = flow.spinup(T=10.0, seed=0)
        max_T = max(40.0, min(10.0 / alpha, 250.0))
        t_eq, w, hist = equilibration_time(flow, w, tol=0.03,
                                           window=5.0, min_T=10.0,
                                           max_T=max_T)
        equilibrated = t_eq < max_T - 1e-9
        rec = dict(alpha=alpha, damping_time=1.0 / alpha,
                   t_equilibrate=t_eq, equilibrated=bool(equilibrated),
                   recommended_T_SPIN=max(30.0, 2.0 * t_eq))
        out[str(alpha)] = rec
        print(f"  [C] alpha={alpha} t_eq={t_eq:.1f} "
              f"equilibrated={equilibrated} "
              f"recommend T_SPIN={rec['recommended_T_SPIN']:.1f}")
    return out


def decorrelation():
    """D: first lag at which enstrophy autocorrelation drops below 1/e."""
    flow = KolmogorovFlow(N=128, nu=NU, alpha=0.1, dt=0.004)
    w = flow.spinup(T=30.0, seed=0)
    series, ts = [], []
    every = max(int(0.2 / flow.dt), 1)
    for s in range(int(80.0 / flow.dt)):
        w = flow.step(w)
        if s % every == 0:
            series.append(flow.enstrophy(w))
            ts.append(s * flow.dt)
    series = np.asarray(series)
    s = series - series.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:] / max(s.dot(s), 1e-30)
    lag = None
    for i, v in enumerate(ac):
        if v < np.exp(-1.0):
            lag = ts[i]
            break
    if lag is None:
        lag = ts[-1]
    print(f"  [D] autocorrelation first below 1/e at t={lag:.2f}")
    return dict(decorr_time=float(lag), recommended_DECORR=max(4.0, float(lag)))


def main():
    os.makedirs(OUT, exist_ok=True)
    res = dict(
        nu=NU, mu=MU, T=T,
        dt=rate_vs_dt(),
        resolution=rate_vs_N(),
        spinup=spinup_vs_alpha(),
        decorr=decorrelation(),
    )
    chosen_dt = res["dt"]["_summary"]["chosen_dt"]
    chosen_N = res["resolution"]["_summary"]["chosen_N"]
    t_spin_map = {a: v["recommended_T_SPIN"]
                  for a, v in res["spinup"].items()}
    res["recommendations"] = dict(
        DT=chosen_dt, N_GRID=chosen_N,
        T_SPIN_alpha=t_spin_map,
        T_SPIN_default=30.0,
        DECORR=res["decorr"]["recommended_DECORR"],
    )
    with open(OUT / "calibrate.json", "w") as f:
        json.dump(res, f, indent=2, allow_nan=False)
    print("\nrecommendations:", json.dumps(res["recommendations"], indent=2))


if __name__ == "__main__":
    main()
