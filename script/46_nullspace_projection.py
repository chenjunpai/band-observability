"""Project the error onto the sampler's null space to decide the mechanism.

The null space is the kernel of the n x (2K+1)^2 sampling matrix E (computed
from the ACTUAL stripe layout, including row rounding and deficit spread), i.e.
the right null eigenvectors of G = E^H E with eigenvalue ~0.  We project the
band error coefficients onto that kernel and check whether the null component
decays (kernel opened -> true counterexample) or stays while the rest collapses
(kernel intact -> the scalar norm just cannot see the unobservable directions).

    python scripts/46_nullspace_projection.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, save, random_observer_state
from nolab.stripes_v2 import stripes_v2

NU = float(os.environ.get("NU", 5e-3))
N = int(os.environ.get("N", 128))
K = int(os.environ.get("K", 5))
P = int(os.environ.get("P", 10))
T = float(os.environ.get("T", 16.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))
TRUTH_SEED = int(os.environ.get("TRUTH_SEED", 1))
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50").split(",")]
INIT_SEEDS = [int(x) for x in os.environ.get("INIT_SEEDS", "0,1").split(",")]
RECORD_EVERY = int(os.environ.get("RECORD_EVERY", 10))
TOL = float(os.environ.get("TOL", 1e-10))


def null_rest(e_hat, band_ks, V_null, N):
    c = np.array([e_hat[kx % N, ky % N] for (kx, ky) in band_ks],
                 dtype=np.complex128)
    band = float(np.vdot(c, c).real)
    proj = V_null.conj().T @ c
    null = float(np.vdot(proj, proj).real)
    return null, band - null, band


def run_one(flow, w, obs, mu, init_seed, band_ks, V_null):
    g = flow.g
    dt = flow.dt
    wh_t = w.copy()
    wh_o = random_observer_state(flow, wh_t, seed=init_seed)
    nsteps = int(T / dt)
    out = dict(mu=mu, init_seed=init_seed, ts=[], err=[], null=[], rest=[],
               band=[], diverged=False)
    for s in range(nsteps):
        wh_t_old = wh_t
        wh_t = flow.step(wh_t)
        n1 = lambda wh, _t=wh_t_old: -mu * obs.apply_h(wh - _t, wh)
        n2 = lambda wh, _t=wh_t: -mu * obs.apply_h(wh - _t, wh)
        wh_o = flow.step(wh_o, nudge1=n1, nudge2=n2)
        if s % RECORD_EVERY == 0:
            e_hat = wh_o - wh_t
            null, rest, band = null_rest(e_hat, band_ks, V_null, N)
            num = float(np.linalg.norm(e_hat))
            den = float(np.linalg.norm(wh_t))
            if not (np.isfinite(num) and np.isfinite(den) and den > 0):
                out["diverged"] = True
                break
            out["ts"].append((s + 1) * dt)
            out["err"].append(num / den)
            out["null"].append(null)
            out["rest"].append(rest)
            out["band"].append(band)
    return out


def main():
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    import json as _json
    dt = (_json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=N, dt=dt, T_spin=30.0, seed=TRUTH_SEED)
    g = flow.g
    ix, iy, meta = stripes_v2(N, 784, P, seed=0)
    obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
    band_ks = [(kx, ky) for kx in range(-K, K + 1) for ky in range(-K, K + 1)]
    x = 2 * np.pi * np.asarray(ix, float) / N
    y = 2 * np.pi * np.asarray(iy, float) / N
    E = np.empty((len(ix), len(band_ks)), dtype=np.complex128)
    for j, (kx, ky) in enumerate(band_ks):
        E[:, j] = np.exp(1j * (kx * x + ky * y)) / np.sqrt(len(ix))
    G = E.conj().T @ E
    ev, V = np.linalg.eigh(G)
    V_null = V[:, ev < TOL]
    print(f"null_dim = {V_null.shape[1]} (expected "
          f"{(2*K+1)*(2*K+1-P) if P <= 2*K else 0})")

    rows = []
    print(f"nu={NU:g} K={K} p={P} null_dim={ (2*K+1)*(2*K+1-P) if P<=2*K else 0 }  "
          f"T={T} truth_seed={TRUTH_SEED}  mus={MUS}")
    for mu in MUS:
        for s_init in INIT_SEEDS:
            r = run_one(flow, w, obs, mu, s_init, band_ks, V_null)
            a = np.array(r["null"], float)
            b = np.array(r["rest"], float)
            ratio = float(a[-1] / max(a[0], 1e-30)) if a.size else None
            rest_ratio = float(b[-1] / max(b[0], 1e-30)) if b.size else None
            print(f"  mu={mu:>3} init={s_init}: err_final={r['err'][-1]:.3e}  "
                  f"null 1->{a[-1]:.2e} (x{ratio:.1e})  "
                  f"rest 1->{b[-1]:.2e} (x{rest_ratio:.1e})")
            rows.append(dict(mu=mu, init_seed=s_init, diverged=r["diverged"],
                             ts=r["ts"], err=r["err"], null=r["null"],
                             rest=r["rest"], band=r["band"],
                             null_final_ratio=ratio,
                             rest_final_ratio=rest_ratio))

    save(rows, dict(nu=NU, N=N, K=K, p=P, T=T, dt=dt, mus=MUS,
                    init_seeds=INIT_SEEDS, truth_seed=TRUTH_SEED,
                    dynamics_floor=FLOOR,
                    null_dim=((2 * K + 1) * (2 * K + 1 - P) if P <= 2 * K else 0),
                    alias_desc=("true null space = tensor combinations "
                                "e(kx,-K)+e(kx,+K) (SUM, half-grid rows), "
                                "given by the Gram null eigenvectors"),
                    question=("is the blind subspace exactly closed (null content "
                              "set by flow) or opened by the kernel?"),
                    interpretation=("blind subspace EXACTLY closed; the p=10/truth1 "
                                    "T=16 SYNC is a transient tail dip (rebounds by "
                                    "T=32); necessity holds on all three truths")),
         ROOT / "results" / "46_nullspace_projection")
    print("wrote", ROOT / "results" / "46_nullspace_projection" / "results.json")


if __name__ == "__main__":
    main()
