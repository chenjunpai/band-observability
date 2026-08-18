"""52 -- finite-time leading transverse (conditional) Lyapunov exponent of the
nudging observer across the stripe pitch threshold.

The tangent field obeys the linearised nudging-error equation about the truth
trajectory; its leading Lyapunov exponent is the transverse CLE that controls
whether synchronisation is possible.  This script measures a finite-time
estimate by integrating the tangent dynamics alongside the truth and rescaling
periodically.
"""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from nolab import get_truth, FixedPointObs
from nolab.stripes_v2 import stripes_v2

OUT = ROOT / "results" / "52_conditional_lyapunov"
NU = float(os.environ.get("NU", 5e-3))
N = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
MU = float(os.environ.get("MU", 10.0))
T_CLE = float(os.environ.get("T_CLE", 20.0))
TAU = float(os.environ.get("TAU", 1.0))
DISCARD = float(os.environ.get("DISCARD", 5.0))
PS = [int(x) for x in os.environ.get("PS", "8,10,11,12,14").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]


def tangent_rhs(flow, wh, dh, obs, mu):
    g = flow.g
    u, v = g.velocity(wh)
    dx = g.inv(1j * g.kx * dh)
    dy = g.inv(1j * g.ky * dh)
    du, dv = g.velocity(dh)
    wx = g.inv(1j * g.kx * wh)
    wy = g.inv(1j * g.ky * wh)
    adv = u * dx + v * dy + du * wx + dv * wy
    return -g.fwd(adv) * g.dealias - mu * obs.apply_h(dh)


def tangent_step(flow, wh, dh, obs, mu):
    dt, E = flow.dt, flow.E
    wh1 = flow.step(wh)
    N1 = tangent_rhs(flow, wh, dh, obs, mu)
    d1 = E * (dh + dt * N1)
    N2 = tangent_rhs(flow, wh1, d1, obs, mu)
    dh_new = E * dh + 0.5 * dt * (E * N1 + N2)
    return wh1, dh_new


def finite_time_cle(flow, wh0, obs, mu, T, tau, seed):
    g = flow.g
    rng = np.random.default_rng(seed)
    dh = (g.fwd(rng.standard_normal((g.N, g.N)))
          * np.exp(-0.5 * g.k2 / 16.0) * g.dealias)
    norm0 = 1e-4 * np.linalg.norm(wh0)
    dh *= norm0 / max(np.linalg.norm(dh), 1e-30)
    wh = wh0.copy()
    renorm_steps = max(int(tau / flow.dt), 1)
    nsteps = int(T / flow.dt)
    lam = []
    for s in range(nsteps):
        wh, dh = tangent_step(flow, wh, dh, obs, mu)
        if (s + 1) % renorm_steps == 0:
            cur = np.linalg.norm(dh)
            lam.append(float(np.log(cur / norm0) / tau))
            dh *= norm0 / max(cur, 1e-300)
    lam = np.asarray(lam)
    keep = lam[int(DISCARD / tau):]
    return float(np.mean(keep)) if keep.size else None


def main():
    os.makedirs(OUT, exist_ok=True)
    flow, w = get_truth(NU, N=N, seed=0)
    rows = []
    for p in PS:
        ix, iy, _ = stripes_v2(N, N_SENS, p, seed=0)
        obs = FixedPointObs(flow.g, ix, iy, denom_floor=0.25)
        vals = [finite_time_cle(flow, w, obs, MU, T_CLE, TAU, s) for s in SEEDS]
        vals = [v for v in vals if v is not None]
        rows.append(dict(p=p, mu=MU, cle_mean=float(np.mean(vals)),
                         cle_std=float(np.std(vals)) if len(vals) > 1 else None,
                         cle_seeds=[float(v) for v in vals]))
        print(f"p={p:>2}  cle={rows[-1]['cle_mean']:+.4f}  "
              f"seeds={[round(v,4) for v in vals]}")

    meta = dict(nu=NU, N=N, n_sensors=N_SENS, mu=MU, T_cle=T_CLE, tau=TAU,
                discard=float(DISCARD), seeds=SEEDS, p_star=11,
                note=("finite-time leading transverse Lyapunov exponent of the "
                      "nudging-error tangent dynamics on truth 0"))
    (OUT / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
