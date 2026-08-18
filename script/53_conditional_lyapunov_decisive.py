"""53 -- decisive finite-time CLE scan for the one-pitch offset.

Truth 0: full gain scan mu in {5,10,20,50} over p = {8,10,11,12,14}.
Truths 1/2: threshold-neighbourhood check at mu = 10 over p = {11,12}.
Two tangent seeds per configuration.  Progress is flushed line by line.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nolab import get_truth, FixedPointObs
from nolab.stripes_v2 import stripes_v2

OUT = ROOT / "results" / "53_conditional_lyapunov_decisive"
NU = float(os.environ.get("NU", 5e-3))
N = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
T_CLE = float(os.environ.get("T_CLE", 20.0))
TAU = float(os.environ.get("TAU", 1.0))
DISCARD = float(os.environ.get("DISCARD", 5.0))
TANGENT_SEEDS = [0, 1]


def log(msg):
    print(msg, flush=True)


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


def run_one(flow, w, p, mu, truth):
    ix, iy, _ = stripes_v2(N, N_SENS, p, seed=0)
    obs = FixedPointObs(flow.g, ix, iy, denom_floor=0.25)
    vals = [finite_time_cle(flow, w, obs, mu, T_CLE, TAU, s)
            for s in TANGENT_SEEDS]
    vals = [v for v in vals if v is not None]
    mean = float(np.mean(vals)) if vals else None
    log(f"truth={truth} mu={mu:>4g} p={p:>2}  "
        f"cle={mean:+.4f}  seeds={[round(v,4) for v in vals]}")
    return dict(truth=truth, mu=mu, p=p, cle_mean=mean,
                cle_seeds=[float(v) for v in vals])


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    total = 0

    # truth 0: full gain scan
    flow0, w0 = get_truth(NU, N=N, seed=0)
    for mu in (5.0, 10.0, 20.0, 50.0):
        for p in (8, 10, 11, 12, 14):
            rows.append(run_one(flow0, w0, p, mu, 0))
            total += 1

    # truths 1/2: threshold neighbourhood at mu=10
    for truth in (1, 2):
        flow, w = get_truth(NU, N=N, seed=truth)
        for p in (11, 12):
            rows.append(run_one(flow, w, p, 10.0, truth))
            total += 1

    meta = dict(nu=NU, N=N, n_sensors=N_SENS, T_cle=T_CLE, tau=TAU,
                discard=float(DISCARD), tangent_seeds=TANGENT_SEEDS,
                p_star=11, n_runs=total,
                note="decisive finite-time leading transverse CLE scan")
    (OUT / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
    log(f"wrote {OUT / 'results.json'} ({total} runs)")


if __name__ == "__main__":
    main()
