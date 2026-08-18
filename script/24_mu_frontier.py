"""24 -- the cost-accuracy frontier with the nudging side actually converged.

WHY THE v3 FRONTIER IS NOT USABLE
---------------------------------
`results/20_letkf_sweep` sweeps nudging over mu in {20, 50, 100, 200} and gets

    seed 0:  mu=20  0.000 (no_fit)   mu=50  0.000   mu=100 0.356  mu=200 0.731
    seed 1:  mu=20  0.000 (no_fit)   mu=50  0.323   mu=100 0.474  mu=200 0.850

The rate is still rising monotonically at the LARGEST mu in the sweep.  The
quoted headline -- LETKF is ~2.6x more accurate at 16-24x the cost -- therefore
compares a converged LETKF configuration against a nudging configuration whose
optimum was never reached.  The 2.6x is an upper bound produced by the sweep
boundary, not a measurement.

This matters in the paper's favour, incidentally: if the true nudging optimum is
1.2-1.5, the story becomes "16-24x cheaper at 1.5x lower accuracy", which is a
much stronger argument for nudging in cost-constrained assimilation than the
current one.

WHAT THIS SCRIPT DOES
---------------------
  * extends mu geometrically until the rate stops improving OR the run goes
    unstable, and records WHICH of the two happened (with temporally sparse
    assimilation, dt_obs = 0.1, the explicit nudging term is only applied every
    m_assim steps, so there is a genuine stability ceiling around
    mu * dt_obs = O(1) -- that ceiling is a result, and should be reported as
    the physical limit of the method rather than hidden by a truncated sweep);
  * reports cost three ways, because "24x" depends on the convention:
    solver steps, wall-clock seconds, and steps-to-reach-1e-3 error;
  * runs both methods on the same layouts, same noise, same dt_obs, same
    initial ensemble/observer statistics.

    python scripts/24_mu_frontier.py
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import (get_truth, run_letkf, FixedPointObs, trial, save,
                   GENERATORS, sync_rate)
from nolab import sync_rate_multi, aggregate_mu

OUT = ROOT / "results" / "24_mu_frontier"
NU = float(os.environ.get("NU", 5e-3))
N_SENS = int(os.environ.get("N_SENS", 784))
T = float(os.environ.get("T", 6.0))
OBS_NOISE = float(os.environ.get("OBS_NOISE", 0.02))
DT_OBS = float(os.environ.get("DT_OBS", 0.1))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
MU_MAX = float(os.environ.get("MU_MAX", 6400))


def steps_to(ts, errs, tol=1e-3, dt=0.002):
    e = np.asarray(errs, float)
    idx = np.argmax(e < tol) if np.any(e < tol) else None
    return None if idx is None else int(np.asarray(ts, float)[idx] / dt)


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    m_assim = max(int(round(DT_OBS / dt)), 1)
    rows = []
    for seed in SEEDS:
        ix, iy = GENERATORS["uniform"](flow.g.N, N_SENS, seed=seed)

        # --- nudging: geometric mu sweep until it stops improving or blows up
        mu, best, stall = 12.5, -1.0, 0
        nud = []
        while mu <= MU_MAX and stall < 2:
            obs = FixedPointObs(flow.g, ix, iy, denom_floor=0.25)
            t0 = time.time()
            r = trial(flow, w, obs, mu, T=T, noise=OBS_NOISE,
                      m_assim=m_assim, seed=seed)
            wall = time.time() - t0
            multi = sync_rate_multi(r["ts"], r["err"])
            rec = dict(method="nudging", seed=seed, mu=mu,
                       solver_steps=int(T / dt), wall_s=wall,
                       steps_to_1e3=steps_to(r["ts"], r["err"], 1e-3, dt),
                       rate=r["rate"], rate_multi=multi["rate"],
                       fit_stable=multi["stable"], status=r["status"],
                       converged=r["converged"], diverged=r["diverged"],
                       reason=r["reason"], final=r["final"])
            rows.append(rec)
            nud.append(rec)
            print(f"  nudging s={seed} mu={mu:<7g} rate={r['rate']:.3f} "
                  f"{'DIVERGED' if r['diverged'] else ''}")
            if r["diverged"]:
                rec["stop_reason"] = "stability ceiling"
                break
            # only count a genuine plateau (a converged run that stopped
            # improving); a no_fit / no_decay run at low mu means "not yet
            # synchronised", not "optimum reached", and must NOT stop the sweep.
            stall = (stall + 1
                     if (r["rate"] > 0 and r["rate"] <= best * 1.02) else 0)
            best = max(best, r["rate"])
            mu *= 2

        if nud and not nud[-1].get("stop_reason"):
            # loop ended without divergence: either a genuine plateau (stall)
            # or we hit MU_MAX still improving.
            nud[-1]["stop_reason"] = ("mu_limit" if mu > MU_MAX else "stall")
        agg = aggregate_mu(nud)
        print(f"  nudging s={seed}: best={agg['best']:.3f} at mu={agg['best_mu']:g}"
              f"  at_edge={agg['at_edge']}")

        # --- LETKF at matched protocol
        for M in (8, 16, 24, 32):
            for loc in (0.5, 0.8, 1.5):
                t0 = time.time()
                ts, er, nsteps, info = run_letkf(
                    flow, w, ix, iy, M=M, T=T, dt_obs=DT_OBS, noise=OBS_NOISE,
                    loc_radius=loc, inflation="rtps", infl_param=0.9,
                    inflate_unobserved=False, block=8, seed=seed)
                wall = time.time() - t0
                m = sync_rate(ts, er)
                rows.append(dict(method="LETKF", seed=seed, M=M, loc=loc,
                                 inflation="rtps", infl_param=0.9,
                                 solver_steps=int(nsteps), wall_s=wall,
                                 steps_to_1e3=steps_to(ts, er, 1e-3, dt),
                                 rate=m["rate"], status=m["status"],
                                 converged=m["converged"],
                                 diverged=bool(info["diverged"]),
                                 reason=info["reason"],
                                 final=(float(er[-1]) if er.size else None)))
                print(f"  LETKF   s={seed} M={M} loc={loc} "
                      f"rate={m['rate']:.3f} steps={nsteps}")

    save(rows, dict(nu=NU, n_sensors=N_SENS, T=T, obs_noise=OBS_NOISE,
                    dt_obs=DT_OBS, dt=dt, seeds=SEEDS, mu_max=MU_MAX,
                    supersedes="20_letkf_sweep (nudging optimum was at the "
                               "sweep edge: rate still rising at mu=200)",
                    question="cost-accuracy frontier with both sides converged"),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
