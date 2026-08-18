"""28 -- does the configuration ranking survive noise, model error and sparse
observation times?

WHY THIS IS NOT OPTIONAL
------------------------
Every configuration result in v3 (scripts 13, 17, 19) is run with perfect
observations, a perfect model and assimilation at EVERY time step.  The paper is
being positioned for a data-assimilation audience, where none of those three
hold, and the first question from that audience will be whether the layout
ranking is a property of the flow or of the idealisation.  The LETKF comparison
already uses 2% noise and dt_obs = 0.1, so the two halves of the paper are not
even run under the same conditions.

This does not need to be a full sweep.  Four representative layouts x three
perturbations is enough to state either "the ordering is unchanged, absolute
rates drop by X%" or "the ordering changes, and here is where".

PERTURBATIONS
    observation noise   0, 1%, 5% of the field rms
    model error         observer integrated with nu' = nu * (1 +- 0.1)
    temporal sparsity   assimilate every 1, 25, 50 steps (dt_obs = 0.002 .. 0.1)

Note on the noise floor: with noise > 0 the error cannot decay past the
observation error level, so `sync_rate` must be read as the transient rate and
the plateau reported separately -- both are recorded here.

    python scripts/28_robustness_noise.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import (get_truth, FixedPointObs, trial, save, GENERATORS,
                   KolmogorovFlow)
from nolab import sync_rate_multi

OUT = ROOT / "results" / "28_robustness_noise"
NU = float(os.environ.get("NU", 5e-3))
T = float(os.environ.get("T", 8.0))
N_SENS = int(os.environ.get("N_SENS", 784))
MUS = [float(x) for x in os.environ.get("MUS", "20,50,100").split(",")]
LAYOUTS = os.environ.get("LAYOUTS", "lattice,uniform,ground_tracks,clustered").split(",")
NOISES = [float(x) for x in os.environ.get("NOISES", "0,0.01,0.05").split(",")]
NU_ERRS = [float(x) for x in os.environ.get("NU_ERRS", "0,0.1,-0.1").split(",")]
M_ASSIMS = [int(x) for x in os.environ.get("M_ASSIMS", "1,25,50").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    g = flow.g
    rows = []
    for name in LAYOUTS:
        for seed in SEEDS:
            ix, iy = GENERATORS[name](g.N, N_SENS, seed=seed)
            for noise in NOISES:
                for nu_err in NU_ERRS:
                    fo = (None if nu_err == 0 else
                          KolmogorovFlow(N=g.N, nu=NU * (1 + nu_err),
                                         alpha=flow.alpha,
                                         n_forcing=flow.nf, dt=dt))
                    for m_assim in M_ASSIMS:
                        best, rec_best = 0.0, None
                        for mu in MUS:
                            obs = FixedPointObs(g, ix, iy, denom_floor=0.25)
                            r = trial(flow, w, obs, mu, T=T, noise=noise,
                                      m_assim=m_assim, flow_obs=fo, seed=seed)
                            m = sync_rate_multi(r["ts"], r["err"])
                            rec = dict(layout=name, seed=seed, noise=noise,
                                       nu_err=nu_err, m_assim=m_assim, mu=mu,
                                       rate=r["rate"], rate_median=m["rate"],
                                       fit_stable=m["stable"],
                                       plateau_err=(min(r["err"])
                                                    if r["err"] else None),
                                       final=r["final"], status=r["status"],
                                       converged=r["converged"],
                                       diverged=r["diverged"])
                            rows.append(rec)
                            if r["rate"] > best:
                                best, rec_best = r["rate"], rec
                        print(f"  {name:14} s={seed} noise={noise:<5} "
                              f"nu_err={nu_err:<5} m_assim={m_assim:<3} "
                              f"best={best:.3f} "
                              f"plateau={rec_best['plateau_err'] if rec_best else None}")

    # ordering check on the PLATEAU error (not the retired rate metric, which is
    # meaningless outside the SYNC region and most noise/model-error runs are not SYNC).
    from nolab.ranking import ordering_by
    res = ordering_by(rows, LAYOUTS, "plateau_err", -1,
                      ["noise", "nu_err", "m_assim"])
    orders = res["orders"]
    unchanged = res["unchanged"]
    total = res["total"]
    print(f"\nlayout ordering unchanged in {unchanged}/{total} conditions")
    save(rows, dict(nu=NU, T=T, n_sensors=N_SENS, dt=dt, mus=MUS,
                    layouts=LAYOUTS, noises=NOISES, nu_errs=NU_ERRS,
                    m_assims=M_ASSIMS, seeds=SEEDS, orderings=orders,
                    ordering_unchanged=f"{unchanged}/{len(orders)}",
                    question=("does the configuration ranking survive noise, "
                              "model error and sparse observation times?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
