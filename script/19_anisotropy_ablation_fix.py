"""19-fix -- does a directional gap matter beyond the coverage radius?

WHY THE ORIGINAL IS INVALID
---------------------------
`scripts/19_anisotropy_ablation.py` had three independent failures, any one of
which removes its power to answer its own question:

  1. its stripe generator refilled the gaps.  Each stripe was one grid row, so
     it held at most N = 128 distinct sensors; with n = 784 over p <= 6 stripes
     the surplus was thrown away by deduplication and replaced by UNIFORM RANDOM
     points.  Measured: 43% on-stripe at p = 3, 63% at p = 6, 86% at p = 16.  A
     uniform layout of 400 sensors already synchronises (rate 1.96), so the
     "anisotropic" family was carrying a fully sufficient isotropic layout
     inside it.
  2. the anisotropy it reported (1.01-1.05) came from
     `nolab.observations.anisotropy_index`, which cannot see stripes at all.
  3. the h range it covered was 0.31-0.55, entirely BELOW the h_c ~ 0.6 it was
     supposed to be probing.

The v3 conclusion drawn from it -- "the isotropic and anisotropic curves
coincide, therefore h explains everything and the geometric residual is
withdrawn" -- is therefore not supported by data.  It should read
"inconclusive", and this script is what makes it conclusive.

WHAT THIS VERSION DOES
----------------------
Two genuinely gapped families, both with NO random fill and both swept across
the threshold:

  * stripes_exact(n_stripes)  h ~ 0.20 -> 0.98, gap anisotropy 27-110
  * corridor(gap_width)       one empty strip of controlled width in an
                              otherwise uniform layout: the single-knob probe

against the isotropic uniform family at matched h.  Every layout is
characterised by h, max corridor width, delta_K and band coupling, so the run
answers the real question directly: do the families collapse on h, or on
delta_K?

FALSIFICATION RULE (fix it before you look at the output)
  * if the families collapse on h and split on delta_K -> report h
  * if they split on h and collapse on delta_K        -> report delta_K
  * if a matched-h pair has opposite outcomes         -> h is refuted, again

    python scripts/19_anisotropy_ablation_fix.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save, GENERATORS
from nolab import stripes_exact, corridor, layout_report, sync_rate_multi

OUT = ROOT / "results" / "19_anisotropy_ablation_fix"
NU = float(os.environ.get("NU", 5e-3))
T = float(os.environ.get("T", 8.0))
N_SENS = int(os.environ.get("N_SENS", 784))
K_BAND = int(os.environ.get("K_BAND", 5))
FLOOR = float(os.environ.get("FLOOR", 0.25))
MUS = [float(x) for x in os.environ.get("MUS", "10,20,50,100,200").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    g = flow.g
    rows = []

    def run(name, ix, iy, extra):
        rep = layout_report(g, ix, iy, K=K_BAND, denom_floor=FLOOR)
        per_mu = []
        for mu in MUS:
            obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
            r = trial(flow, w, obs, mu, T=T)
            multi = sync_rate_multi(r["ts"], r["err"])
            per_mu.append(dict(mu=mu, rate=r["rate"], status=r["status"],
                               converged=r["converged"], diverged=r["diverged"],
                               final=r["final"], rate_multi=multi["rate"],
                               rate_spread=multi["rate_spread"],
                               fit_stable=multi["stable"]))
        best = max(p["rate"] for p in per_mu)
        rep.pop("n", None)
        rows.append(dict(family=name, **extra, **rep, per_mu=per_mu,
                         best_rate=float(best),
                         best_at_edge=bool(
                             np.argmax([p["rate"] for p in per_mu])
                             in (0, len(MUS) - 1))))
        print(f"  {name:18} {str(extra):28} h={rep['h']:.3f} "
              f"corr={rep['max_corridor']:.3f} dK={rep['delta_K_floor']:+.4f} "
              f"best={best:.3f}")

    # isotropic reference, spanning the threshold
    for n in (150, 200, 250, 300, 400, 500, 784):
        for seed in SEEDS:
            ix, iy = GENERATORS["uniform"](g.N, n, seed=seed)
            run("uniform", ix, iy, dict(n=n, seed=seed))

    # true stripes at fixed n: h varies only through the directional gap
    for p in (3, 4, 6, 8, 12, 16):
        for seed in SEEDS:
            ix, iy, meta = stripes_exact(g.N, N_SENS, p, seed=seed, jitter=True)
            run("stripes_exact", ix, iy,
                dict(n_stripes=p, seed=seed, thickness=meta["thickness"],
                     gap=meta["gap"]))

    # single corridor at fixed n: one knob, purely directional
    for gw in (0.0, 0.4, 0.8, 1.2, 1.6, 2.0):
        for seed in SEEDS:
            if gw == 0.0:
                ix, iy = GENERATORS["uniform"](g.N, N_SENS, seed=seed)
            else:
                ix, iy, _ = corridor(g.N, N_SENS, gw, seed=seed)
            run("corridor", ix, iy, dict(gap_width=gw, seed=seed))

    save(rows, dict(nu=NU, T=T, n_sensors=N_SENS, dt=dt, mus=MUS,
                    seeds=SEEDS, K_band=K_BAND, floor=FLOOR,
                    question=("do gapped and uniform families collapse on h "
                              "or on delta_K?"),
                    supersedes="19_anisotropy_ablation (stripe generator "
                               "refilled its own gaps with uniform points)"),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
