"""23 -- is the failure of gapped layouts a property of the FLOW or of the
interpolation operator?

THE CONFOUND
------------
`nolab.observations.PointObs` sets the Shepard kernel width from the sensor
COUNT, not from the geometry:

    h_kernel = width_factor * L / sqrt(n)

so two layouts with the same coverage radius but different n get different
observation operators, and a layout with a large gap gets a kernel tuned to its
mean density rather than to its local density.  On top of that, `denom_floor`
only bites where the kernel denominator is small -- i.e. only inside the gaps.
Measured on the v3 layouts (delta_K at K = 5):

    uniform n=784        floor 0.00 -> +0.1235     floor 0.25 -> +0.1234
    ground_tracks n=784  floor 0.00 -> -0.2056     floor 0.25 -> -0.1152
    clustered n=784      floor 0.00 -> -0.8615     floor 0.25 -> -0.0935

The mechanism number for the failing layouts moves by up to a factor of nine
with a regularisation parameter that does not affect the succeeding layouts at
all.  Until this is swept, "ground_tracks cannot synchronise" is not separable
from "ground_tracks was given a badly tuned interpolator", and that is the first
thing a data-assimilation referee will say.

WHAT THIS SCRIPT DOES
---------------------
Holds the layout fixed and sweeps the two knobs of the observation operator:

    width_factor in {0.5, 0.75, 1.0, 1.5, 2.0, 3.0}
    denom_floor  in {0.0, 0.1, 0.25, 0.5}
    mu           in {10, 50, 100, 200}

for ground_tracks, clustered, blind_half and (as the control) uniform, and
reports delta_K alongside every rate.

HOW TO READ THE RESULT
  * no (width, floor) rescues the gapped layouts  -> the failure is intrinsic,
    the paper's central negative result is safe, and delta_K < delta* is the
    right description of it;
  * some (width, floor) rescues them              -> the paper's claim changes
    to "the standard Shepard interpolator is mis-specified for gapped layouts,
    and here is the geometry-aware width that fixes it", which is a better
    paper, but it must be written that way.

    python scripts/23_kernel_ablation.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save, GENERATORS
from nolab.observations import covering_radius
from nolab import delta_K, corridor_diagnostics

OUT = ROOT / "results" / "23_kernel_ablation"
NU = float(os.environ.get("NU", 5e-3))
T = float(os.environ.get("T", 8.0))
N_SENS = int(os.environ.get("N_SENS", 784))
K_BAND = int(os.environ.get("K_BAND", 5))
WIDTHS = [float(x) for x in os.environ.get("WIDTHS", "0.5,0.75,1.0,1.5,2.0,3.0").split(",")]
FLOORS = [float(x) for x in os.environ.get("FLOORS", "0.0,0.1,0.25,0.5").split(",")]
MUS = [float(x) for x in os.environ.get("MUS", "10,50,100,200").split(",")]
LAYOUTS = os.environ.get("LAYOUTS", "ground_tracks,clustered,blind_half,uniform").split(",")


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    g = flow.g
    rows = []
    for name in LAYOUTS:
        ix, iy = GENERATORS[name](g.N, N_SENS, seed=0)
        h = covering_radius(g, ix, iy)
        corr = corridor_diagnostics(g, ix, iy)
        for wf in WIDTHS:
            for fl in FLOORS:
                dK, dmax = delta_K(g, ix, iy, K_BAND, fl, width_factor=wf)
                best, best_mu = 0.0, None
                for mu in MUS:
                    obs = FixedPointObs(g, ix, iy, denom_floor=fl,
                                        width_factor=wf)
                    r = trial(flow, w, obs, mu, T=T)
                    rows.append(dict(layout=name, width_factor=wf, floor=fl,
                                     mu=mu, h=float(h),
                                     max_corridor=corr["max_corridor"],
                                     delta_K=dK, delta_K_max_eig=dmax,
                                     rate=r["rate"], status=r["status"],
                                     converged=r["converged"],
                                     diverged=r["diverged"], final=r["final"]))
                    if r["rate"] > best:
                        best, best_mu = r["rate"], mu
                print(f"  {name:14} wf={wf:<4} floor={fl:<5} dK={dK:+.4f} "
                      f"best={best:.3f} @mu={best_mu}")
    rescued = sorted({r["layout"] for r in rows
                      if r["layout"] != "uniform" and r["converged"]})
    print("rescued layouts:", rescued or "none -- failure is intrinsic")
    save(rows, dict(nu=NU, T=T, n_sensors=N_SENS, dt=dt, K_band=K_BAND,
                    widths=WIDTHS, floors=FLOORS, mus=MUS, layouts=LAYOUTS,
                    rescued=rescued,
                    question=("can any Shepard kernel width / denom_floor "
                              "rescue a gapped layout?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
