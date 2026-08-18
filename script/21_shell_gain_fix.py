"""21-fix -- is the N_c staircase an integer-shell artifact?  Now with the
control that makes the question answerable.

WHY THE ORIGINAL CANNOT SUPPORT ITS CONCLUSION
----------------------------------------------
`scripts/21_shell_gain.py` compares a smooth radial taper
exp(-0.5 (|k|/rho)^6) against the hard spectral observation of script 10, and
concludes that the critical dof is ~40-70 rather than 121, "so the staircase is
an integer-shell artifact".  Two things are conflated:

  * the hard shell is a SQUARE box |kx|, |ky| <= K, which holds (2K+1)^2 = 121
    modes at K = 5.  The smooth taper is RADIAL, and a disc of radius 5 holds
    about pi*25 ~ 78 modes; the reported effective dof of 70 at rho = 5 is
    almost exactly that.  Most of the 121 -> 70 difference is square versus
    round, not smooth versus stepped.
  * "effective dof" was defined as sum(taper^2).  With sum(taper) instead the
    same operator has a materially larger dof.  A number that moves with an
    arbitrary convention cannot be compared against an exact mode count.

Also, the two largest rho values in the original output have rate_std of 2.48
and 3.81 against means of 3.72 and 5.05 -- those two points carry no
information and should not be plotted.

WHAT THIS VERSION ADDS
----------------------
  * a HARD RADIAL shell |k| <= rho as the missing control.  Comparing
    smooth-radial against hard-radial isolates smoothness; comparing
    hard-radial against hard-square isolates geometry.  Only the first
    comparison is evidence about the staircase.
  * both dof conventions (sum g, sum g^2) plus the exact mode count |k| <= rho;
  * more snapshots and the multi-protocol fit, so unstable points are flagged
    rather than averaged into a mean.

    python scripts/21_shell_gain_fix.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, trial, save
from nolab import sync_rate_multi

OUT = ROOT / "results" / "21_shell_gain_fix"
NU = float(os.environ.get("NU", 5e-3))
MU = float(os.environ.get("MU", 50.0))
T = float(os.environ.get("T", 8.0))
M_SNAP = int(os.environ.get("M_SNAP", 6))
RHOS = [float(x) for x in os.environ.get("RHOS", "3,3.5,4,4.5,5,5.5,6,7,8,10,12").split(",")]


class RadialObs:
    """gain(|k|) observation.  kind='smooth' -> exp(-0.5(|k|/rho)^beta),
    kind='hard' -> indicator(|k| <= rho)."""

    def __init__(self, g, rho, kind="smooth", beta=6.0):
        kmag = np.sqrt(g.k2)
        self.kind, self.rho = kind, rho
        self.gain = (np.exp(-0.5 * (kmag / rho) ** beta) if kind == "smooth"
                     else (kmag <= rho).astype(float))
        self.dof_sum = float(self.gain.sum())
        self.dof_sq = float((self.gain ** 2).sum())
        self.modes_in_disc = int((kmag <= rho).sum())
        self.ndof = self.dof_sq          # the v3 convention, kept for comparison
        self.name = f"{kind}_radial"

    def apply_h(self, resid_h, state_h=None):
        return resid_h * self.gain


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    snaps = [w.copy()]
    for _ in range(M_SNAP - 1):
        for _ in range(int(4.0 / flow.dt)):
            w = flow.step(w)
        snaps.append(w.copy())

    rows = []
    for kind in ("hard", "smooth"):
        for rho in RHOS:
            obs = RadialObs(flow.g, rho, kind=kind)
            rates, stable = [], []
            for ww in snaps:
                r = trial(flow, ww, obs, MU, T=T)
                m = sync_rate_multi(r["ts"], r["err"])
                rates.append(m["rate"])
                stable.append(m["stable"])
            rows.append(dict(kind=kind, rho=rho, dof_sum=obs.dof_sum,
                             dof_sq=obs.dof_sq,
                             modes_in_disc=obs.modes_in_disc,
                             square_equiv=(2 * int(rho) + 1) ** 2,
                             rate_mean=float(np.mean(rates)),
                             rate_std=float(np.std(rates, ddof=1)),
                             rates=rates,
                             all_fits_stable=bool(all(stable)),
                             usable=bool(all(stable)
                                         and np.std(rates, ddof=1)
                                         < 0.25 * max(np.mean(rates), 1e-9))))
            print(f"  {kind:6} rho={rho:<5g} disc={obs.modes_in_disc:4d} "
                  f"sum_g={obs.dof_sum:7.1f} sum_g2={obs.dof_sq:7.1f} "
                  f"rate={np.mean(rates):.3f}+-{np.std(rates, ddof=1):.3f} "
                  f"{'' if rows[-1]['usable'] else '(UNUSABLE)'}")

    save(rows, dict(nu=NU, mu=MU, T=T, M_snap=M_SNAP, dt=dt, rhos=RHOS,
                    supersedes="21_shell_gain (no hard-radial control; "
                               "square-vs-disc geometry conflated with "
                               "smooth-vs-stepped; two unusable points)",
                    question=("does smoothing remove the staircase once the "
                              "square-vs-disc difference is controlled for?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
