"""25 -- the one prediction in this project that can be made before the run.

CLAIM
-----
A regular m x m lattice samples the torus at spacing 2*pi/m, so Fourier modes k
and k + m are indistinguishable to it.  The observation operator built on such a
lattice is therefore singular on the band |k| <= K whenever m <= 2K, and
non-singular as soon as m >= 2K + 1.  If synchronisation is controlled by the
observability of the low band -- rather than by the coverage radius h -- then
for the determining wavenumber K_c(nu) measured in script 12 there must be a
sharp transition at

    m* = 2 K_c + 1,        n* = m*^2

with FAILURE for every m <= 2K_c no matter how large mu is, and success above.

This is falsifiable, parameter-free, and it separates the two candidate
explanations cleanly, because h and m move together within the lattice family
but the aliasing threshold does not sit at any particular h -- it sits at an
integer determined by K_c(nu), so it MOVES with viscosity while h_c would not.

EVIDENCE ALREADY IN THE v3 DATA (nu = 5e-3, K_c = 5, so m* = 11)
    lattice n = 100 (m = 10)  h = 0.417  fails at every mu   delta_K = -0.0012
    lattice n = 144 (m = 12)  h = 0.347  converges, rate 0.88 delta_K = +0.0010
    uniform n = 400           h = 0.483  converges, rate 1.96
i.e. a layout with SMALLER h fails while a layout with larger h succeeds, and
delta_K crosses zero exactly between the two lattices.  This script turns that
coincidence into a test by repeating it at three viscosities, where K_c and
hence the predicted m* differ.

PREDICTION TABLE (fill in K_c from results/12_reynolds before running)
    nu = 1.5e-2   K_c = 4   ->  m* = 9    fail m <= 8,  pass m >= 9
    nu = 5.0e-3   K_c = 5   ->  m* = 11   fail m <= 10, pass m >= 11
    nu = 2.5e-3   K_c = 8   ->  m* = 17   fail m <= 16, pass m >= 17

If the transition tracks m* = 2K_c + 1 across viscosities, the mechanism is
established and h is demoted to a proxy.  If it sits at a fixed h instead, the
coverage-radius story survives and this script is the strongest possible
evidence for it.  Either outcome is publishable; the current draft asserts the
second without having run the test.

PARTIAL VALIDATION ALREADY RUN (results/25_lattice_aliasing_partial)
    nu = 5e-3, K_c = 5, m* = 11, mu in {10, 20}
        m = 10  n = 100  h = 0.417  delta_K = -0.0012  best rate 0.000  predicted fail  -> fail
        m = 12  n = 144  h = 0.347  delta_K = +0.0010  best rate 0.875  predicted pass  -> pass
    2/2.  Note the mu sensitivity discovered while running it: the SAME m = 12 lattice
    gives best rate 0.034 when the sweep is restricted to mu in {50, 200}.  Close to the
    aliasing boundary a large nudging strength amplifies the near-null direction of I_h
    and destroys the contraction, so the sweep must include small mu.  That is why MUS
    starts at 5 here, and it is a reportable observation in its own right.

    python scripts/25_lattice_aliasing.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, FixedPointObs, trial, save
from nolab.observations import covering_radius
from nolab import lattice_m, delta_K, band_coupling, sync_rate_multi

OUT = ROOT / "results" / "25_lattice_aliasing"
T = float(os.environ.get("T", 8.0))
FLOOR = float(os.environ.get("FLOOR", 0.25))
MUS = [float(x) for x in os.environ.get("MUS", "5,10,20,50,100,200").split(",")]

# nu -> (K_c from script 12, lattice sizes to sweep around m* = 2 K_c + 1)
CASES = {
    1.5e-2: (4, [6, 7, 8, 9, 10, 12]),
    5.0e-3: (5, [8, 9, 10, 11, 12, 14]),
    2.5e-3: (8, [12, 14, 16, 17, 18, 20]),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    rows = []
    for nu, (K_c, ms) in CASES.items():
        N = 128 if nu > 3e-3 else 256
        flow, w = get_truth(nu, N=N, dt=dt, T_spin=30.0)
        g = flow.g
        m_star = 2 * K_c + 1
        print(f"nu={nu:g}  K_c={K_c}  predicted m* = {m_star}")
        for m in ms:
            ix, iy, meta = lattice_m(g.N, m)
            h = covering_radius(g, ix, iy)
            dK0, dmax = delta_K(g, ix, iy, K_c, 0.0)
            dKf, _ = delta_K(g, ix, iy, K_c, FLOOR)
            bc = band_coupling(g, ix, iy, K_c, FLOOR)
            per_mu = []
            for mu in MUS:
                obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
                r = trial(flow, w, obs, mu, T=T)
                multi = sync_rate_multi(r["ts"], r["err"])
                per_mu.append(dict(mu=mu, rate=r["rate"], status=r["status"],
                                   converged=r["converged"],
                                   diverged=r["diverged"], final=r["final"],
                                   rate_multi=multi["rate"],
                                   fit_stable=multi["stable"]))
            best = max(p["rate"] for p in per_mu)
            predicted = "pass" if m >= m_star else "fail"
            observed = "pass" if best > 0.05 else "fail"
            rows.append(dict(nu=nu, N_grid=N, K_c=K_c, m=m, m_star=m_star,
                             n=meta["n_actual"], nyquist_K=meta["nyquist_K"],
                             h=float(h), delta_K_floor0=dK0,
                             delta_K_floor=dKf, delta_K_max_eig=dmax,
                             leak_fraction=bc["leak_fraction"],
                             best_rate=float(best), per_mu=per_mu,
                             predicted=predicted, observed=observed,
                             prediction_holds=bool(predicted == observed)))
            flag = "OK " if predicted == observed else "!! "
            print(f"  {flag}m={m:3d} n={meta['n_actual']:4d} h={h:.3f} "
                  f"dK={dKf:+.4f} best={best:.3f} "
                  f"predicted={predicted} observed={observed}")
    hold = sum(r["prediction_holds"] for r in rows)
    print(f"prediction m >= 2K_c+1 holds in {hold}/{len(rows)} cases")
    save(rows, dict(T=T, dt=dt, mus=MUS, floor=FLOOR,
                    cases={str(k): v for k, v in CASES.items()},
                    prediction="sync iff m >= 2*K_c + 1 (Nyquist on the "
                               "determining band), independent of h",
                    question="does the lattice threshold track K_c(nu) or a "
                             "fixed coverage radius?"),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
