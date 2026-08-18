"""Extend 32_lattice_aliasing_fix for nu=2.5e-3 only, to find the first SYNC.

The main run (32) swept m up to 20 at nu=2.5e-3 and every m was still PARTIAL
(plateau 2.0e-2 at m=20), so the offset above m*=15 could not be quoted as a
number.  This adds m = 22, 24, 26 and writes a tiny sidecar result.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nolab import get_truth, FixedPointObs, trial, save
from nolab.observations import covering_radius
from nolab import lattice_m, delta_K
from nolab.verdict import outcome, best_verdict

OUT = ROOT / "results" / "32_lattice_aliasing_fix" / "extension_nu25"
NU = 2.5e-3
K_C = 7
M_STAR = 15
MUS = [5.0, 10.0, 20.0, 50.0]
M_EXTRA = [22, 24, 26]
FLOOR = 0.25
T = 8.0


def main():
    pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
    dt = (json.loads(pcal.read_text())["recommendations"]["DT"]
          if pcal.exists() else 0.002)
    flow, w = get_truth(NU, N=128, dt=dt, T_spin=30.0)
    g = flow.g
    rows = []
    for m in M_EXTRA:
        ix, iy, meta = lattice_m(g.N, m)
        h = float(covering_radius(g, ix, iy))
        dK0, _ = delta_K(g, ix, iy, K_C, 0.0)
        per_mu, verdicts = [], []
        for mu in MUS:
            obs = FixedPointObs(g, ix, iy, denom_floor=FLOOR)
            r = trial(flow, w, obs, mu, T=T)
            v = outcome(r["ts"], r["err"])
            verdicts.append(v["verdict"])
            per_mu.append(dict(mu=mu, verdict=v["verdict"],
                               plateau_rel=v["plateau_rel"],
                               rate=r["rate"]))
        bv = best_verdict(verdicts)
        rels = [x["plateau_rel"] for x in per_mu if x["plateau_rel"] is not None]
        rows.append(dict(nu=NU, K_c=K_C, m=m, m_star=M_STAR, n=int(meta["n_actual"]),
                         h=h, delta_K_floor0=dK0, best_verdict=bv,
                         best_plateau_rel=(float(min(rels)) if rels else None),
                         per_mu=per_mu))
        print(f"  m={m:3d} h={h:.3f} dK={dK0:+.5f} "
              f"plateau={'%.1e' % min(rels) if rels else 'n/a'} {bv}", flush=True)
    syncs = [r["m"] for r in rows if r["best_verdict"] == "SYNC"]
    first = min(syncs) if syncs else None
    print(f"first SYNC at m={first}; offset above m*={M_STAR} = "
          f"{(first - M_STAR) if first is not None else '>= ' + str(M_EXTRA[-1] - M_STAR)}",
          flush=True)
    save(rows, dict(nu=NU, K_c=K_C, m_star=M_STAR, T=T, dt=dt, mus=MUS,
                    floor=FLOOR, first_sync=first,
                    offset_above_m_star=(first - M_STAR) if first is not None else None,
                    extends="32_lattice_aliasing_fix (m=22,24,26 for nu=2.5e-3)"),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
