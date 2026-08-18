"""30 -- one canonical delta_K table for the whole paper.  No dynamics, ~5 min.

TWO THINGS THIS SETTLES
-----------------------
1.  WHICH `denom_floor` TO REPORT.  `denom_floor` only bites where the Shepard
    kernel denominator is small, i.e. only inside the sensor-free gaps, so it
    hands its largest correction to precisely the layouts that deserve the
    largest penalty.  Measured (K = 5, width_factor = 1, n = 784):

        layout          delta_K(floor 0)   delta_K(floor 0.25)   measured outcome
        uniform n=400        -0.050              -0.050          PARTIAL (plateau 3e-2)
        blind_half           -0.477              -0.023          FAIL    (plateau 0.85)
        clustered            -0.862              -0.094          FAIL    (plateau 1.4)
        ground_tracks        -0.206              -0.115          FAIL    (plateau 0.76)

    At floor 0.25, blind_half (-0.023) outranks uniform n = 400 (-0.050) while
    being far worse in every run.  At floor 0 the ordering is correct for every
    layout in the repository.  => report delta_K at floor 0; keep floor 0.25 in
    the dynamics if it is needed for numerical stability, and say plainly in the
    methods that the diagnostic and the simulated operator differ in that one
    parameter.

2.  THAT delta_K IS NOT A PROPERTY OF THE SENSOR GEOMETRY.  It is a functional of
    (geometry, interpolation kernel).  `stripes_exact p = 8` is the only layout
    in results/19_anisotropy_ablation_fix that the delta_K ordering gets wrong,
    and the reason is the kernel width: at width_factor 1 it scores -0.029,
    ahead of uniform n = 400 at -0.050, but it fails hard while uniform n = 400
    nearly synchronises.  At width_factor 0.5 the order is right (-0.020 vs
    +0.088).  So the paper must (a) bind delta_K to one named operator, (b) print
    that operator's parameters next to every number, and (c) state that delta_K
    values computed at different kernel widths are not comparable.

WHAT IT WRITES
--------------
For every layout family used anywhere in the repository, at every
(width_factor, denom_floor, K) of interest: delta_K, the largest eigenvalue of
sym(I_h) (the normalisation -- the operator is not a contraction, max eig runs
1.00-1.77), the out-of-band leakage, covering radius, and the corridor
diagnostics.  This is the table Fig 1 is plotted from and the table every
delta_K quoted in the text must be traceable to.

    python scripts/30_deltaK_canonical.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import Grid, GENERATORS, save
from nolab.observations import covering_radius, anisotropy_index
from nolab import (stripes_exact, corridor, lattice_m, delta_K,
                       band_coupling, corridor_diagnostics)

OUT = ROOT / "results" / "30_deltaK_canonical"
N_GRID = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
KS = [int(x) for x in os.environ.get("KS", "4,5,7").split(",")]
WIDTHS = [float(x) for x in os.environ.get("WIDTHS", "0.5,1.0,2.0").split(",")]
FLOORS = [float(x) for x in os.environ.get("FLOORS", "0.0,0.25").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
K_MAIN = int(os.environ.get("K_MAIN", 5))       # the K used for the printed table


def layouts(g):
    """Every layout family the paper refers to, as (label, ix, iy, extra)."""
    out = []
    for n in (150, 200, 250, 300, 400, 500, 784):
        for s in SEEDS:
            ix, iy = GENERATORS["uniform"](g.N, n, seed=s)
            out.append((f"uniform_n{n}", ix, iy, dict(family="uniform", n=n, seed=s)))
    for name in ("ground_tracks", "clustered", "blind_half"):
        for s in SEEDS:
            ix, iy = GENERATORS[name](g.N, N_SENS, seed=s)
            out.append((name, ix, iy, dict(family=name, n=N_SENS, seed=s)))
    for p in (3, 4, 6, 8, 9, 10, 11, 12, 14, 16):
        for s in SEEDS:
            try:
                ix, iy, meta = stripes_exact(g.N, N_SENS, p, seed=s, jitter=True)
            except ValueError:
                continue
            out.append((f"stripes_p{p}", ix, iy,
                        dict(family="stripes_exact", n_stripes=p, seed=s,
                             thickness=meta["thickness"], gap=meta["gap"])))
    for gw in (0.4, 0.8, 1.2, 1.6, 2.0):
        for s in SEEDS:
            ix, iy, _ = corridor(g.N, N_SENS, gw, seed=s)
            out.append((f"corridor_gw{gw}", ix, iy,
                        dict(family="corridor", gap_width=gw, seed=s)))
    for m in (8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20):
        ix, iy, meta = lattice_m(g.N, m)
        out.append((f"lattice_m{m}", ix, iy,
                    dict(family="lattice_m", m=m, seed=0,
                         n=meta["n_actual"], nyquist_K=meta["nyquist_K"])))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    g = Grid(N_GRID)
    rows = []
    print(f"grid N={N_GRID}, n_sensors={N_SENS}, K={KS}, "
          f"widths={WIDTHS}, floors={FLOORS}")
    print(f"{'layout':<18}{'seed':>5}{'h':>7}{'corr':>7}"
          f"{'dK(f0,wf1)':>12}{'dK(f.25,wf1)':>14}{'maxeig':>8}{'leak':>10}")
    for label, ix, iy, extra in layouts(g):
        h = float(covering_radius(g, ix, iy))
        cd = corridor_diagnostics(g, ix, iy)
        rec = dict(label=label, **extra, n_actual=int(len(ix)), h=h,
                   anisotropy_old=float(anisotropy_index(g, ix, iy)),
                   N_grid=N_GRID, **{k: cd[k] for k in
                                     ("max_corridor", "min_corridor",
                                      "gap_anisotropy", "argmax_angle")})
        dks = {}
        for K in KS:
            for wf in WIDTHS:
                for fl in FLOORS:
                    lo, hi = delta_K(g, ix, iy, K, fl, width_factor=wf)
                    key = f"K{K}_wf{wf:g}_floor{fl:g}"
                    dks[key] = dict(delta_K=lo, max_eig=hi)
                    if K == K_MAIN and wf == 1.0:
                        bc = band_coupling(g, ix, iy, K, fl, width_factor=wf)
                        dks[key]["leak_fraction"] = bc["leak_fraction"]
        rec["delta_K"] = dks
        rows.append(rec)
        a = dks.get(f"K{K_MAIN}_wf1_floor0", {})
        b = dks.get(f"K{K_MAIN}_wf1_floor0.25", {})
        print(f"{label:<18}{extra.get('seed', 0):>5}{h:>7.3f}"
              f"{cd['max_corridor']:>7.2f}"
              f"{a.get('delta_K', float('nan')):>+12.4f}"
              f"{b.get('delta_K', float('nan')):>+14.4f}"
              f"{a.get('max_eig', float('nan')):>8.3f}"
              f"{a.get('leak_fraction', float('nan')):>10.1e}")

    # the two claims this script is meant to settle, checked numerically
    def get(r, K, wf, fl):
        return r["delta_K"].get(f"K{K}_wf{wf:g}_floor{fl:g}", {}).get("delta_K")

    floor_flips = []
    for r in rows:
        a, b = get(r, K_MAIN, 1.0, 0.0), get(r, K_MAIN, 1.0, 0.25)
        if a is not None and b is not None and abs(b - a) > 0.02:
            floor_flips.append(dict(label=r["label"], seed=r.get("seed"),
                                    floor0=a, floor025=b, shift=b - a))
    width_spread = []
    for r in rows:
        vals = [get(r, K_MAIN, wf, 0.0) for wf in WIDTHS]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            width_spread.append(dict(label=r["label"], seed=r.get("seed"),
                                     values=vals,
                                     range=float(max(vals) - min(vals))))
    ws = sorted(width_spread, key=lambda d: -d["range"])[:10]
    print("\nlayouts whose delta_K moves most with denom_floor "
          "(these are exactly the gapped ones -- report floor 0):")
    for f in sorted(floor_flips, key=lambda d: -abs(d["shift"]))[:8]:
        print(f"   {f['label']:<18} s={f['seed']} "
              f"{f['floor0']:+.4f} -> {f['floor025']:+.4f}  "
              f"(shift {f['shift']:+.4f})")
    print("\nlayouts whose delta_K moves most with kernel width "
          "(delta_K is not geometry alone):")
    for f in ws[:8]:
        print(f"   {f['label']:<18} s={f['seed']} "
              + " ".join(f"{v:+.4f}" for v in f["values"])
              + f"   range {f['range']:.4f}")

    save(rows, dict(N_grid=N_GRID, n_sensors=N_SENS, Ks=KS, widths=WIDTHS,
                    floors=FLOORS, seeds=SEEDS, K_main=K_MAIN,
                    floor_sensitive_layouts=floor_flips,
                    width_sensitive_layouts=width_spread,
                    canonical=("delta_K reported in the paper = "
                               "K=5, width_factor=1.0, denom_floor=0.0, "
                               "Shepard Gaussian kernel with "
                               "h_kernel = width_factor * L / sqrt(n)"),
                    question=("one traceable delta_K per layout, and the "
                              "sensitivity of delta_K to the two operator "
                              "parameters it silently depends on")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
