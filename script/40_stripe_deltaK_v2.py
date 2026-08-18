"""40 -- the stripe delta_K ladder, recomputed on layouts whose count deficit is
spread instead of truncated.  No dynamics, ~10 min.

WHAT IT DECIDES
---------------
`results/37_stripe_nyquist_fixedn` holds the sensor count at exactly 784 by
truncating the tail of the layout, which deletes `deficit` CONSECUTIVE sensors
from the last band (see nolab/stripes_v2.py for the measured sizes).  Two
consequences, both visible in that results file:

  * the covering radius is non-monotone in p (p=13 h=0.491 and p=15 h=0.396 look
    worse than p=12 h=0.405) because the hole, not the band pitch, sets h;
  * delta_K is non-monotone in p, and the only two p with deficit = 0 (14 and
    16) are the only two that behave.  Within the nu = 2.5e-3 block the rank
    correlation between delta_K and the measured plateau is tau = +0.33, while
    between p and the plateau it is +1.00 -- i.e. in the family built to isolate
    the mechanism, the mechanism's own diagnostic is the WORSE predictor.  That
    is a referee-visible inconsistency in the main line.

With the deficit spread one-per-row, delta_K becomes monotone in p and its sign
flip lands exactly on p* = 2 K_c + 1 at all three viscosities:

    K_c = 4, p* =  9:  p=8  -0.00000  ->  p=9  +0.3877
    K_c = 5, p* = 11:  p=10 -0.00103  ->  p=11 +0.2610
    K_c = 7, p* = 15:  p=14 -0.00199  ->  p=15 +0.0757

This script regenerates that table (both variants side by side) so the numbers
in RESULTS.md 2.1 / 2.2 and PAPER_DATA_APPENDIX can be quoted from a run rather
than from this docstring, and so the monotonicity and threshold-location checks
are machine-verified rather than eyeballed.

READING
-------
  * `sign_flip_at_p_star` True for all three K_c  -> quote the jump, not "+0.073",
    and state that delta_K reproduces the parameter-free threshold exactly.
    Then rerun the dynamics with the fixed generator (scripts/41).
  * monotone_spread True, monotone_trunc False    -> the non-monotonicity in
    results/37 is a generator artefact and must be reported as such rather than
    as a property of delta_K.
  * If the flip does NOT land on p*, do not patch anything: report delta_K as an
    imperfect proxy and let the pitch carry the aliasing claim.

    python scripts/40_stripe_deltaK_v2.py
    KCS=5 python scripts/40_stripe_deltaK_v2.py        # one band only, ~3 min
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import Grid, save, stripes_exact, delta_K
from nolab.observations import covering_radius
from nolab.stripes_v2 import stripes_v2

OUT = ROOT / "results" / "40_stripe_deltaK_v2"
N_GRID = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
WIDTHS = [float(x) for x in os.environ.get("WIDTHS", "1.0").split(",")]

# K_c -> the p ladder to walk around p* = 2 K_c + 1.  K_c values are the
# c* = 0.75 crossings re-measured AT N = 128 in results/36_resolution_audit
# (nu = 1.5e-2 -> 4, nu = 5e-3 -> 5, nu = 2.5e-3 -> 7).
LADDERS = {
    4: [5, 6, 7, 8, 9, 10, 12],
    5: [6, 8, 9, 10, 11, 12, 14],
    7: [10, 12, 13, 14, 15, 16, 18],
}
if os.environ.get("KCS"):
    keep = {int(x) for x in os.environ["KCS"].split(",")}
    LADDERS = {k: v for k, v in LADDERS.items() if k in keep}


def main():
    os.makedirs(OUT, exist_ok=True)
    g = Grid(N_GRID)
    rows, summary = [], {}
    for K_c, ps in sorted(LADDERS.items()):
        p_star = 2 * K_c + 1
        print(f"\nK_c = {K_c}   p* = {p_star}   (N = {N_GRID}, n = {N_SENS}, "
              f"delta_K at K = {K_c}, denom_floor = 0)")
        print(f"   {'p':>3}{'def':>5}{'h trunc':>9}{'h spread':>9}"
              f"{'dK trunc':>11}{'dK spread':>11}")
        block = []
        for p in ps:
            per_seed = []
            for s in SEEDS:
                ta = stripes_exact(N_GRID, N_SENS, p, seed=s, jitter=True,
                                   jitter_mode="rowshift", strict_count=True)
                sp = stripes_v2(N_GRID, N_SENS, p, seed=s,
                                jitter_mode="rowshift")
                rec = dict(K_c=K_c, p=p, p_star=p_star, seed=s,
                           deficit=int(sp[2]["deficit"]),
                           thickness=int(sp[2]["thickness"]),
                           per_row=int(sp[2]["per_row"]),
                           rows_occupied=int(sp[2]["rows_occupied"]),
                           gap=float(sp[2]["gap"]),
                           n_trunc=int(len(ta[0])), n_spread=int(len(sp[0])),
                           h_trunc=float(covering_radius(g, ta[0], ta[1])),
                           h_spread=float(covering_radius(g, sp[0], sp[1])),
                           delta_K=dict(), N_grid=N_GRID,
                           n_sensors=N_SENS, denom_floor=0.0,
                           jitter_mode="rowshift")
                for wf in WIDTHS:
                    a, amax = delta_K(g, ta[0], ta[1], K_c, 0.0, width_factor=wf)
                    b, bmax = delta_K(g, sp[0], sp[1], K_c, 0.0, width_factor=wf)
                    rec["delta_K"][f"wf{wf:g}"] = dict(
                        trunc=a, trunc_max_eig=amax,
                        spread=b, spread_max_eig=bmax)
                per_seed.append(rec)
                rows.append(rec)
            a = float(np.mean([r["delta_K"]["wf1"]["trunc"] for r in per_seed]))
            b = float(np.mean([r["delta_K"]["wf1"]["spread"] for r in per_seed]))
            block.append((p, a, b))
            r0 = per_seed[0]
            print(f"   {p:>3}{r0['deficit']:>5}{r0['h_trunc']:>9.3f}"
                  f"{r0['h_spread']:>9.3f}{a:>+11.5f}{b:>+11.5f}"
                  + ("   <- p*" if p == p_star else ""))

        def monotone(vals, rel_tol=0.05):
            """Monotone up to `rel_tol` of the ladder's own range.

            An exact test is too strict here: above the threshold delta_K
            plateaus at O(0.3) and wobbles by ~0.002 between neighbouring p,
            which is the rounding of the row positions onto the grid, not a
            reversal.  The question is whether the ladder rises through the
            threshold, not whether it is exactly sorted.
            """
            span = max(vals) - min(vals)
            slack = rel_tol * max(span, 1e-12)
            return all(y >= x - slack for x, y in zip(vals, vals[1:]))

        def flip_at(idx):
            """p at which the value first becomes positive."""
            for p, a, b in block:
                if (b if idx == "spread" else a) > 0:
                    return p
            return None

        summary[str(K_c)] = dict(
            p_star=p_star,
            monotone_trunc=bool(monotone([a for _, a, _ in block])),
            monotone_spread=bool(monotone([b for _, _, b in block])),
            first_positive_trunc=flip_at("trunc"),
            first_positive_spread=flip_at("spread"),
            sign_flip_at_p_star=bool(flip_at("spread") == p_star),
            ladder=[(p, a, b) for p, a, b in block])
        s = summary[str(K_c)]
        print(f"   monotone in p:  tail-trunc {s['monotone_trunc']}, "
              f"deficit-spread {s['monotone_spread']}")
        print(f"   first delta_K > 0:  tail-trunc p = "
              f"{s['first_positive_trunc']}, deficit-spread p = "
              f"{s['first_positive_spread']}  (p* = {p_star})  "
              f"{'ON TARGET' if s['sign_flip_at_p_star'] else 'OFF TARGET'}")

    hits = sum(v["sign_flip_at_p_star"] for v in summary.values())
    print(f"\ndelta_K sign flip lands on p* in {hits}/{len(summary)} bands "
          f"(deficit-spread layouts)")
    print("tail-truncated layouts:  "
          + ", ".join(f"K_c={k}: flip at p={v['first_positive_trunc']}"
                      for k, v in sorted(summary.items())))
    save(rows, dict(N_grid=N_GRID, n_sensors=N_SENS, seeds=SEEDS,
                    widths=WIDTHS, denom_floor=0.0, jitter_mode="rowshift",
                    ladders={str(k): v for k, v in LADDERS.items()},
                    K_c_source="results/36_resolution_audit part B, c*=0.75, N=128",
                    summary=summary, sign_flip_on_p_star=f"{hits}/{len(summary)}",
                    question=("does the stripe delta_K ladder become monotone "
                              "and does its sign flip land on p* once the count "
                              "deficit is spread instead of truncated?"),
                    supersedes=("the delta_K column of "
                                "results/37_stripe_nyquist_fixedn and the stripe "
                                "rows of results/30_deltaK_canonical, both of "
                                "which carry a hole in the last band")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
