"""34 -- redo of 21-fix keeping the per-run curves, and the staircase question
answered properly.

WHAT 21-fix ALREADY SHOWED, AND WHAT IT COULD NOT
-------------------------------------------------
`results/21_shell_gain_fix` compares a hard radial shell |k| <= rho against a
smooth radial taper exp(-0.5(|k|/rho)^6) over rho = 3 ... 12.  Two things came
out of it, and one of them is not what the paper currently says.

  * BOTH curves are smooth and monotone in rho.  There is no staircase in the
    hard shell either.  So the "staircase" seen in v2/v3 was never a property of
    integer shells versus smooth ones -- it is what you get when a smooth,
    monotone rate(dof) curve is sampled only at integer K and then thresholded
    at c* = 0.75.  Every occurrence of "staircase"/"integer-shell artifact" in
    the draft has to go, including the sentence that claims to explain it.

  * At matched dof the smooth taper is 1.7-2.2x faster than the hard shell
    (e.g. sum g^2 ~ 70-85: hard 0.649, smooth 1.145).  Soft weighting beats hard
    truncation.  That is a cleaner and more useful result than the one it
    replaces, and it belongs in the classical-baseline section.

  * The square-versus-disc bookkeeping: at c* = 0.75 the hard radial shell needs
    rho ~ 5.6, about 97 modes, against 121 for the square |kx|,|ky| <= 5.  So of
    the original "121 -> 70" claim, 121 -> 97 is square-versus-disc and only
    97 -> ~36 is smoothing.

WHAT IT COULD NOT DO: 21-fix stored `rate_mean` and `rate_std` per (kind, rho)
and discarded every curve and every final error, so `scripts/29_reclassify`
cannot touch it.  Its rates come from the same `sync_rate` whose `converged`
flag is unreliable, and spectral observation at small rho is exactly the regime
where the error decays to a plateau rather than to zero.  Until the plateaus are
known, none of the three findings above is safe.

WHAT THIS VERSION DOES
----------------------
Same two families, same rho grid, but per snapshot it stores the full curve and
the plateau verdict, and it reports:
  * rate ONLY for runs whose verdict is SYNC;
  * the plateau for the rest;
  * the dof at which each family first reaches SYNC, under both dof conventions
    (sum g and sum g^2) and against the exact mode count in the disc;
  * the hard-square control at integer K, so the square/disc/smooth decomposition
    is measured in one run instead of being inferred across scripts.

    python scripts/34_shell_gain_verdicts.py       # ~40 min
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import get_truth, trial, save
from nolab.verdict import outcome, best_verdict

OUT = ROOT / "results" / "34_shell_gain_verdicts"
NU = float(os.environ.get("NU", 5e-3))
MU = float(os.environ.get("MU", 50.0))
T = float(os.environ.get("T", 8.0))
M_SNAP = int(os.environ.get("M_SNAP", 6))
RHOS = [float(x) for x in os.environ.get(
    "RHOS", "3,3.5,4,4.5,5,5.5,6,7,8,10,12").split(",")]
SQUARE_KS = [int(x) for x in os.environ.get("SQUARE_KS", "3,4,5,6,7,8").split(",")]


class ShellObs:
    """Three shell families on one interface.

    kind = 'hard'   indicator(|k| <= rho)          -- radial, sharp
    kind = 'smooth' exp(-0.5 (|k|/rho)^beta)       -- radial, tapered
    kind = 'square' indicator(|kx|,|ky| <= rho)    -- the family script 10 used

    Comparing hard vs smooth isolates smoothness; comparing hard vs square
    isolates disc-vs-box.  Only the first comparison is evidence about the
    shape of rate(dof); 21 (the original) ran only smooth vs square and
    attributed the whole difference to smoothness.
    """

    def __init__(self, g, rho, kind="smooth", beta=6.0):
        kmag = np.sqrt(g.k2)
        self.kind, self.rho = kind, rho
        if kind == "smooth":
            self.gain = np.exp(-0.5 * (kmag / rho) ** beta)
        elif kind == "hard":
            self.gain = (kmag <= rho).astype(float)
        elif kind == "square":
            self.gain = ((np.abs(g.kx) <= rho)
                         & (np.abs(g.ky) <= rho)).astype(float)
        else:
            raise ValueError(kind)
        self.dof_sum = float(self.gain.sum())
        self.dof_sq = float((self.gain ** 2).sum())
        self.modes_in_disc = int((kmag <= rho).sum())
        self.modes_in_square = int(((np.abs(g.kx) <= rho)
                                    & (np.abs(g.ky) <= rho)).sum())
        self.ndof = self.dof_sq
        self.name = f"{kind}_{rho:g}"

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

    jobs = ([("hard", r) for r in RHOS]
            + [("smooth", r) for r in RHOS]
            + [("square", float(k)) for k in SQUARE_KS])

    rows = []
    print(f"{'kind':<8}{'rho':>6}{'disc':>6}{'sq':>6}{'sum_g':>9}{'sum_g2':>9}"
          f"{'plateau':>10}{'rate|SYNC':>11}  verdict")
    for kind, rho in jobs:
        obs = ShellObs(flow.g, rho, kind=kind)
        per_run, verdicts, rels, rates = [], [], [], []
        for i, ww in enumerate(snaps):
            r = trial(flow, ww, obs, MU, T=T)
            v = outcome(r["ts"], r["err"])
            verdicts.append(v["verdict"])
            if v["plateau_rel"] is not None:
                rels.append(v["plateau_rel"])
            if v["rate_is_meaningful"]:
                rates.append(r["rate"])
            per_run.append(dict(snapshot=i, verdict=v["verdict"],
                                plateau_rel=v["plateau_rel"],
                                rate=r["rate"],
                                rate_is_meaningful=v["rate_is_meaningful"],
                                status_old=r["status"],
                                converged_old=r["converged"],
                                final=r["final"], ts=r["ts"], err=r["err"]))
        bv = best_verdict(verdicts)
        all_sync = all(v == "SYNC" for v in verdicts)
        rows.append(dict(kind=kind, rho=rho, dof_sum=obs.dof_sum,
                         dof_sq=obs.dof_sq,
                         modes_in_disc=obs.modes_in_disc,
                         modes_in_square=obs.modes_in_square,
                         best_verdict=bv, all_snapshots_sync=bool(all_sync),
                         plateau_rel_median=(float(np.median(rels))
                                             if rels else None),
                         plateau_rel_spread=(float(max(rels) / max(min(rels), 1e-30))
                                             if len(rels) >= 2 else None),
                         rate_mean_sync_only=(float(np.mean(rates))
                                              if rates else None),
                         rate_std_sync_only=(float(np.std(rates, ddof=1))
                                             if len(rates) >= 2 else None),
                         n_sync=int(sum(v == "SYNC" for v in verdicts)),
                         n_snap=len(verdicts), per_run=per_run))
        rr = rows[-1]
        print(f"{kind:<8}{rho:>6g}{obs.modes_in_disc:>6d}"
              f"{obs.modes_in_square:>6d}{obs.dof_sum:>9.1f}{obs.dof_sq:>9.1f}"
              f"{'n/a' if rr['plateau_rel_median'] is None else '%.1e' % rr['plateau_rel_median']:>10}"
              f"{'--' if rr['rate_mean_sync_only'] is None else '%.3f' % rr['rate_mean_sync_only']:>11}"
              f"  {bv} ({rr['n_sync']}/{rr['n_snap']})")

    # the three quantities the section needs, measured rather than inferred
    def first_sync(kind, key):
        cand = [r for r in rows if r["kind"] == kind and r["all_snapshots_sync"]]
        if not cand:
            return None
        best = min(cand, key=lambda r: r[key])
        return dict(rho=best["rho"], dof_sum=best["dof_sum"],
                    dof_sq=best["dof_sq"], modes=best["modes_in_disc"]
                    if kind != "square" else best["modes_in_square"])

    summary = {k: first_sync(k, "dof_sq") for k in ("hard", "smooth", "square")}
    print("\nfirst configuration where EVERY snapshot reaches SYNC:")
    for k, v in summary.items():
        print(f"   {k:<7}", v)
    if summary["hard"] and summary["square"]:
        print(f"   disc-vs-square saving: "
              f"{summary['square']['dof_sq']:.0f} -> {summary['hard']['dof_sq']:.0f} modes")
    if summary["hard"] and summary["smooth"]:
        print(f"   smoothing saving:      "
              f"{summary['hard']['dof_sq']:.0f} -> {summary['smooth']['dof_sq']:.0f} "
              f"(sum g^2 convention)")

    # is there any step in rate(dof) at all, once rho is continuous?
    hard = sorted((r for r in rows if r["kind"] == "hard"),
                  key=lambda r: r["dof_sq"])
    mono = all(a["plateau_rel_median"] is None or b["plateau_rel_median"] is None
               or b["plateau_rel_median"] <= a["plateau_rel_median"] * 1.5
               for a, b in zip(hard, hard[1:]))
    print(f"\nhard-radial plateau is monotone in dof (no step): {mono}")

    save(rows, dict(nu=NU, mu=MU, T=T, M_snap=M_SNAP, dt=dt, rhos=RHOS,
                    square_Ks=SQUARE_KS, first_sync=summary,
                    hard_monotone=bool(mono),
                    verdict_criterion="nolab.verdict.outcome (plateau)",
                    supersedes=("21_shell_gain_fix: stored only rate_mean, so "
                                "its numbers rest on the sync_rate converged "
                                "flag and cannot be reclassified"),
                    question=("with rho continuous, is there any staircase at "
                              "all, and how does the square/disc/smooth "
                              "decomposition split the dof saving?")),
         OUT)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
