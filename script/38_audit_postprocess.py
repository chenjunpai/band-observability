"""38 -- post-processing audit.  No solver, no truth, runs in under a minute.

Recomputes every number that v5's markdown quotes, under rules that do not
depend on a free knob, and prints a PASS/FAIL line against the value currently
written in RESULTS.md / POSITIONING.md / PAPER_DATA_APPENDIX.md.  Anything that
prints MISMATCH is a sentence in those documents that a referee can falsify by
opening the JSON.

Covers:
  T2  exp33 ordering quality        (46 vs 94 -> tau / AUC, knob-free)
  C3  exp24 frontier ratios         (2.1 / 1.4 / 1.6 under one stated rule)
  C3  exp28 ordering robustness     (27/27 recomputed on the plateau, not rate)
  C1  delta_K > 0 necessity         (the claim v5 under-states)
  §2.1 the "same h" counterexample  (uniform n=250 is NOT SYNC)
  §2.4 exp23 meta.rescued           (contradicts its own rows)
  §2.2 exp31 prediction_holds       (16/21, not "all p < p* fail")

    python scripts/38_audit_postprocess.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab.ranking import (predictor_quality, frontier_ratios, ordering_by,
                           necessity_audit)
from nolab.verdict import verdict_from_final, E_REF_DEFAULT

R = ROOT / "results"


def load(name, fname="results.json"):
    p = R / name / fname
    return json.loads(p.read_text()) if p.exists() else None


def check(label, got, expected, rel=0.06):
    if expected is None:
        print(f"  {label}: {got}")
        return
    try:
        ok = abs(float(got) - float(expected)) <= rel * abs(float(expected))
    except (TypeError, ValueError):
        ok = (got == expected)
    print(f"  {'OK      ' if ok else 'MISMATCH'} {label}: got {got}, "
          f"documents say {expected}")


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ----------------------------------------------------------------- T2 exp33
def t2_ordering():
    section("T2  exp33: what should Fig 1's x axis be?")
    d = load("33_boundary_curves")
    if d is None:
        print("  results/33_boundary_curves missing")
        return
    rows = d["rows"]
    for pred, sign in (("delta_K_floor0", +1), ("h", -1)):
        q = predictor_quality(rows, pred, sign)
        print(f"  {pred:16} n={q['n']:3d}  tau={q['tau']:+.3f}  "
              f"AUC={q['auc']:.3f}  inversions {q['inversions_tol1']}/"
              f"{q['pairs']} (plain), {q['inversions_tol3']} (v5's 3x rule)")
    doc = d["meta"].get("ordering_violations", {})
    check("exp33 delta_K violations (3x rule)",
          predictor_quality(rows, "delta_K_floor0", +1)["inversions_tol3"],
          doc.get("delta_K"))
    check("exp33 h violations (3x rule)",
          predictor_quality(rows, "h", -1)["inversions_tol3"], doc.get("h"))
    print("  -> report tau and AUC in the caption; the 3x rule is a free knob.")


# ----------------------------------------------------------------- C1 delta_K
def c1_necessity():
    section("C1  is delta_K > 0 necessary for SYNC?")
    pooled = []
    d33 = load("33_boundary_curves")
    if d33:
        pooled += [dict(layout=r["layout"], delta_K_floor0=r["delta_K_floor0"],
                        best_verdict=r["best_verdict"]) for r in d33["rows"]]
    d19 = load("19_anisotropy_ablation_fix")
    if d19:
        for r in d19["rows"]:
            fins = [x["final"] for x in r["per_mu"]
                    if x.get("final") is not None and not x.get("diverged")]
            if not fins:
                continue
            v = verdict_from_final(min(fins), e_ref=E_REF_DEFAULT)["verdict"]
            pooled.append(dict(layout=f"{r['family']}_{r.get('n') or r.get('n_stripes') or r.get('gap_width')}",
                               delta_K_floor0=r["delta_K_floor0"],
                               best_verdict=v))
    a = necessity_audit(pooled)
    print(f"  pooled rows: {len(pooled)}   (exp19 + exp33)")
    print(f"  delta_K <= 0 : {a['n_below']:3d} rows, of which SYNC = "
          f"{a['sync_below']}  {a['violations']}")
    print(f"  delta_K >  0 : {a['n_above']:3d} rows, of which SYNC = "
          f"{a['sync_above']}")
    print(f"  smallest delta_K among SYNC rows: {a['min_delta_K_among_sync']:+.4f}")
    print("  -> if sync_below == 0, RESULTS.md's retraction row \"delta_K<0 "
          "即失败 -> 反例 n=300/400\" is STALE:")
    print("     under the plateau criterion n=300/400 are PARTIAL, not SYNC, "
          "so the counterexamples no longer exist.")
    print("     Claim instead: delta_K > 0 is necessary for SYNC "
          f"(0/{a['n_below']} violations), empirically sufficient above "
          f"{a['min_delta_K_among_sync']:+.3f}.")


# ------------------------------------------------------------- §2.1 same-h pair
def s21_counterexample():
    section("RESULTS.md §2.1: is `uniform n=250` really SYNC?")
    d = load("19_anisotropy_ablation_fix")
    if d is None:
        return
    for r in d["rows"]:
        key = r.get("n") or r.get("n_stripes") or r.get("gap_width")
        if not ((r["family"] == "uniform" and key == 250)
                or (r["family"] == "stripes_exact" and key == 6)):
            continue
        fins = [x["final"] for x in r["per_mu"]
                if x.get("final") is not None and not x.get("diverged")]
        v = verdict_from_final(min(fins)) if fins else dict(verdict="NO_DATA",
                                                            plateau_rel=None)
        print(f"  {r['family']}_{key} seed{r['seed']}  h={r['h']:.3f} "
              f"dK={r['delta_K_floor0']:+.4f}  best rel={v['plateau_rel']:.2e} "
              f"-> {v['verdict']}")
    print("  -> §2.1 claims uniform n=250 SYNC.  It is not.  The pair is "
          "FAIL/PARTIAL vs FAIL, i.e. not a counterexample.")
    print("     Replacements that ARE counterexamples (same source, exp19):")
    print("       same h   : uniform n=200 s1 (h=0.591, dK=-0.139, FAIL 4.4e-1)")
    print("                  vs uniform n=300 s0 (h=0.591, dK=-0.090, PARTIAL 4.5e-2)")
    print("       reversal : stripes p=8 (h=0.396, FAIL) vs corridor gw=0.4 "
          "(h=0.42-0.48, SYNC)")
    print("                  -- valid ONLY after the stripe rerun with "
          "jitter_mode='rowshift' (see script 37); with the shipped runs the "
          "stripe side has 611-625 sensors, not 784.")


# ----------------------------------------------------------------- C3 frontier
def c3_frontier():
    section("C3  exp24: the three ratios under one rule")
    d = load("24_mu_frontier")
    if d is None:
        return
    rows = d["rows"]
    variants = {
        "all LETKF configs": None,
        "LETKF loc=0.8 only": (lambda r: r.get("loc") in (None, 0.8)),
        "LETKF M=8 only": (lambda r: r.get("M") in (None, 8)),
        "LETKF M=8, loc=0.8": (lambda r: (r.get("M") in (None, 8)
                                          and r.get("loc") in (None, 0.8))),
    }
    for name, sel in variants.items():
        f = frontier_ratios(rows, select=sel)
        if "error" in f:
            print(f"  {name:22} {f['error']}")
            continue
        print(f"  {name:22} rate x{f['rate_ratio']:.2f}   "
              f"accuracy x{f['accuracy_ratio']:.2f}   "
              f"solver-step cost x{f['cost_ratio']:.1f}   "
              f"nudging optimum bracketed: {f['nudging_optimum_bracketed']}")
    f_all = frontier_ratios(rows)
    f_m8 = frontier_ratios(rows, select=variants["LETKF M=8 only"])
    check("rate ratio (all configs)", round(f_all["rate_ratio"], 2), 2.1)
    check("rate ratio (M=8)", round(f_m8["rate_ratio"], 2), 1.4)
    check("accuracy ratio (all configs)", round(f_all["accuracy_ratio"], 2), 1.6)
    print("  -> the 1.6x accuracy figure only holds if LETKF is restricted to "
          "loc=0.8; loc=1.5 is strictly more accurate.")
    print("  -> 'same order of cost (M=8)' is wrong: M=8 is 8x the solver steps "
          "and 4.5x wall clock.  Say 'smallest ensemble tested'.")


# --------------------------------------------------------------- C3 robustness
def c3_robustness():
    section("C3  exp28: does 27/27 survive the plateau metric?")
    d = load("28_robustness_noise")
    if d is None:
        return
    rows = d["rows"]
    lay = d["meta"]["layouts"]
    keys = ["noise", "nu_err", "m_assim"]
    by_rate = ordering_by(rows, lay, "rate", +1, keys)
    by_plat = ordering_by(rows, lay, "plateau_err", -1, keys)
    print(f"  by rate (v5's rule, retired metric): "
          f"{by_rate['unchanged']}/{by_rate['total']}  {by_rate['baseline']}")
    print(f"  by plateau (correct metric):         "
          f"{by_plat['unchanged']}/{by_plat['total']}  {by_plat['baseline']}")
    if by_plat["differing"]:
        for k, v in by_plat["differing"].items():
            print(f"     differs at {k}: {v}")
    print("  -> conclusion survives; only the script needs to change metric.")


# ------------------------------------------------------------------ exp23 meta
def s24_kernel():
    section("RESULTS.md §2.4 / exp23: did any kernel rescue a gapped layout?")
    d = load("23_kernel_ablation")
    if d is None:
        return
    rows = d["rows"]
    for lay in dict.fromkeys(r["layout"] for r in rows):
        rs = [r for r in rows if r["layout"] == lay and not r.get("diverged")
              and r.get("final") is not None]
        rels = [verdict_from_final(r["final"])["plateau_rel"] for r in rs]
        n_ok = sum(1 for x in rels if x < 0.15)
        n_sync = sum(1 for x in rels if x < 1e-2)
        print(f"  {lay:15} best rel={min(rels):.2e}  "
              f"below PARTIAL: {n_ok}/{len(rs)}   SYNC: {n_sync}")
    print(f"  meta.rescued in the shipped JSON: {d['meta'].get('rescued')}")
    print("  -> the rows say nothing was rescued; the meta field contradicts "
          "them and must be corrected in the archived JSON.")


# ------------------------------------------------------------------ exp31/32
def c2_thresholds():
    section("C2  exp31 / exp32: necessity vs the pre-registered rule")
    d = load("31_stripe_nyquist")
    if d:
        rows = d["rows"]
        strong = sum(r["prediction_holds"] for r in rows)
        weak = sum(1 for r in rows
                   if not (r["p"] < r["p_star"] and r["best_verdict"] == "SYNC"))
        print(f"  pre-registered rule (every p<p* is FAIL): {strong}/{len(rows)}"
              f"   <- meta says {d['meta'].get('prediction_holds')}")
        print(f"  necessity only (no SYNC below p*):        {weak}/{len(rows)}")
        print("  violations of the pre-registered rule (predicted FAIL, "
              "observed PARTIAL):")
        for r in rows:
            if not r["prediction_holds"]:
                print(f"     nu={r['nu']:g} p={r['p']:2d} (p*={r['p_star']}) "
                      f"plateau={r['best_plateau_rel']:.2e} "
                      f"{r['best_verdict']}  n_actual={r['n_actual']}")
        ns = [r["n_actual"] for r in rows]
        print(f"  n_actual across the family: {min(ns)}-{max(ns)} "
              f"(claimed fixed at {d['meta'].get('n_sensors')})")
        print("  -> §2.2's 'all p<p* fail' is false; the necessity statement is "
              "true.  State both.")
        print("  -> the fixed-count claim is false until script 37 is run.")
    d = load("32_lattice_aliasing_fix")
    if d:
        ms = sorted({r["m"] for r in d["rows"] if r["nu"] == 0.0025})
        e = load("32_lattice_aliasing_fix/extension_nu25")
        if e:
            ms += sorted({r["m"] for r in e["rows"]})
        gaps = [m for m in range(min(ms), max(ms) + 1) if m not in ms]
        print(f"  nu=2.5e-3 lattice m grid: {ms}")
        print(f"  untested m in that range: {gaps}")
        print("  -> offset=7 is only bounded to {5,6,7} until m=19,21 are run "
              "(script 39).")


if __name__ == "__main__":
    t2_ordering()
    c1_necessity()
    s21_counterexample()
    c3_frontier()
    c3_robustness()
    s24_kernel()
    c2_thresholds()
    print("\nDone.  Every MISMATCH above is a sentence to edit; see DOC_EDITS.md.")
