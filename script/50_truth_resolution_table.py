"""50 -- assemble the H4 verification table from the frozen 44/43 runs.

No dynamics: this only reads the existing per-truth and per-grid verdict tables
and writes a compact table for the paper.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "50_truth_resolution_table"


def main():
    os.makedirs(OUT, exist_ok=True)

    p44 = json.loads((ROOT / "results" / "44_truth_ensemble" / "results_T8_full.json").read_text())
    p43 = json.loads((ROOT / "results" / "43_resolution_decisive" / "results_merged_T8.json").read_text())

    truth = p44["meta"]["per_truth"]
    grad = p44["meta"]["gradient_scales"]

    truth_table = []
    for t in ("0", "1", "2"):
        s = truth[f"stripes_truth{t}"]
        l = truth[f"lattice_truth{t}"]
        truth_table.append(dict(
            truth=t,
            grad_Linf=grad[t]["sup_linf"],
            stripes=dict(verdicts=s["verdicts"], first_sync=s["first_sync"],
                         offset=s["offset"], necessity_ok=s["necessity_ok"]),
            lattice=dict(verdicts=l["verdicts"], first_sync=l["first_sync"],
                         offset=l["offset"], necessity_ok=l["necessity_ok"])))

    res_table = []
    for r in p43["rows"]:
        res_table.append(dict(N_grid=r["N_grid"], p=r["p"], p_star=r["p_star"],
                              delta_K=r["delta_K"], verdict=r["verdict"],
                              bestmu_median=r["aggregation"]["bestmu_median"],
                              best_of_all=r["aggregation"]["best_of_all"]))

    meta = dict(
        nu_truth=0.005, K_c=5, threshold=11,
        nu_resolution=0.0025, K_c_res=7, p_star_res=15,
        necessity_on_every_truth=p44["meta"]["necessity_on_every_truth"],
        offsets_across_truths=p44["meta"]["offsets_across_truths"],
        offsets_n_measured=p44["meta"]["offsets_n_measured"],
        resolution_summary=p43["meta"]["summary"],
        headline_aggregation="min over mu of the median over observer inits",
        note=("H4: the necessary bound holds on all three sampled truth fields "
              "and both resolutions; the sufficiency offset is trajectory- and "
              "resolution-dependent."))

    (OUT / "results.json").write_text(json.dumps(dict(meta=meta,
                                                       truth_table=truth_table,
                                                       resolution_table=res_table), indent=1))
    print("wrote", OUT / "results.json")
    print("necessity_on_every_truth =", meta["necessity_on_every_truth"])
    print("offsets_across_truths =", meta["offsets_across_truths"])
    for r in res_table:
        print(f"  N={r['N_grid']:>3} p={r['p']:>2}  {r['verdict']:<8} "
              f"median={r['bestmu_median']:.2e}")


if __name__ == "__main__":
    main()
