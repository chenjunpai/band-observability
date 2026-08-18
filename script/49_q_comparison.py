"""49 -- add the classical separation distance q to the Fig 1 ranking table.

No dynamics: q is computed from the same layout generators used by exp19/33/41,
and ranked with the same parameter-free statistics as delta_K and h.

The result is reported as-is.  On this dense-random-layout set the minimum
pairwise separation q is dominated by accidental near-coincidences and is
uninformative (AUC near 0.5), so it does not challenge delta_K.
"""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from nolab import GENERATORS
from nolab.configs_fix import corridor, lattice_m
from nolab.stripes_v2 import stripes_v2
from nolab.ranking import predictor_quality

OUT = ROOT / "results" / "49_q_comparison"
N = int(os.environ.get("N_GRID", 128))
N_SENS = int(os.environ.get("N_SENS", 784))


def sep_q(ix, iy):
    """Minimum periodic pairwise distance between distinct sensors."""
    x = np.asarray(ix, float) * 2 * np.pi / N
    y = np.asarray(iy, float) * 2 * np.pi / N
    n = len(x)
    best = np.inf
    for i in range(n - 1):
        dx = np.abs(x[i + 1:] - x[i])
        dy = np.abs(y[i + 1:] - y[i])
        dx = np.minimum(dx, 2 * np.pi - dx)
        dy = np.minimum(dy, 2 * np.pi - dy)
        d = np.hypot(dx, dy)
        if d.size:
            best = min(best, float(d.min()))
    return float(best) if np.isfinite(best) else None


def nn_dists(ix, iy):
    """Per-sensor nearest-neighbour distances on the periodic box."""
    x = np.asarray(ix, float) * 2 * np.pi / N
    y = np.asarray(iy, float) * 2 * np.pi / N
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        dx = np.abs(x - x[i])
        dy = np.abs(y - y[i])
        dx = np.minimum(dx, 2 * np.pi - dx)
        dy = np.minimum(dy, 2 * np.pi - dy)
        d = np.hypot(dx, dy)
        d[i] = np.inf
        out[i] = d.min()
    return out


def build(row):
    fam = row["family"]
    label = row["label"]
    seed = row.get("seed") or 0
    if fam == "uniform":
        n = int(label.split("_n")[1])
        ix, iy = GENERATORS["uniform"](N, n, seed=seed)
    elif fam == "corridor":
        gw = float(label.split("_gw")[1])
        ix, iy, _ = corridor(N, N_SENS, gw, seed=seed)
    elif fam == "lattice":
        m = int(label.split("_m")[1])
        ix, iy, _ = lattice_m(N, m)
    elif fam == "stripes_v2":
        p = int(label.split("_p")[1])
        ix, iy, _ = stripes_v2(N, N_SENS, p, seed=0)
    else:
        raise ValueError(f"unknown family {fam}")
    return ix, iy


def main():
    os.makedirs(OUT, exist_ok=True)
    src = json.loads((ROOT / "results" / "42_unify_aggregation" / "results.json").read_text())
    rows = src["rows"]["fig1"]
    flags = {r["label"]: bool(r.get("resolution_flag")) for r in rows}

    out = []
    for r in rows:
        ix, iy = build(r)
        q = sep_q(ix, iy)
        nn = nn_dists(ix, iy)
        out.append(dict(label=r["label"], family=r["family"], seed=r.get("seed"),
                        q=q, h=r["h"], delta_K_floor0=r["delta_K_floor0"],
                        q_median=float(np.median(nn)),
                        q_p10=float(np.percentile(nn, 10)),
                        best_plateau_rel=r["best_plateau_rel"],
                        best_verdict=r["best_verdict"],
                        resolution_flag=flags[r["label"]]))

    quality = {}
    for pred, sgn in (("delta_K_floor0", +1), ("h", -1), ("q", +1),
                      ("q_median", +1), ("q_p10", +1)):
        quality[pred] = predictor_quality(out, pred, sgn)

    noflag = [r for r in out if not r["resolution_flag"]]
    quality_nocorridor = {}
    for pred, sgn in (("delta_K_floor0", +1), ("h", -1), ("q", +1),
                      ("q_median", +1), ("q_p10", +1)):
        quality_nocorridor[pred] = predictor_quality(noflag, pred, sgn)

    meta = dict(N=N, N_SENS=N_SENS, n_rows=len(out),
                q_definition="minimum periodic pairwise distance between sensors",
                q_median_definition="median per-sensor nearest-neighbour distance",
                q_p10_definition="10th-percentile per-sensor nearest-neighbour distance",
                quality=quality, quality_without_corridor=quality_nocorridor,
                note=("q is quantised to the spectral grid: 59/64 layouts share "
                      "q=2*pi/128; bulk separation statistics are also uninformative"))

    (OUT / "results.json").write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
    print("wrote", OUT / "results.json")
    for pred in ("delta_K_floor0", "h", "q", "q_median", "q_p10"):
        qq = quality[pred]
        print(f"  {pred:<16} tau={qq['tau']:+.3f} auc={qq['auc']:.3f}")
    print("without corridor rows:")
    for pred in ("delta_K_floor0", "h", "q", "q_median", "q_p10"):
        qq = quality_nocorridor[pred]
        print(f"  {pred:<16} tau={qq['tau']:+.3f} auc={qq['auc']:.3f}")


if __name__ == "__main__":
    main()
