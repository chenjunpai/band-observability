"""Exact point-sampling band Gram spectrum: verify dim ker = (2K+1)(2K+1-p)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import save
from nolab.stripes_v2 import stripes_v2

N = int(os.environ.get("N", 128))
N_SENS = int(os.environ.get("N_SENS", 784))
K = int(os.environ.get("K", 5))
PS = [int(x) for x in os.environ.get("PS", "8,9,10,11,12,14").split(",")]
TOL = float(os.environ.get("TOL", 1e-10))


def band_gram(N, ix, iy, K):
    ks = [(kx, ky) for kx in range(-K, K + 1) for ky in range(-K, K + 1)]
    x = 2 * np.pi * np.asarray(ix, float) / N
    y = 2 * np.pi * np.asarray(iy, float) / N
    E = np.empty((len(ix), len(ks)), dtype=np.complex128)
    for j, (kx, ky) in enumerate(ks):
        E[:, j] = np.exp(1j * (kx * x + ky * y)) / np.sqrt(len(ix))
    G = E.conj().T @ E
    ev = np.linalg.eigvalsh(G)
    return float(ev[0]), float(ev[-1]), int(np.sum(ev < TOL)), ks


def main():
    rows = []
    print(f"{'p':>4}{'n':>6}{'lambda_min':>14}{'lambda_max':>12}{'null_dim':>9}  formula")
    for p in PS:
        ix, iy, meta = stripes_v2(N, N_SENS, p, seed=0)
        lo, hi, null, _ = band_gram(N, ix, iy, K)
        formula = (2 * K + 1) * (2 * K + 1 - p) if p <= 2 * K else 0
        rows.append(dict(p=p, n=int(meta["n_actual"]), K=K,
                         band_dim=(2 * K + 1) ** 2, lambda_min=lo,
                         lambda_max=hi, null_dim=null, formula=formula,
                         tol=TOL, thickness=int(meta["thickness"]),
                         deficit=int(meta["deficit"])))
        print(f"{p:>4}{meta['n_actual']:>6}{lo:>14.2e}{hi:>12.4f}{null:>9}  "
              f"(2K+1)(2K+1-p)={formula}")
    ok = all(r["null_dim"] == r["formula"] for r in rows)
    print(f"\nmatch formula on all p: {ok}")
    save(rows, dict(N=N, n_sensors=N_SENS, K=K, band_dim=(2 * K + 1) ** 2,
                    ps=PS, tol=TOL, generator="nolab.stripes_v2 (rowshift)",
                    match_formula=bool(ok),
                    proposition=("dim ker = (2K+1)(2K+1-R) for R<=2K, 0 for "
                                 "R>=2K+1, where R = number of DISTINCT sensor "
                                 "rows; R=p iff thickness=1 (p >= ceil(n/N)=7). "
                                 "machine-precision 33/22/11/0"),
                    row_count_premise=("R = distinct rows; for p-stripe layout "
                                       "R=p iff thickness=1 (p >= ceil(n/N)=7)")),
         ROOT / "results" / "45_band_gram_spectrum")
    print("wrote", ROOT / "results" / "45_band_gram_spectrum" / "results.json")


if __name__ == "__main__":
    main()
