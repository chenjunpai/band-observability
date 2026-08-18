"""Build the exp14 dataset with train/val/test split on DIFFERENT
spin-up seeds (P1-3 fix: v1 trained and evaluated on the same seed-0
trajectory).

    python make_dataset.py

Outputs into exp14/data/:
    truth_traj.npy   (n_train, steps+1, N, N)  float32 physical vorticity
    val_traj.npy     (n_val, ...)
    test_traj.npy    (n_test, ...)
    splits.json      seeds + parameters + provenance
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from nolab import KolmogorovFlow


def write_traj(flow, out_path, n_traj, steps, spin, decorr, seed_base,
               dtype):
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype,
        shape=(n_traj, steps + 1, flow.g.N, flow.g.N))
    seeds = []
    for j in range(n_traj):
        seed = seed_base + j
        w = flow.spinup(T=spin, seed=seed)
        for _ in range(int(decorr / flow.dt)):
            w = flow.step(w)
        ww = w.copy()
        arr[j, 0] = flow.g.inv(ww)
        for s in range(steps):
            ww = flow.step(ww)
            arr[j, s + 1] = flow.g.inv(ww)
        arr.flush()
        seeds.append(seed)
        print(f"  trajectory {j} (seed {seed}) done", flush=True)
    arr.flush()
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-val", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4200)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--nu", type=float, default=5e-3)
    ap.add_argument("--spin", type=float, default=30.0)
    ap.add_argument("--decorr", type=float, default=None,
                    help="default: read from 01_calibrate, else 8.0")
    ap.add_argument("--dt", type=float, default=None,
                    help="default: read DT from 01_calibrate, else 0.004")
    ap.add_argument("--dtype", choices=["float32", "float64"],
                    default="float32")
    a = ap.parse_args()

    decorr = a.decorr
    dt = a.dt
    if decorr is None:
        pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
        if pcal.exists():
            rec = json.loads(pcal.read_text())["recommendations"]
            decorr = rec["DECORR"]
            if dt is None:
                dt = rec["DT"]
        else:
            decorr = 8.0
    if dt is None:
        dt = 0.004
    os.makedirs(a.out_dir, exist_ok=True)
    flow = KolmogorovFlow(N=a.N, nu=a.nu, dt=dt)
    np_dtype = np.float32 if a.dtype == "float32" else np.float64
    seeds_train = write_traj(flow, os.path.join(a.out_dir, "truth_traj.npy"),
                             a.n_train, a.steps, a.spin, decorr, 1, np_dtype)
    seeds_val = write_traj(flow, os.path.join(a.out_dir, "val_traj.npy"),
                           a.n_val, a.steps, a.spin, decorr, 1000, np_dtype)
    seeds_test = write_traj(flow, os.path.join(a.out_dir, "test_traj.npy"),
                            a.n_test, a.steps, a.spin, decorr, 2000, np_dtype)
    splits = dict(
        train=dict(file="truth_traj.npy", seeds=seeds_train),
        val=dict(file="val_traj.npy", seeds=seeds_val),
        test=dict(file="test_traj.npy", seeds=seeds_test),
        params=dict(N=a.N, nu=a.nu, spin=a.spin, decorr=decorr,
                    steps=a.steps, dtype=a.dtype, dt=dt),
    )
    with open(os.path.join(a.out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2, allow_nan=False)
    for name in ("truth_traj.npy", "val_traj.npy", "test_traj.npy"):
        p = os.path.join(a.out_dir, name)
        print(name, os.path.getsize(p) / 1e6, "MB")


if __name__ == "__main__":
    main()
