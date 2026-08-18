"""Zero-shot evaluation for exp14, v2.

P1-3 fix: evaluation uses the held-out test trajectory (seed 2000), never
the training seed-0 trajectory.
P0-2/P1-2 fix: the classical baseline is the SAME model with
use_learned=False and the same denom_floor -- side by side, same truth,
same initial state.
The 'continuous coordinates' selling point is tested with an off-grid
uniform layout as well as the grid-aligned layouts.

    python eval_zeroshot.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from nolab import sync_rate, GENERATORS
from solver_torch import TorchFlow
from model import NeuralObserverGain
from train import spread_operators, sensor_batch, tensor_tree_finite

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def run_learned(flow, model, traj, xy, sidx, T, use_learned, denom_floor,
                init_seed=0, record_every=20):
    N = flow.N
    dev = flow.kx.device
    mask, kernel, denom = spread_operators(flow, sidx, len(sidx),
                                           floor=denom_floor)
    nsteps = int(T / flow.dt)
    wt = torch.fft.fft2(torch.as_tensor(traj[0], dtype=flow.dtype,
                                        device=dev).to(flow.cdtype))[None]
    rng = np.random.default_rng(init_seed)
    f = torch.fft.fft2(torch.tensor(rng.standard_normal((N, N)),
                                    dtype=flow.dtype, device=dev)
                       .to(flow.cdtype))
    wh_obs = (f * torch.exp(-0.5 * flow.k2 / 16.0).to(flow.cdtype)
              * flow.dealias.to(flow.cdtype))[None]
    wh_obs = wh_obs * (wt.norm() / wh_obs.norm())
    ts, errs = [], []
    info = dict(diverged=False, reason=None)
    for s in range(nsteps):
        wt_old = wt
        wt = flow.step(wt)

        def n1(wh, tg=wt_old):
            resid = torch.fft.ifft2(wh - tg).real
            return -model(flow, xy[None], sidx, mask, kernel, denom, resid,
                          use_learned=use_learned)

        def n2(wh, tg=wt):
            resid = torch.fft.ifft2(wh - tg).real
            return -model(flow, xy[None], sidx, mask, kernel, denom, resid,
                          use_learned=use_learned)

        wh_obs = flow.step(wh_obs, nudge1=n1, nudge2=n2)
        if not torch.isfinite(wh_obs).all():
            info.update(diverged=True,
                        reason=f"observer_nonfinite_step_{s}")
            break
        if s % record_every == 0:
            err = float((wh_obs - wt).norm() / wt.norm())
            if not np.isfinite(err):
                info.update(diverged=True,
                            reason=f"error_nonfinite_step_{s}")
                break
            ts.append((s + 1) * flow.dt)
            errs.append(err)
    return np.array(ts), np.array(errs), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "ckpt_best.pt"))
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--nu", type=float, default=5e-3)
    ap.add_argument("--dt", type=float, default=None,
                    help="default: read DT from 01_calibrate, else 0.004")
    ap.add_argument("--T", type=float, default=8.0)
    ap.add_argument("--sensors", default="200,400,600,784")
    ap.add_argument("--denom_floor", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(HERE / "zeroshot.json"))
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {a.ckpt}")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not tensor_tree_finite(ck.get("model", {})):
        raise RuntimeError("checkpoint contains NaN/Inf; refuse to evaluate")
    ckargs = ck.get("args", {})
    dtype_name = ckargs.get("dtype", "float64")
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    model = NeuralObserverGain(
        N=a.N,
        eps=ckargs.get("eps", 0.5),
        n_modes=ckargs.get("n_modes", 24),
        mu_init=ckargs.get("mu_init", 20.0),
        mu_min=ckargs.get("mu_min", 1.0),
        mu_max=ckargs.get("mu_max", 100.0),
        width_min=ckargs.get("width_min", 0.02),
        width_max=ckargs.get("width_max", 1.0),
        dtype=dtype).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    dt = a.dt
    if dt is None:
        pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
        if pcal.exists():
            dt = json.loads(pcal.read_text())["recommendations"]["DT"]
        else:
            dt = 0.004
    flow = TorchFlow(N=a.N, nu=a.nu, dt=dt, device=dev, dtype=dtype)

    test_path = os.path.join(a.data_dir, "test_traj.npy")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"missing {test_path}; run make_dataset.py")
    test = np.load(test_path, mmap_mode="r")
    traj = np.array(test[0], copy=True)
    nsteps = int(a.T / flow.dt)
    if test.shape[1] < nsteps + 1:
        raise ValueError(f"test trajectory too short ({test.shape[1]} "
                         f"frames, need {nsteps + 1})")

    rows = []
    for name, gen in GENERATORS.items():
        for n in [int(x) for x in a.sensors.split(",")]:
            ix, iy = gen(a.N, n, seed=0)
            xy = torch.tensor(np.stack([ix, iy], 1) * 2 * np.pi / a.N,
                              dtype=flow.dtype, device=dev)
            sidx = torch.tensor(ix * a.N + iy, dtype=torch.long, device=dev)
            for method, use_learned in [("learned", True),
                                        ("classical", False)]:
                ts, er, info = run_learned(flow, model, traj, xy, sidx,
                                           a.T, use_learned, a.denom_floor)
                m = sync_rate(ts, er)
                rows.append(dict(
                    config=name, n_requested=n, n_actual=int(len(ix)),
                    method=method, rate=m["rate"], r2=m["r2"],
                    status=m["status"], converged=m["converged"],
                    final=(float(er[-1]) if er.size and np.isfinite(er[-1])
                           else None),
                    diverged=info["diverged"], reason=info["reason"],
                    in_distribution=bool(name == "uniform"
                                         and 200 <= n <= 800
                                         and method == "learned")))
                print("  ", rows[-1], flush=True)

    # off-grid uniform: continuous coordinates, rounded only for reads
    for n in [int(x) for x in a.sensors.split(",")]:
        xy_c, sidx = sensor_batch(a.N, n, np.random.default_rng(0),
                                  dev, flow.dtype)
        for method, use_learned in [("learned", True),
                                    ("classical", False)]:
            ts, er, info = run_learned(flow, model, traj, xy_c, sidx,
                                       a.T, use_learned, a.denom_floor)
            m = sync_rate(ts, er)
            rows.append(dict(
                config="uniform_offgrid", n_requested=n, n_actual=n,
                method=method, rate=m["rate"], r2=m["r2"],
                status=m["status"], converged=m["converged"],
                final=(float(er[-1]) if er.size and np.isfinite(er[-1])
                       else None),
                diverged=info["diverged"], reason=info["reason"],
                in_distribution=bool(200 <= n <= 800
                                     and method == "learned")))
            print("  ", rows[-1], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(dict(
            meta=dict(ckpt=a.ckpt, checkpoint_epoch=ck.get("epoch"),
                      nu=a.nu, T=a.T, device=str(dev), dtype=dtype_name,
                      denom_floor=a.denom_floor),
            rows=rows), f, indent=2, allow_nan=False)
    os.replace(a.out + ".tmp", a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
