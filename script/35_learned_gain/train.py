"""Train the learned observer (exp14, v2).

Every P1 audit point is addressed here:
  * P1-3: train/val/test come from different spin-up seeds (make_dataset)
    and the best checkpoint is chosen on the VALIDATION loss, never on the
    training EMA;
  * P1-4: the loss uses GLOBAL time (t0 passed per window) and defaults to
    the log-mean mode, so it is a true rate objective;
  * P1-2: --denom_floor is explicit and matches the numpy experiments;
  * sensor indices are de-duplicated, so n_unique == requested n;
  * the classical (eps=0) baseline rate is recorded alongside the learned
    rate during validation, on the same layout and truth.

    python train.py --epochs 200
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from nolab import sync_rate
from solver_torch import TorchFlow
from model import NeuralObserverGain, rate_loss

HERE = Path(__file__).resolve().parent


def sensor_batch(N, n, rng, device, dtype):
    """Uniform continuous positions, accumulated until EXACTLY n unique
    grid cells are occupied (v1 sampled n and hoped for no collisions)."""
    xys, sids = [], []
    seen = set()
    while len(xys) < n:
        m = max(n * 2, 64)
        xy = torch.tensor(rng.uniform(0, 2 * np.pi, size=(m, 2)),
                          dtype=dtype, device=device)
        ix = (xy[:, 0] * N / (2 * np.pi)).round().long() % N
        iy = (xy[:, 1] * N / (2 * np.pi)).round().long() % N
        sidx = (ix * N + iy).tolist()
        for k in range(m):
            if sidx[k] not in seen:
                seen.add(sidx[k])
                xys.append(xy[k])
                sids.append(sidx[k])
                if len(xys) >= n:
                    break
    xy = torch.stack(xys[:n])
    sidx = torch.tensor(sids[:n], dtype=torch.long, device=device)
    return xy, sidx


def spread_operators(flow, sensor_idx, n, width_factor=1.0, floor=0.25):
    N = flow.N
    mask = torch.zeros(N * N, dtype=flow.dtype, device=flow.kx.device)
    mask[sensor_idx] = 1.0
    mask = mask.reshape(N, N)
    h = width_factor * flow.L / np.sqrt(n)
    kernel = torch.exp(-0.5 * (h ** 2) * flow.k2).to(flow.cdtype)
    d = torch.fft.ifft2(torch.fft.fft2(mask.to(flow.cdtype)) * kernel).real
    return mask, kernel, torch.clamp(d, min=floor * d.mean())


def rollout(flow, model, wh_true_seq, xy, sensor_idx, mask, kernel, denom,
            wh_obs, window, use_learned=True, detach=True):
    """One truncated-BPTT window; returns (errors, final observer state)."""
    errs = []
    for s in range(window):
        wt_old, wt_new = wh_true_seq[s], wh_true_seq[s + 1]

        def nudge(wh, target=wt_old):
            resid = torch.fft.ifft2(wh - target).real
            return -model(flow, xy[None], sensor_idx, mask, kernel, denom,
                          resid, use_learned=use_learned)

        def nudge2(wh, target=wt_new):
            resid = torch.fft.ifft2(wh - target).real
            return -model(flow, xy[None], sensor_idx, mask, kernel, denom,
                          resid, use_learned=use_learned)

        wh_obs = flow.step(wh_obs, nudge1=nudge, nudge2=nudge2)
        errs.append((wh_obs - wt_new).norm() / wt_new.norm())
    e = torch.stack(errs)[None]
    return e, (wh_obs.detach() if detach else wh_obs)


def tensor_tree_finite(state):
    if torch.is_tensor(state):
        return bool(torch.isfinite(state).all().item())
    if isinstance(state, dict):
        return all(tensor_tree_finite(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return all(tensor_tree_finite(v) for v in state)
    return True


def params_finite(model):
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def grads_finite(model):
    return all(p.grad is None or torch.isfinite(p.grad).all().item()
               for p in model.parameters())


def effective_nmin(epoch, target_nmin, start_nmin, curriculum_epochs):
    start = max(int(start_nmin), int(target_nmin))
    if curriculum_epochs <= 0 or start == target_nmin:
        return int(target_nmin)
    frac = min(max(epoch, 0) / float(curriculum_epochs), 1.0)
    return int(round(start + frac * (target_nmin - start)))


@torch.no_grad()
def evaluate(flow, model, traj_np, n_sensors, T, denom_floor, use_learned,
             seed=0, record_every=20):
    """Roll out from a random state against a held-out trajectory."""
    N = flow.N
    dev = flow.kx.device
    rng = np.random.default_rng(seed)
    xy, sidx = sensor_batch(N, n_sensors, rng, dev, flow.dtype)
    mask, kernel, denom = spread_operators(flow, sidx, n_sensors,
                                           floor=denom_floor)
    truth0 = torch.as_tensor(traj_np[0], dtype=flow.dtype, device=dev)
    wt = torch.fft.fft2(truth0.to(flow.cdtype))[None]
    f = torch.fft.fft2(torch.tensor(rng.standard_normal((N, N)),
                                    dtype=flow.dtype, device=dev)
                       .to(flow.cdtype))
    wh_obs = (f * torch.exp(-0.5 * flow.k2 / 16.0).to(flow.cdtype)
              * flow.dealias.to(flow.cdtype))[None]
    wh_obs = wh_obs * (wt.norm() / wh_obs.norm())
    nsteps = int(T / flow.dt)
    ts, errs = [], []
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
            return dict(mean_log_err=float("inf"), rate=0.0, r2=0.0,
                        status="no_fit", converged=False, final=None,
                        diverged=True)
        if s % record_every == 0:
            err = float((wh_obs - wt).norm() / wt.norm())
            ts.append((s + 1) * flow.dt)
            errs.append(err)
    m = sync_rate(np.array(ts), np.array(errs))
    return dict(mean_log_err=float(np.log(np.clip(np.array(errs), 1e-30,
                                                  1e30)).mean()),
                rate=m["rate"], r2=m["r2"], status=m["status"],
                converged=m["converged"],
                final=float(errs[-1]) if errs else None,
                diverged=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--nu", type=float, default=5e-3)
    ap.add_argument("--dt", type=float, default=None,
                    help="default: read DT from 01_calibrate, else 0.004")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--nmin", type=int, default=200)
    ap.add_argument("--nmax", type=int, default=800)
    ap.add_argument("--curriculum_start", type=int, default=400)
    ap.add_argument("--curriculum_epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_backoff", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_consecutive_skips", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--loss_mode", choices=["log", "exp"], default="log")
    ap.add_argument("--mu_init", type=float, default=20.0)
    ap.add_argument("--mu_min", type=float, default=1.0)
    ap.add_argument("--mu_max", type=float, default=100.0)
    ap.add_argument("--width_min", type=float, default=0.02)
    ap.add_argument("--width_max", type=float, default=1.0)
    ap.add_argument("--denom_floor", type=float, default=0.25)
    ap.add_argument("--dtype", choices=["float32", "float64"],
                    default="float64")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_modes", type=int, default=24)
    ap.add_argument("--val_n", type=int, default=400)
    ap.add_argument("--val_T", type=float, default=6.0)
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--ckpt-dir", default=str(HERE / "ckpt"))
    a = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    if str(dev).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    dt = a.dt
    if dt is None:
        pcal = ROOT / "results" / "01_calibrate" / "calibrate.json"
        if pcal.exists():
            dt = json.loads(pcal.read_text())["recommendations"]["DT"]
        else:
            dt = 0.004
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    train_path = os.path.join(a.data_dir, "truth_traj.npy")
    val_path = os.path.join(a.data_dir, "val_traj.npy")
    for p in (train_path, val_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}; run make_dataset.py first")
    traj = np.load(train_path, mmap_mode="r")
    val = np.load(val_path, mmap_mode="r")
    if traj.ndim != 4 or traj.shape[2:] != (a.N, a.N):
        raise ValueError(f"truth shape {traj.shape} incompatible N={a.N}")
    needed = a.windows * a.window + 1
    if traj.shape[1] < needed:
        raise ValueError(
            f"dataset has {traj.shape[1]} frames; need {needed}; "
            "increase --steps in make_dataset.py")
    if a.window * dt < 0.2:
        print("WARNING: window too short to see O(1) decay; "
              "increase --window", flush=True)

    flow = TorchFlow(N=a.N, nu=a.nu, dt=dt, device=dev, dtype=dtype)
    model = NeuralObserverGain(
        N=a.N, eps=a.eps, n_modes=a.n_modes, mu_init=a.mu_init, dtype=dtype,
        mu_min=a.mu_min, mu_max=a.mu_max,
        width_min=a.width_min, width_max=a.width_max).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    ckpt_last = os.path.join(a.ckpt_dir, "ckpt.pt")
    ckpt_best = os.path.join(a.ckpt_dir, "ckpt_best.pt")
    train_log, val_log = [], []
    best_val = float("inf")
    consecutive_skips = 0

    for ep in range(a.epochs):
        t0 = time.time()
        model_before = copy.deepcopy(model.state_dict())
        opt_before = copy.deepcopy(opt.state_dict())
        j = int(rng.integers(0, traj.shape[0]))
        max_start = traj.shape[1] - needed
        t_start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        nmin_eff = effective_nmin(ep, a.nmin, a.curriculum_start,
                                  a.curriculum_epochs)
        n = int(rng.integers(nmin_eff, a.nmax + 1))
        xy, sidx = sensor_batch(a.N, n, rng, dev, dtype)
        n_unique = int(torch.unique(sidx).numel())
        mask, kernel, denom = spread_operators(flow, sidx, n,
                                               floor=a.denom_floor)
        f = torch.fft.fft2(torch.tensor(rng.standard_normal((a.N, a.N)),
                                        dtype=dtype, device=dev)
                           .to(flow.cdtype))
        wh_obs = (f * torch.exp(-0.5 * flow.k2 / 16.0).to(flow.cdtype)
                  * flow.dealias.to(flow.cdtype))[None]
        truth0_np = np.array(traj[j, t_start], copy=True)
        truth0 = torch.as_tensor(truth0_np, dtype=dtype, device=dev)
        truth0_h = torch.fft.fft2(truth0.to(flow.cdtype))
        wh_obs = wh_obs * (truth0_h.norm() / wh_obs.norm())

        tot, failed_reason, last_e = 0.0, None, None
        for wdx in range(a.windows):
            s0 = t_start + wdx * a.window
            seg_np = np.array(traj[j, s0:s0 + a.window + 1], copy=True)
            seg = torch.as_tensor(seg_np, dtype=dtype, device=dev)
            sl = torch.fft.fft2(seg.to(flow.cdtype))[:, None]
            e, wh_obs = rollout(flow, model, sl, xy, sidx, mask, kernel,
                                denom, wh_obs, a.window)
            last_e = e
            t0_global = (t_start + wdx * a.window) * flow.dt
            loss = rate_loss(e, flow.dt, t0=t0_global, gamma=a.gamma,
                             mode=a.loss_mode)
            if (not torch.isfinite(e).all()
                    or not torch.isfinite(wh_obs).all()
                    or not torch.isfinite(loss)):
                failed_reason = f"nonfinite_forward_window_{wdx}"
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not grads_finite(model):
                failed_reason = f"nonfinite_gradient_window_{wdx}"
                break
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                a.grad_clip)
            if not torch.isfinite(torch.as_tensor(gn)):
                failed_reason = f"nonfinite_gradnorm_window_{wdx}"
                break
            opt.step()
            if not params_finite(model):
                failed_reason = f"nonfinite_parameter_window_{wdx}"
                break
            tot += float(loss.detach().cpu())

        if failed_reason is not None:
            model.load_state_dict(model_before)
            opt.load_state_dict(opt_before)
            for pg in opt.param_groups:
                pg["lr"] = max(float(pg["lr"]) * a.lr_backoff, a.min_lr)
            consecutive_skips += 1
            rec = dict(epoch=ep, status="skipped_nonfinite",
                       reason=failed_reason, n_sensors=n,
                       n_unique=n_unique, nmin_effective=nmin_eff,
                       mu=float(model.mu().detach().cpu()),
                       lr=float(opt.param_groups[0]["lr"]),
                       sec=time.time() - t0)
            train_log.append(rec)
            print("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float)
                                  else f"{k}={v}" for k, v in rec.items()),
                  flush=True)
            torch.save(dict(model=model.state_dict(),
                            optimizer=opt.state_dict(), args=vars(a),
                            epoch=ep, train_log=train_log,
                            val_log=val_log,
                            best_val=float(best_val)), ckpt_last)
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
            if consecutive_skips >= a.max_consecutive_skips:
                print(f"stopping after {consecutive_skips} consecutive "
                      f"non-finite epochs", flush=True)
                break
            continue

        consecutive_skips = 0
        epoch_loss = tot / a.windows
        rec = dict(epoch=ep, status="ok", loss=epoch_loss,
                   n_sensors=n, n_unique=n_unique, nmin_effective=nmin_eff,
                   mu=float(model.mu().detach().cpu()),
                   lr=float(opt.param_groups[0]["lr"]),
                   final_err=float(last_e[0, -1].detach().cpu()),
                   sec=time.time() - t0)
        train_log.append(rec)
        print("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float)
                              else f"{k}={v}" for k, v in rec.items()),
              flush=True)

        if a.save_every > 0 and ep % a.save_every == 0:
            torch.save(dict(model=model.state_dict(),
                            optimizer=opt.state_dict(), args=vars(a),
                            epoch=ep, train_log=train_log,
                            val_log=val_log, best_val=float(best_val)),
                       ckpt_last)

        if ep % a.val_every == 0:
            learned = evaluate(flow, model, np.array(val[0], copy=True),
                               a.val_n, a.val_T, a.denom_floor, True)
            classical = evaluate(flow, model, np.array(val[0], copy=True),
                                 a.val_n, a.val_T, a.denom_floor, False)
            vrec = dict(epoch=ep, learned=learned, classical=classical)
            val_log.append(vrec)
            print(f"  [val] learned rate={learned['rate']:.3f} "
                  f"status={learned['status']} | classical "
                  f"rate={classical['rate']:.3f} status={classical['status']}",
                  flush=True)
            if learned["mean_log_err"] < best_val:
                best_val = learned["mean_log_err"]
                torch.save(dict(model=model.state_dict(),
                                optimizer=opt.state_dict(), args=vars(a),
                                epoch=ep, train_log=train_log,
                                val_log=val_log,
                                best_val=float(best_val)),
                           ckpt_best)
                print(f"  [val] new best (val mean_log_err={best_val:.4f})",
                      flush=True)

    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                    args=vars(a), epoch=a.epochs - 1,
                    train_log=train_log, val_log=val_log,
                    best_val=float(best_val)), ckpt_last)
    with open(os.path.join(a.ckpt_dir, "train_log.json"), "w") as f:
        json.dump(dict(train_log=train_log, val_log=val_log, args=vars(a)),
                  f, indent=2, allow_nan=False)
    print("done; best_val=", best_val)


if __name__ == "__main__":
    main()
