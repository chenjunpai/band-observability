"""The nudging observer loop.

v2 fixes versus v1:
  * the default observer initial condition is a random field with the same
    spectral norm as the truth -- identical to the LETKF initialisation, so
    the baselines start from the same difficulty (v1 nudging started from
    zero, which is strictly easier);
  * divergence is returned as a structured status, never as NaN in results;
  * model error, observation noise and temporal sparsity are explicit.
"""

import numpy as np


def random_observer_state(flow, wh_true, seed=0):
    """Smooth random field with the same spectral norm as the truth.
    Used by both the nudging observer (default) and the LETKF ensemble."""
    g = flow.g
    rng = np.random.default_rng(seed)
    wh = (g.fwd(rng.standard_normal((g.N, g.N)))
          * np.exp(-0.5 * g.k2 / 16.0) * g.dealias)
    norm = np.linalg.norm(wh_true)
    if norm > 0:
        wh *= norm / max(np.linalg.norm(wh), 1e-30)
    return wh


def run_observer(flow, wh0_true, obs, mu, T=12.0, wh0_obs=None,
                 record_every=10, flow_obs=None, noise=0.0, m_assim=1,
                 seed=0, init_seed=0, diverge_factor=1e6):
    """Run truth + nudging observer; return (ts, errs, info).

    info: dict(diverged, reason, final, init_norm_ratio)
    errs contains only finite values; divergence is reported in info.
    """
    g = flow.g
    fo = flow if flow_obs is None else flow_obs
    rng = np.random.default_rng(1000 + seed)
    wh_t = wh0_true.copy()
    if wh0_obs is None:
        wh_o = random_observer_state(flow, wh0_true, seed=init_seed)
    else:
        wh_o = wh0_obs.copy()
    init_norm_ratio = float(np.linalg.norm(wh_o) / max(np.linalg.norm(wh_t), 1e-30))
    nsteps = int(T / flow.dt)
    ts, errs = [], []
    info = dict(diverged=False, reason=None, final=None,
                init_norm_ratio=init_norm_ratio)
    for s in range(nsteps):
        wh_t_old = wh_t
        wh_t = flow.step(wh_t)
        if s % m_assim == 0:
            if noise > 0.0:
                amp = noise * np.sqrt(np.mean(g.inv(wh_t) ** 2))
                eta = g.fwd(rng.standard_normal((g.N, g.N)) * amp)
                yh_old, yh_new = wh_t_old + eta, wh_t + eta
            else:
                yh_old, yh_new = wh_t_old, wh_t
            n1 = lambda wh, _t=yh_old: -mu * obs.apply_h(wh - _t, wh)
            n2 = lambda wh, _t=yh_new: -mu * obs.apply_h(wh - _t, wh)
        else:
            n1 = n2 = None
        wh_o = fo.step(wh_o, nudge1=n1, nudge2=n2)
        if s % record_every == 0:
            num = np.linalg.norm(wh_o - wh_t)
            den = np.linalg.norm(wh_t)
            if not np.isfinite(num) or not np.isfinite(den) or den == 0:
                info.update(diverged=True, reason="non_finite_error")
                break
            e = float(num / den)
            ts.append((s + 1) * flow.dt)
            errs.append(e)
            if not np.isfinite(wh_o).all() or e > diverge_factor:
                info.update(diverged=True,
                            reason=("observer_nonfinite"
                                    if not np.isfinite(wh_o).all()
                                    else f"error_exceeded_{diverge_factor:g}"))
                break
    if errs:
        info["final"] = float(errs[-1])
    return np.array(ts), np.array(errs), info
