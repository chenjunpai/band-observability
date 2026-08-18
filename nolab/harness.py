"""Shared trial harness.

v2 fixes versus v1:
  * truth cache keys contain every physical parameter (alpha, dt, forcing,
    spin-up length, seed) -- v1's key was only (nu, N), so exp18 silently
    loaded the wrong state when a shared TRUTH_DIR was used;
  * results are written atomically with allow_nan=False and carry
    provenance (solver version + fingerprint);
  * `trial` returns the full metric dict plus layout diagnostics.
"""

import datetime
import json
import os
import sys

import numpy as np

from .solver import KolmogorovFlow, SOLVER_VERSION, solver_fingerprint
from .observer import run_observer
from .metrics import sync_rate


TRUTH_DIR = os.environ.get(
    "V2_TRUTH_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "truths"))


def truth_path(nu, N=128, alpha=0.1, n_forcing=4, dt=0.004,
               T_spin=30.0, seed=0):
    return os.path.join(
        TRUTH_DIR,
        f"truth_N{N}_nu{nu:.1e}_a{alpha:.1e}_nf{n_forcing}"
        f"_dt{dt:.3g}_spin{T_spin:g}_s{seed}.npy")


def get_truth(nu, N=128, alpha=0.1, n_forcing=4, dt=0.004, T_spin=30.0,
              seed=0, cfl=False):
    """Load or generate one spin-up; the cache key includes every parameter
    that changes the state."""
    os.makedirs(TRUTH_DIR, exist_ok=True)
    p = truth_path(nu, N, alpha, n_forcing, dt, T_spin, seed)
    flow = KolmogorovFlow(N=N, nu=nu, alpha=alpha, n_forcing=n_forcing,
                          dt=dt)
    if os.path.exists(p):
        w = np.load(p)
        if np.all(np.isfinite(w)):
            return flow, w
        os.remove(p)
    w = flow.spinup(T=T_spin, seed=seed, cfl=cfl)
    np.save(p, w)
    return flow, w


def trial(flow, w, obs, mu, T=8.0, **kw):
    """One observer trial -> full result dict (never NaN)."""
    ts, er, info = run_observer(flow, w, obs, mu, T=T, **kw)
    m = sync_rate(ts, er)
    cov = getattr(obs, "covering_radius", None)
    aniso = getattr(obs, "anisotropy", None)
    return dict(
        ndof=int(getattr(obs, "ndof", 0)),
        mu=float(mu), T=float(T),
        rate=float(m["rate"]), r2=float(m["r2"]),
        n_fit=int(m["n_fit"]), status=m["status"],
        converged=bool(m["converged"]),
        final=(None if info["final"] is None else float(info["final"])),
        diverged=bool(info["diverged"]), reason=info["reason"],
        init_norm_ratio=float(info["init_norm_ratio"]),
        covering_radius=(None if cov is None else float(cov)),
        anisotropy=(None if aniso is None else float(aniso)),
        ts=ts.tolist(), err=[float(x) for x in er if np.isfinite(x)],
    )


def provenance(extra=None):
    d = dict(
        solver_version=SOLVER_VERSION,
        solver_fingerprint=solver_fingerprint(),
        numpy=str(np.__version__),
        python=sys.version.split()[0],
        written_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    if extra:
        d.update(extra)
    return d


def save(rows, meta, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "results.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(meta=provenance(meta), rows=rows), f,
                  indent=2, allow_nan=False)
    os.replace(tmp, p)


def load_results(outdir="."):
    with open(os.path.join(outdir, "results.json")) as f:
        return json.load(f)


def logline(**kv):
    print("  " + "  ".join(f"{k}={v}" for k, v in kv.items()), flush=True)
