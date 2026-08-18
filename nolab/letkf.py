"""LETKF baseline, made fair.

v2 fixes versus v1 (P0-5):
  * ensemble initialisation matches the nudging observer: smooth random
    fields with the same norm as the truth;
  * unobserved local blocks are NOT inflated by default
    (`inflate_unobserved=False`); v1 multiplied them by 1.05 per cycle,
    which is the reason clustered layouts always diverged;
  * RTPP and RTPS relaxation are implemented as alternatives to plain
    multiplicative inflation;
  * divergence is an experimental outcome, never a NaN.
"""

import numpy as np


class LETKFDivergence(RuntimeError):
    pass


def gaspari_cohn(r):
    """Gaspari-Cohn taper with support normalised to r <= 1."""
    r = np.abs(np.asarray(r, float))
    z = 2.0 * r
    out = np.zeros_like(z)
    a = z <= 1.0
    b = (z > 1.0) & (z <= 2.0)
    x = z[a]
    out[a] = (-(x ** 5) / 4 + (x ** 4) / 2 + (x ** 3) * 5 / 8
              - (x ** 2) * 5 / 3 + 1)
    x = z[b]
    out[b] = ((x ** 5) / 12 - (x ** 4) / 2 + (x ** 3) * 5 / 8
              + (x ** 2) * 5 / 3 - 5 * x + 4 - 2 / (3 * x))
    return np.clip(out, 0.0, 1.0)


def periodic_dist(ax, ay, bx, by, L=2 * np.pi):
    dx = np.abs(ax[:, None] - bx[None, :])
    dx = np.minimum(dx, L - dx)
    dy = np.abs(ay[:, None] - by[None, :])
    dy = np.minimum(dy, L - dy)
    return np.sqrt(dx ** 2 + dy ** 2)


class LETKF:
    def __init__(self, N, ix, iy, sigma, loc_radius=0.8, block=8,
                 inflation="rtps", infl_param=0.9,
                 inflate_unobserved=False, L=2 * np.pi):
        self.N, self.L = N, L
        if N % block != 0:
            raise ValueError(f"LETKF block={block} must divide N={N}")
        if sigma <= 0 or not np.isfinite(sigma):
            raise ValueError("observation sigma must be finite and positive")
        if inflation not in ("mult", "rtps", "rtpp"):
            raise ValueError("inflation must be one of mult/rtps/rtpp")
        self.ix, self.iy = ix, iy
        self.obs_flat = ix * N + iy
        self.sigma, self.loc = float(sigma), float(loc_radius)
        self.inflation = inflation
        self.infl_param = float(infl_param)
        self.inflate_unobserved = bool(inflate_unobserved)
        self.sx, self.sy = ix * L / N, iy * L / N
        nb = N // block
        self.block, self.nb = block, nb
        bc = (np.arange(nb) * block + block / 2.0) * L / N
        BX, BY = np.meshgrid(bc, bc, indexing="ij")
        D = periodic_dist(BX.ravel(), BY.ravel(), self.sx, self.sy, L)
        self.rho = gaspari_cohn(D / loc_radius)
        self.blocks = [np.where(self.rho[b] > 1e-6)[0]
                       for b in range(nb * nb)]
        gi = np.arange(N * N).reshape(N, N)
        self.block_idx = [gi[i * block:(i + 1) * block,
                             j * block:(j + 1) * block].ravel()
                          for i in range(nb) for j in range(nb)]

    def analysis(self, Xb, y):
        """Xb: (M, N*N) real-grid ensemble; y: (p,) observations."""
        Xb = np.asarray(Xb, float)
        y = np.asarray(y, float)
        if not np.all(np.isfinite(Xb)):
            raise LETKFDivergence("background ensemble contains NaN/Inf")
        if not np.all(np.isfinite(y)):
            raise LETKFDivergence("observation vector contains NaN/Inf")
        M = Xb.shape[0]
        with np.errstate(over="ignore", invalid="ignore"):
            xbar = Xb.mean(0)
            Xp = Xb - xbar
        if not np.all(np.isfinite(xbar)):
            raise LETKFDivergence("background mean became NaN/Inf")
        infl = self.infl_param if self.inflation == "mult" else 1.0
        d = y - xbar[self.obs_flat]
        Xa = np.empty_like(Xb)
        s2 = self.sigma ** 2
        for b, sel in enumerate(self.blocks):
            gidx = self.block_idx[b]
            if sel.size == 0:
                if self.inflate_unobserved:
                    Xa[:, gidx] = xbar[gidx] + infl * Xp[:, gidx]
                else:
                    Xa[:, gidx] = xbar[gidx] + Xp[:, gidx]
                continue
            Xpl = infl * Xp if self.inflation == "mult" else Xp
            # sel indexes the LOCAL observation list; map it to grid columns.
            Yl = Xpl[:, self.obs_flat[sel]]
            C = Yl * (self.rho[b, sel] / s2)
            A = (M - 1) * np.eye(M) + C @ Yl.T
            A = 0.5 * (A + A.T)
            if not np.all(np.isfinite(A)):
                raise LETKFDivergence(f"local analysis matrix non-finite "
                                      f"in block {b}")
            try:
                evals, evecs = np.linalg.eigh(A)
            except np.linalg.LinAlgError as e:
                raise LETKFDivergence(
                    f"eigendecomposition failed in block {b}: {e}") from e
            if not np.all(np.isfinite(evals)) or not np.all(np.isfinite(evecs)):
                raise LETKFDivergence(f"non-finite eigensystem in block {b}")
            evals = np.maximum(evals, 1e-12)
            Pa = evecs @ np.diag(1.0 / evals) @ evecs.T
            W = evecs @ np.diag(np.sqrt((M - 1) / evals)) @ evecs.T
            wbar = Pa @ (C @ d[sel])
            Xa[:, gidx] = xbar[gidx] + (W + wbar[:, None]).T @ Xpl[:, gidx]
        # RTPP / RTPS relaxation.
        if self.inflation == "rtpp":
            a = self.infl_param
            Xa = xbar[None, :] + (1.0 - a) * Xp + a * (Xa - xbar[None, :])
        elif self.inflation == "rtps":
            sb = Xp.std(axis=0, ddof=1)
            sa = (Xa - Xa.mean(0)[None, :]).std(axis=0, ddof=1)
            den = np.maximum(sb, 1e-30)
            rho = 1.0 + self.infl_param * (sb - sa) / den
            Xa = xbar[None, :] + rho[None, :] * (Xa - xbar[None, :])
        if not np.all(np.isfinite(Xa)):
            raise LETKFDivergence("analysis ensemble contains NaN/Inf")
        return Xa


def _row_rms_safe(X):
    X = np.asarray(X, float)
    scale = np.max(np.abs(X), axis=1)
    rms = np.zeros_like(scale)
    nz = scale > 0.0
    if np.any(nz):
        Z = X[nz] / scale[nz, None]
        rms[nz] = scale[nz] * np.sqrt(np.mean(Z * Z, axis=1))
    return rms


def run_letkf(flow, wh_true0, ix, iy, M=24, T=8.0, dt_obs=0.1,
              noise=0.02, loc_radius=0.8, inflation="rtps",
              infl_param=0.9, inflate_unobserved=False, block=8,
              seed=0, record_every=1, diverge_factor=1e4):
    """Fair LETKF baseline.  Returns (ts, rel_err, n_solver_steps, info)."""
    from .observer import random_observer_state
    g = flow.g
    rng = np.random.default_rng(seed)
    m_assim = max(int(round(dt_obs / flow.dt)), 1)
    truth_rms = float(np.sqrt(np.mean(g.inv(wh_true0) ** 2)))
    obs_sigma = noise * truth_rms
    sigma = max(obs_sigma, 1e-8 * truth_rms, 1e-12)
    filt = LETKF(g.N, ix, iy, sigma, loc_radius, block, inflation,
                 infl_param, inflate_unobserved)

    wh_t = wh_true0.copy()
    Wh = np.stack([random_observer_state(flow, wh_true0, seed=seed + k)
                   for k in range(M)])
    ts, errs, nsteps = [], [], 0
    info = dict(diverged=False, reason=None, cycle=None,
                max_member_rms_ratio=1.0)
    ncyc = int(T / (m_assim * flow.dt))
    for c in range(ncyc):
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(m_assim):
                wh_t = flow.step(wh_t)
                for m in range(M):
                    Wh[m] = flow.step(Wh[m])
                nsteps += M
        if not np.all(np.isfinite(wh_t)):
            raise RuntimeError("truth solver became non-finite in LETKF run")
        if not np.all(np.isfinite(Wh)):
            info.update(diverged=True,
                        reason="forecast ensemble became NaN/Inf", cycle=c)
            break
        Xb = np.stack([g.inv(Wh[m]).ravel() for m in range(M)])
        if not np.all(np.isfinite(Xb)):
            info.update(diverged=True,
                        reason="real-space background became NaN/Inf", cycle=c)
            break
        member_rms = _row_rms_safe(Xb)
        ratio = float(np.max(member_rms) / max(truth_rms, 1e-30))
        if not np.isfinite(ratio):
            info.update(diverged=True,
                        reason="ensemble RMS ratio exceeded float64 range",
                        cycle=c)
            break
        info["max_member_rms_ratio"] = max(float(info["max_member_rms_ratio"]),
                                           ratio)
        if ratio > diverge_factor:
            info.update(diverged=True,
                        reason=f"ensemble RMS exceeded {diverge_factor:g}x",
                        cycle=c)
            break
        y = g.inv(wh_t).ravel()[filt.obs_flat] + rng.standard_normal(
            len(ix)) * obs_sigma
        try:
            Xa = filt.analysis(Xb, y)
        except LETKFDivergence as e:
            info.update(diverged=True, reason=str(e), cycle=c)
            break
        for m in range(M):
            Wh[m] = g.fwd(Xa[m].reshape(g.N, g.N)) * g.dealias
        if not np.all(np.isfinite(Wh)):
            info.update(diverged=True,
                        reason="spectral analysis became NaN/Inf", cycle=c)
            break
        if c % record_every == 0:
            what = Wh.mean(0)
            err = float(np.linalg.norm(what - wh_t)
                        / np.linalg.norm(wh_t))
            if not np.isfinite(err):
                info.update(diverged=True,
                            reason="relative error became NaN/Inf", cycle=c)
                break
            ts.append((c + 1) * m_assim * flow.dt)
            errs.append(err)
    return np.array(ts), np.array(errs), nsteps, info
