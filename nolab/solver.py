"""Single pseudo-spectral solver for the whole v2 package.

Vorticity form on the periodic box [0, 2*pi)^2:

    d_t w + u.grad w = nu * Lap w - alpha * w + f,
    Lap psi = -w,   u = (d_y psi, -d_x psi),
    f = -n_f * cos(n_f * y)          (curl of the Kolmogorov forcing)

Full-spectrum FFT, 2/3 dealiasing, integrating-factor Heun (RK2) with the
linear part treated exactly.  The nudging term is evaluated at both RK
stages with the truth taken at the matching stage time.

v2 fixes versus v1:
  * one canonical implementation everywhere (v1 had two different solvers);
  * full fft2 everywhere, so every error ratio satisfies Parseval;
  * grashof() includes the Ekman damping alpha;
  * CFL-aware spin-up and an equilibrium check;
  * SOLVER_VERSION + a source fingerprint written into every results file.
"""

import hashlib
import inspect

import numpy as np

SOLVER_VERSION = "2.0.0"


def _fp():
    try:
        src = inspect.getsource(Grid) + inspect.getsource(KolmogorovFlow)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    except (OSError, TypeError):  # pragma: no cover
        return hashlib.sha256(SOLVER_VERSION.encode("utf-8")).hexdigest()[:16]


def solver_fingerprint():
    return _fp()


class Grid:
    """Spectral utilities on the periodic box."""

    def __init__(self, N, L=2 * np.pi):
        self.N, self.L = N, L
        k = np.fft.fftfreq(N, d=1.0 / N) * (2 * np.pi / L)
        self.kx = k[:, None] * np.ones((1, N))
        self.ky = np.ones((N, 1)) * k[None, :]
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2inv = np.zeros_like(self.k2)
        nz = self.k2 > 0
        self.k2inv[nz] = 1.0 / self.k2[nz]
        kmax = (2.0 / 3.0) * (N // 2) * (2 * np.pi / L)
        self.dealias = ((np.abs(self.kx) <= kmax)
                        & (np.abs(self.ky) <= kmax)).astype(float)
        self.x = np.arange(N) * L / N
        self.X, self.Y = np.meshgrid(self.x, self.x, indexing="ij")

    def fwd(self, a):
        return np.fft.fft2(a)

    def inv(self, ah):
        return np.real(np.fft.ifft2(ah))

    def velocity(self, wh):
        psih = wh * self.k2inv
        return self.inv(1j * self.ky * psih), self.inv(-1j * self.kx * psih)


class KolmogorovFlow:
    def __init__(self, N=128, nu=5e-3, alpha=0.1, n_forcing=4, dt=None,
                 L=2 * np.pi):
        self.g = Grid(N, L)
        self.nu, self.alpha, self.nf = nu, alpha, n_forcing
        self.dt = 0.004 if dt is None else dt
        g = self.g
        self.fh = g.fwd(-n_forcing * np.cos(n_forcing * g.Y))
        self.lin = -(nu * g.k2 + alpha)
        self.E = np.exp(self.lin * self.dt)

    # -- diagnostics ---------------------------------------------------
    def Re(self):
        """Crude Reynolds number from the laminar Kolmogorov balance."""
        U = 1.0 / (self.nu * self.nf ** 2 + self.alpha)
        return float(U * self.g.L / (2 * np.pi * self.nu))

    def grashof(self):
        """Grashof number from the laminar balance, including Ekman damping.

        G = f0 / (nu * kf * (nu*kf**2 + alpha)),  f0 = nf.

        At alpha = 0 this reduces to 1/(nu^2 * nf^2), the v1 formula, which is
        the standard G = f0/(nu^2 kf^3) for a vorticity forcing of amplitude
        f0 = kf.  With alpha > 0 the damping weakens the response, which is
        exactly the direction exp18 explores, so it must appear in G.
        """
        f0 = self.nf
        kf = self.nf
        return float(f0 / (self.nu * kf * (self.nu * kf ** 2 + self.alpha)))

    def enstrophy(self, wh):
        return float(0.5 * np.mean(self.g.inv(wh) ** 2))

    def rms(self, wh):
        return float(np.sqrt(np.mean(self.g.inv(wh) ** 2)))

    # -- stability -----------------------------------------------------
    def dt_cfl(self, wh, safety=0.4):
        """Largest stable dt for the current state (advective CFL)."""
        u, v = self.g.velocity(wh)
        umax = float(np.max(np.sqrt(u ** 2 + v ** 2)))
        dx = self.g.L / self.g.N
        return safety * dx / max(umax, 1e-12)

    def set_dt(self, dt):
        self.dt = float(dt)
        self.E = np.exp(self.lin * self.dt)

    # -- dynamics ------------------------------------------------------
    def nonlin(self, wh):
        g = self.g
        u, v = g.velocity(wh)
        wx = g.inv(1j * g.kx * wh)
        wy = g.inv(1j * g.ky * wh)
        return -g.fwd(u * wx + v * wy) * g.dealias + self.fh

    def step(self, wh, nudge1=None, nudge2=None):
        dt, E = self.dt, self.E
        N1 = self.nonlin(wh)
        if nudge1 is not None:
            N1 = N1 + nudge1(wh)
        w1 = E * (wh + dt * N1)
        N2 = self.nonlin(w1)
        if nudge2 is not None:
            N2 = N2 + nudge2(w1)
        return E * wh + 0.5 * dt * (E * N1 + N2)

    def spinup(self, T=30.0, seed=0, amp=1.0, cfl=False,
               safety=0.5, check_every=20):
        """Spin up from a smooth random field, optionally CFL-aware."""
        g = self.g
        rng = np.random.default_rng(seed)
        wh = (g.fwd(rng.standard_normal((g.N, g.N)))
              * np.exp(-0.5 * g.k2 / 16.0) * g.dealias)
        wh *= amp * g.N / max(np.sqrt(np.mean(g.inv(wh) ** 2)), 1e-12) / g.N
        t, step = 0.0, 0
        nsteps = int(T / self.dt)
        for _ in range(nsteps):
            if cfl and step % check_every == 0:
                dt_cfl = self.dt_cfl(wh, safety=safety)
                if not np.isfinite(dt_cfl):
                    raise FloatingPointError("non-finite CFL estimate")
                if dt_cfl < self.dt:
                    self.set_dt(dt_cfl)
            wh = self.step(wh)
            if not np.all(np.isfinite(wh)):
                raise FloatingPointError(
                    "non-finite state during spin-up; increase N or lower dt")
            t += self.dt
            step += 1
        return wh


def equilibration_time(flow, wh, tol=0.02, window=5.0, min_T=5.0,
                       max_T=None, every=25, cfl=False, cfl_safety=0.5):
    """Advance until the mean-enstrophy drift over a sliding window is
    below `tol`.  Returns (t_equilibrated, wh_final, history).

    This is the honest replacement for the v1 habit of picking T_SPIN by
    hand (exp18 used 15 for states whose damping time is 333).
    """
    if max_T is None:
        max_T = max(10.0 / max(flow.alpha, 1e-9), 200.0)
    hist = [dict(t=0.0, enstrophy=flow.enstrophy(wh))]
    t = 0.0
    nwin = max(int(window / flow.dt), 1)
    buf = []
    step = 0
    while t < max_T:
        if cfl:
            dt_cfl = flow.dt_cfl(wh, safety=cfl_safety)
            if np.isfinite(dt_cfl) and dt_cfl < flow.dt:
                flow.set_dt(dt_cfl)
        wh = flow.step(wh)
        if not np.all(np.isfinite(wh)):
            raise FloatingPointError("non-finite state in equilibration run")
        t += flow.dt
        step += 1
        if step % every == 0:
            buf.append(flow.enstrophy(wh))
            hist.append(dict(t=t, enstrophy=buf[-1]))
            if len(buf) >= 2 * nwin // every:
                a = np.mean(buf[-2 * nwin // every:-nwin // every])
                b = np.mean(buf[-nwin // every:])
                denom = max(abs(b), 1e-30)
                if t >= min_T and abs(b - a) / denom < tol:
                    return t, wh, hist
    return t, wh, hist
