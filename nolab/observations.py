"""Observation operators for the v2 package.

v2 fixes versus v1:
  * the Gaussian kernel is applied exactly once (v1 step1 doubled it);
  * denom_floor is a REQUIRED explicit argument (v1 silently defaulted it,
    and training used a different value from the numpy experiments);
  * covering radius and anisotropy are first-class diagnostics.
"""

import numpy as np


def covering_radius(g, ix, iy):
    """Largest distance from any grid point to its nearest sensor
    (periodic box, exact on the grid).
    """
    from scipy.spatial import cKDTree
    N = g.N
    s = np.stack([ix.astype(float) * g.L / N,
                  iy.astype(float) * g.L / N], 1)
    off = np.array([[-1, -1], [-1, 0], [-1, 1],
                    [0, -1], [0, 0], [0, 1],
                    [1, -1], [1, 0], [1, 1]], float) * g.L
    reps = np.concatenate([s + o for o in off], 0)
    tree = cKDTree(reps)
    gx, gy = np.meshgrid(g.x, g.x, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    d, _ = tree.query(pts, k=1, workers=-1)
    return float(d.max())


def anisotropy_index(g, ix, iy):
    """Ratio sqrt(lambda_max / lambda_min) of the circular covariance of
    sensor positions.  1 = isotropic, >1 = directionally clustered."""
    x = ix.astype(float) * 2 * np.pi / g.N
    y = iy.astype(float) * 2 * np.pi / g.N
    mx = np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x)))
    my = np.arctan2(np.mean(np.sin(y)), np.mean(np.cos(y)))
    dx = np.mod(x - mx + np.pi, 2 * np.pi) - np.pi
    dy = np.mod(y - my + np.pi, 2 * np.pi) - np.pi
    C = np.cov(np.stack([dx, dy]))
    w = np.linalg.eigvalsh(C)
    w = np.maximum(w, 1e-30)
    return float(np.sqrt(w[-1] / w[0]))


class SpectralObs:
    """Ideal projection onto Fourier modes with |kx|,|ky| <= K."""
    name = "spectral"

    def __init__(self, g, K):
        self.g, self.K = g, K
        self.mask = ((np.abs(g.kx) <= K) & (np.abs(g.ky) <= K)).astype(float)
        self.ndof = int((2 * K + 1) ** 2)

    def apply_h(self, resid_h, state_h=None):
        return resid_h * self.mask


class PointObs:
    """n_sensors nodal values, spread back with a normalised (Shepard)
    Gaussian.  denom_floor is required: training and evaluation must agree
    on it or the 'learned gain beats classical' gap is an artifact."""
    name = "points"

    def __init__(self, g, n_sensors, denom_floor, seed=0, width_factor=1.0):
        self.g = g
        rng = np.random.default_rng(seed)
        idx = rng.choice(g.N * g.N, size=int(n_sensors), replace=False)
        self.ix, self.iy = idx // g.N, idx % g.N
        self._finish(denom_floor, width_factor)

    def _finish(self, denom_floor, width_factor=1.0):
        g = self.g
        if denom_floor is None:
            raise TypeError("denom_floor is required for PointObs")
        self.denom_floor = float(denom_floor)
        self.n = len(self.ix)
        self.mask = np.zeros((g.N, g.N))
        self.mask[self.ix, self.iy] = 1.0
        h = width_factor * g.L / np.sqrt(self.n)
        self.kernel_h = np.exp(-0.5 * (h ** 2) * g.k2)
        d = g.inv(g.fwd(self.mask) * self.kernel_h)
        self.denom = np.maximum(d, max(self.denom_floor * d.mean(), 1e-12))
        self.ndof = self.n
        self.covering_radius = covering_radius(g, self.ix, self.iy)
        self.anisotropy = anisotropy_index(g, self.ix, self.iy)

    def _spread(self, field):
        g = self.g
        num = g.inv(g.fwd(field * self.mask) * self.kernel_h)
        return num / self.denom

    def apply_h(self, resid_h, state_h=None):
        g = self.g
        r = g.inv(resid_h)
        return g.fwd(self._spread(r)) * g.dealias


class FixedPointObs(PointObs):
    """PointObs with externally supplied sensor indices (layouts)."""

    def __init__(self, g, ix, iy, denom_floor, width_factor=1.0):
        self.g = g
        self.ix, self.iy = np.asarray(ix, int), np.asarray(iy, int)
        self._finish(denom_floor, width_factor)


class ShellGainObs(SpectralObs):
    """Spectral observation with a wavenumber-shell-dependent gain."""
    name = "shell_gain"

    def __init__(self, g, K, gains=None):
        super().__init__(g, K)
        self.shell = np.minimum(np.round(np.sqrt(g.k2)).astype(int), K)
        self.nshell = K + 1
        self.gains = np.ones(self.nshell) if gains is None else np.asarray(gains, float)

    def set_gains(self, gains):
        self.gains = np.asarray(gains, float)

    def apply_h(self, resid_h, state_h=None):
        return resid_h * self.mask * self.gains[self.shell]


class VelocityPointObs(PointObs):
    """Sensors measure the two velocity components, not vorticity."""
    name = "velocity_points"

    def apply_h(self, resid_h, state_h=None):
        g = self.g
        psih = resid_h * g.k2inv
        du = g.inv(1j * g.ky * psih)
        dv = g.inv(-1j * g.kx * psih)
        Uh = g.fwd(self._spread(du)) * g.dealias
        Vh = g.fwd(self._spread(dv)) * g.dealias
        return 1j * g.kx * Vh - 1j * g.ky * Uh          # curl back to vorticity


class NonlinearPointObs(PointObs):
    """Saturating sensor: y = h(w) = s * tanh(w / s)."""
    name = "nonlinear_points"

    def __init__(self, g, n_sensors, denom_floor, seed=0, sat=1.0,
                 mode="newton", width_factor=1.0, jac_floor=0.05):
        super().__init__(g, n_sensors, denom_floor, seed=seed,
                         width_factor=width_factor)
        self.sat, self.mode, self.jac_floor = sat, mode, jac_floor

    def h(self, z):
        return self.sat * np.tanh(z / self.sat)

    def dh(self, z):
        return 1.0 / np.cosh(z / self.sat) ** 2

    def apply_h(self, resid_h, state_h=None):
        g = self.g
        if state_h is None:
            raise ValueError("nonlinear observation needs the observer state")
        what = g.inv(state_h)
        w_true = g.inv(state_h - resid_h)
        r = self.h(what) - self.h(w_true)
        if self.mode == "adjoint":
            r = r * self.dh(what)
        elif self.mode == "newton":
            r = r / np.maximum(self.dh(what), self.jac_floor)
        return g.fwd(self._spread(r)) * g.dealias


class PointShellObs(PointObs):
    """Point sensors + a shell-dependent gain on the spread-back field."""
    name = "point_shell_gain"

    def __init__(self, g, n_sensors, denom_floor, nshell=6, gains=None, **kw):
        super().__init__(g, n_sensors, denom_floor, **kw)
        edges = np.linspace(0, g.N / 3.0, nshell + 1)
        self.nshell = nshell
        self.shell = np.clip(np.digitize(np.sqrt(g.k2), edges) - 1, 0, nshell - 1)
        self.gains = np.ones(nshell) if gains is None else np.asarray(gains, float)

    def apply_h(self, resid_h, state_h=None):
        return super().apply_h(resid_h, state_h) * self.gains[self.shell]
