"""Layout diagnostics that actually discriminate between layouts.

Three things live here.

1. `corridor_diagnostics` -- directional gap statistics, replacing
   `nolab.observations.anisotropy_index`.  That index is the eccentricity of
   the circular covariance of the sensor positions; for any layout that covers
   the torus it is ~1 regardless of structure (measured: 1.01-1.05 for perfect
   horizontal stripes, which are maximally directional).  It only reacts to
   layouts whose sensor MASS is concentrated (clustered 1.56, blind_half 1.99),
   which is a different property from having a gap.

2. `delta_K` -- the low-frequency observation efficiency used in
   `scripts/18_theory_diagnostics.py`, moved into the library so that every
   script reports the same number, plus the two things 18 did not report:
   the largest eigenvalue (the operator is not a contraction: max eig runs
   1.00-1.21, and delta_K is only meaningful relative to that normalisation)
   and the floor used.

3. `band_coupling` -- the size of the block of I_h that maps the observed band
   |k| <= K into its complement.  delta_K only certifies coercivity WITHIN the
   band; the energy estimate also needs the band-to-tail leakage to be small,
   and aliasing acts exactly through this block.  Reporting delta_K without it
   is the gap a theory referee will find first.
"""

import numpy as np
from scipy.linalg import eigh

from nolab.observations import FixedPointObs


def corridor_diagnostics(g, ix, iy, n_dirs=64):
    """Largest empty strip in each direction.

    For each direction theta we project every sensor onto the unit normal
    (cos theta, sin theta) and take the largest circular gap between
    consecutive projections.  That gap is the width of the widest sensor-free
    strip whose long axis is perpendicular to theta.

    Returns dict(max_corridor, min_corridor, gap_anisotropy, argmax_angle).
    A uniform random layout of n points has max_corridor ~ O(L/n); stripes with
    pitch p have max_corridor ~ 2*pi/p; blind_half has max_corridor ~ pi.
    """
    x = np.asarray(ix, float) * 2 * np.pi / g.N
    y = np.asarray(iy, float) * 2 * np.pi / g.N
    widths = []
    thetas = np.linspace(0, np.pi, n_dirs, endpoint=False)
    for th in thetas:
        s = np.mod(x * np.cos(th) + y * np.sin(th), 2 * np.pi)
        s = np.sort(s)
        d = np.diff(s)
        wrap = s[0] + 2 * np.pi - s[-1]
        widths.append(float(max(d.max() if d.size else 0.0, wrap)))
    widths = np.asarray(widths)
    return dict(max_corridor=float(widths.max()),
                min_corridor=float(widths.min()),
                gap_anisotropy=float(widths.max() / max(widths.min(), 1e-12)),
                argmax_angle=float(thetas[int(np.argmax(widths))]))


def _band_basis(g, K):
    ks = [(kx, ky) for kx in range(-K, K + 1) for ky in range(-K, K + 1)]
    xx, yy = g.X.ravel(), g.Y.ravel()
    B = np.empty((g.N * g.N, len(ks)), dtype=np.complex128)
    for j, (kx, ky) in enumerate(ks):
        B[:, j] = np.exp(1j * (kx * xx + ky * yy))
    return B, ks


def delta_K(g, ix, iy, K=5, denom_floor=0.25, width_factor=1.0):
    """(min, max) generalised eigenvalue of sym(I_h) restricted to |k| <= K.

    delta_K > 0 is the coercivity that the nudging energy estimate needs on the
    observed band.  Note the empirical calibration in the v3 data: the zero
    crossing is NOT the sync/fail boundary (uniform n = 300 and n = 400 have
    delta_K = -0.09 and -0.05 and both synchronise), because coercivity is a
    sufficient, not a necessary, condition.  Report the value, not just its sign.
    """
    obs = FixedPointObs(g, ix, iy, denom_floor=denom_floor,
                        width_factor=width_factor)
    B, _ = _band_basis(g, K)
    IhB = np.empty_like(B)
    for j in range(B.shape[1]):
        e = B[:, j].reshape(g.N, g.N)
        IhB[:, j] = (obs._spread(e.real).ravel()
                     + 1j * obs._spread(e.imag).ravel())
    G = B.conj().T @ B
    A = B.conj().T @ IhB
    S = (A + A.conj().T) / 2
    ev = eigh(S, G, eigvals_only=True)
    return float(np.real(ev[0])), float(np.real(ev[-1]))


def band_coupling(g, ix, iy, K=5, denom_floor=0.25, width_factor=1.0):
    """|| (I - P_K) I_h P_K || -- how much observed-band energy the observation
    operator scatters outside the band (relative to || I_h P_K ||).

    For a spectral projection this is exactly 0.  For an m x m regular lattice
    with m <= 2K it is O(1), because mode k and mode k + m are indistinguishable
    to the sampler: that is aliasing, and it is invisible to delta_K alone.
    """
    obs = FixedPointObs(g, ix, iy, denom_floor=denom_floor,
                        width_factor=width_factor)
    B, _ = _band_basis(g, K)
    n2 = g.N * g.N
    tot, inband = 0.0, 0.0
    for j in range(B.shape[1]):
        e = B[:, j].reshape(g.N, g.N)
        r = obs._spread(e.real) + 1j * obs._spread(e.imag)
        rh = np.fft.fft2(r) / n2
        mask = ((np.abs(g.kx) <= K) & (np.abs(g.ky) <= K))
        tot += float(np.sum(np.abs(rh) ** 2))
        inband += float(np.sum(np.abs(rh[mask]) ** 2))
    leak = max(tot - inband, 0.0)
    return dict(leak_fraction=float(leak / max(tot, 1e-30)),
                in_band_fraction=float(inband / max(tot, 1e-30)))


def layout_report(g, ix, iy, K=5, denom_floor=0.25, width_factor=1.0,
                  with_coupling=True):
    """Everything a layout should be characterised by, in one call."""
    from nolab.observations import covering_radius, anisotropy_index
    d0, dmax = delta_K(g, ix, iy, K, 0.0, width_factor)
    df, _ = delta_K(g, ix, iy, K, denom_floor, width_factor)
    out = dict(n=int(len(ix)),
               h=float(covering_radius(g, ix, iy)),
               anisotropy_old=float(anisotropy_index(g, ix, iy)),
               delta_K_floor0=d0, delta_K_floor=df,
               delta_K_max_eig=dmax, denom_floor=float(denom_floor),
               width_factor=float(width_factor), K=int(K))
    out.update(corridor_diagnostics(g, ix, iy))
    if with_coupling:
        out.update(band_coupling(g, ix, iy, K, denom_floor, width_factor))
    return out
