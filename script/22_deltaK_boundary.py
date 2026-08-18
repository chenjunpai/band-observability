"""22 -- delta_K at the synchronisation boundary (the control that 18 is missing).

Script 18 computes delta_K only at n=784, seed=0, for four layouts whose h values
differ wildly.  That cannot test the mechanism claim, because h and geometry move
together.  The decisive test is on the pairs from scripts 17/19 that sit ON the
sync/fail boundary:

    uniform n=200 seed1   h = 0.5911  -> FAILS      (17, all mu)
    uniform n=300 seed0   h = 0.5911  -> converges  (17, rate 0.79)
    lattice n=100 (m=10)  h = 0.4165  -> FAILS      (17, all mu)
    lattice n=144 (m=12)  h = 0.3471  -> converges  (17, rate 0.88)

Same h, opposite outcome => h is not the controlling variable.  This script
checks whether delta_K separates them where h does not, and prints the
lattice-aliasing series (m = 8, 10, 12, 14), where an exact prediction exists:
a regular m x m lattice aliases |k| >= m/2 onto the observed band, so the
observation operator is singular on |k| <= K unless m >= 2K + 1.  With
K_c = 5 at nu = 5e-3 that predicts failure for m <= 10 and success for m >= 11.

Run from the repo root (same layout as the other scripts):

    python scripts/22_deltaK_boundary.py

Reference output (N = 128, K = 5, numpy 2.4):

    layout                             h   dK(f=0)  dK(f=.25)   maxeig
    uniform n=200 s=0              0.725   -0.1748    -0.1748    1.152
    uniform n=200 s=1              0.591   -0.1391    -0.1391    1.153
    uniform n=300 s=0              0.591   -0.0900    -0.0900    1.084
    uniform n=300 s=1              0.531   -0.0828    -0.0828    1.091
    uniform n=400 s=0              0.483   -0.0495    -0.0495    1.103
    uniform n=400 s=1              0.483   -0.0553    -0.0553    1.155
    uniform n=500 s=0              0.405    0.0366     0.0366    1.083
    uniform n=784 s=0              0.422    0.1235     0.1234    1.041
    lattice n=64  (m=8)            0.555   -0.0127    -0.0127    1.000
    lattice n=100 (m=10)           0.417   -0.0012    -0.0012    1.001
    lattice n=144 (m=12)           0.347    0.0010     0.0010    1.002
    lattice n=196 (m=14)           0.347    0.0058     0.0058    1.001
    lattice n=400 (m=20)           0.208    0.0830     0.0830    1.000
    lattice n=784 (m=28)           0.139    0.2827     0.2827    1.000
    ground_tracks n=784            0.621   -0.2056    -0.1152    1.211

Reading of that table:

  * delta_K separates the h = 0.591 pair (-0.139 fails vs -0.090 converges);
    h does not.  So report delta_K, not h, as the controlling quantity.
  * the ZERO crossing is not the empirical boundary: uniform n=300 and n=400
    have delta_K < 0 and still converge.  The boundary sits near -0.10.
    "delta_K < 0 => failure" must be weakened to "delta_K below a
    mu-dependent threshold => failure", or the threshold must be derived.
  * the lattice series crosses zero exactly between m = 10 (fails) and
    m = 12 (converges), matching the aliasing prediction m >= 2K_c + 1.
    This is the one case where the mechanism is provable, not just measured.
  * denom_floor changes delta_K only for gapped layouts (uniform: identical at
    floor 0 and 0.25; ground_tracks: -0.206 -> -0.115; clustered: -0.86 -> -0.09).
    Any delta_K number reported in the paper must state its floor.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.linalg import eigh

from nolab import Grid, GENERATORS, covering_radius
from nolab.observations import FixedPointObs

K = 5
N_GRID = 128


def delta_K_point(g, ix, iy, K, denom_floor=0.25):
    """(min, max) generalised eigenvalue of sym(I_h) restricted to |k| <= K.

    Identical to the definition in 18_theory_diagnostics.delta_K_point; the max
    eigenvalue is returned as well because it is the normalisation of the
    operator (I_h is not a contraction: max eig runs 1.00-1.21 here), and
    delta_K is only interpretable relative to it.
    """
    obs = FixedPointObs(g, ix, iy, denom_floor=denom_floor)
    ks = [(kx, ky) for kx in range(-K, K + 1) for ky in range(-K, K + 1)]
    xx, yy = g.X.ravel(), g.Y.ravel()
    B = np.empty((g.N * g.N, len(ks)), dtype=np.complex128)
    for j, (kx, ky) in enumerate(ks):
        B[:, j] = np.exp(1j * (kx * xx + ky * yy))
    IhB = np.empty_like(B)
    for j in range(len(ks)):
        e = B[:, j].reshape(g.N, g.N)
        IhB[:, j] = (obs._spread(e.real).ravel()
                     + 1j * obs._spread(e.imag).ravel())
    G = B.conj().T @ B                     # standard L2 Gram
    A = B.conj().T @ IhB                   # <I_h e_j, e_i>
    S = (A + A.conj().T) / 2               # symmetric part
    ev = eigh(S, G, eigvals_only=True)
    return float(np.real(ev[0])), float(np.real(ev[-1]))


def main():
    g = Grid(N_GRID)
    cases = []
    for n, seed in [(200, 0), (200, 1), (300, 0), (300, 1),
                    (400, 0), (400, 1), (500, 0), (784, 0)]:
        ix, iy = GENERATORS["uniform"](g.N, n, seed=seed)
        cases.append((f"uniform n={n} s={seed}", ix, iy))
    for n in (64, 100, 144, 196, 400, 784):
        ix, iy = GENERATORS["lattice"](g.N, n, seed=0)
        cases.append((f"lattice n={n} (m={int(np.sqrt(n))})", ix, iy))
    ix, iy = GENERATORS["ground_tracks"](g.N, 784, seed=0)
    cases.append(("ground_tracks n=784", ix, iy))

    print(f"{'layout':28}{'h':>8}{'dK(f=0)':>10}{'dK(f=.25)':>11}{'maxeig':>9}")
    for name, ix, iy in cases:
        h = covering_radius(g, ix, iy)
        d0, _ = delta_K_point(g, ix, iy, K, 0.0)
        d25, mx = delta_K_point(g, ix, iy, K, 0.25)
        print(f"{name:28}{h:>8.3f}{d0:>10.4f}{d25:>11.4f}{mx:>9.3f}")


if __name__ == "__main__":
    main()
