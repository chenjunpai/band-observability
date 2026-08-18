"""Sensor layouts with GENUINE directional gaps.

Why this file exists
--------------------
`scripts/19_anisotropy_ablation.py` defined its stripe family inline as

    per = n // n_stripes
    for each stripe: draw `per` uniform x, snap to the grid, dedup
    while len(ix) < n:  add UNIFORM RANDOM points until the count is reached

On an N = 128 grid a single stripe row holds at most 128 distinct sensors, so
whenever `n // n_stripes > 128` the deduplication throws points away and the
`while` loop refills them uniformly over the whole box.  Measured on the actual
generator (n = 784, seed 0):

    n_stripes    on-stripe sensors    distinct rows occupied
        3          340 / 784 (43%)            123
        4          419 / 784 (53%)            122
        6          494 / 784 (63%)            112
       16          672 / 784 (86%)             85

i.e. the "anisotropic" family was a stripe pattern sitting on top of 100-450
uniformly scattered sensors -- and a uniform layout of 400 sensors already
synchronises on its own (rate 1.96).  The gaps the ablation was built to test
were filled in by its own fallback.  That is why the isotropic and anisotropic
curves coincided, and it is why the conclusion drawn from that coincidence
("coverage radius explains everything, the geometric residual is withdrawn")
is not supported.

The generators below never fall back to random filling.  If the requested count
does not fit in the requested geometry they thicken the structure (deterministic
and reported) or raise, and every generator returns the layout together with a
`meta` dict recording what was actually built.

Also note that `nolab.observations.anisotropy_index` cannot detect any of these
families: it is the eccentricity of the circular covariance of sensor positions,
which is ~1 for any layout that covers the torus, including perfect stripes
(measured 1.01-1.05 for every stripe layout above).  Use
`nolab_fix.geometry_fix.corridor_diagnostics` instead.
"""

import numpy as np


def _dedup(ix, iy, N):
    seen, ox, oy = set(), [], []
    for a, b in zip(ix, iy):
        k = int(a) * N + int(b)
        if k not in seen:
            seen.add(k)
            ox.append(int(a))
            oy.append(int(b))
    return np.asarray(ox, int), np.asarray(oy, int)


def stripes_exact(N, n, n_stripes, seed=0, jitter=False, jitter_mode="point",
                  strict_count=False):
    """Horizontal bands with a genuine empty corridor between them.

    `n` sensors are distributed over `n_stripes` bands.  Each band is made
    `thickness` grid rows deep, the smallest thickness such that the requested
    per-band count fits without duplication; along a row the sensors are placed
    at regular grid spacing, never scattered over the box.

    JITTER MODES -- read this before choosing one
    ---------------------------------------------
    `jitter_mode="point"` (what `jitter=True` did before this patch, kept only
    for bit-exact reproduction of results/19, /30, /31, /33) displaces EACH
    sensor by +-1 grid cell independently.  On an N = 128 grid with per_row
    close to N that creates collisions, and the deduplication then DELETES the
    duplicates, so the realised count falls well below `n`:

        p     n_actual (seeds 0,1,2)      requested
        3     662 / 651 / 664             784
        4     625 / 615 / 611             784
        8     625 / 615 / 611             784
        10    676 / 668 / 680             784
        12    681 / 686 / 701             784
        16    740 / 746 / 757             784

    The loss is 3-22% and it SHRINKS AS p GROWS, i.e. the sensor count rises
    monotonically with the band count.  Any stripe experiment whose stated
    purpose is "the sensor count is held fixed so the threshold cannot be a
    counting effect" is invalidated by that: count and outcome move together.
    It is also pointless -- the jitter acts only in x, while the aliasing the
    stripe family probes is the comb in y, which it leaves untouched.

    `jitter_mode="rowshift"` (use this) gives every occupied row an independent
    uniform offset in x.  It breaks the row-to-row alignment exactly as intended
    and creates no collisions, so `n_actual == n` for every p and every seed.
    The delta_K structure is preserved; measured at N = 128, n = 784, K = 5,
    denom_floor = 0:

        p      delta_K (rowshift)     n_actual
        8      -0.0247                784
        10     -0.0010                784
        11     +0.0727                784      <- p* = 2 K_c + 1
        12     +0.1077                784
        16     +0.2829                784

    The sign flip still lands on p*.  Note the magnitude above p* is smaller
    than the point-jitter numbers previously reported (+0.206 at p = 11): part
    of that value came from the collisions, so any delta_K quoted for a stripe
    layout must state its jitter mode.

    `strict_count=True` raises instead of silently returning fewer than `n`
    sensors.  Turn it on in any experiment that claims a fixed sensor budget.

    Returns (ix, iy, meta) with
        meta['thickness']    band depth in grid rows
        meta['gap']          empty corridor width in physical units
        meta['n_actual']     realised sensor count
        meta['jitter_mode']  which mode was used (record it in results.json)
        meta['on_stripe']    fraction of sensors inside a band (1.0 by construction)
    """
    rng = np.random.default_rng(seed)
    per_band = int(np.ceil(n / n_stripes))
    thickness = int(np.ceil(per_band / N))
    band_rows = N // n_stripes
    if thickness >= band_rows:
        raise ValueError(
            f"n={n} over {n_stripes} stripes needs thickness {thickness} but "
            f"the band pitch is only {band_rows} rows: no corridor would be "
            f"left.  Lower n or lower n_stripes.")
    per_row = int(np.ceil(per_band / thickness))
    if per_row > N:
        raise ValueError(f"per_row={per_row} exceeds N={N}: cannot place "
                         f"{n} sensors in {n_stripes} bands without duplication")
    mode = (jitter_mode if jitter else None)
    if mode not in (None, "point", "rowshift"):
        raise ValueError(f"unknown jitter_mode {jitter_mode!r}")
    ix, iy = [], []
    base = np.round(np.linspace(0, N, per_row, endpoint=False)).astype(int) % N
    for j in range(n_stripes):
        r0 = int(round((j + 0.5) * N / n_stripes)) - thickness // 2
        for t in range(thickness):
            row = (r0 + t) % N
            if mode == "point":
                xs = (base + rng.integers(-1, 2, per_row)) % N
            elif mode == "rowshift":
                xs = (base + int(rng.integers(0, N))) % N
            else:
                xs = base
            ix.extend(xs.tolist())
            iy.extend([row] * per_row)
    ix, iy = _dedup(np.asarray(ix), np.asarray(iy), N)
    if strict_count and len(ix) < n:
        raise RuntimeError(
            f"stripes_exact(p={n_stripes}, jitter_mode={mode!r}) realised "
            f"{len(ix)}/{n} sensors.  Use jitter_mode='rowshift' (collision "
            f"free) or lower n.")
    ix, iy = ix[:n], iy[:n]
    gap = (band_rows - thickness) * 2 * np.pi / N
    meta = dict(family="stripes_exact", n_stripes=n_stripes,
                thickness=thickness, per_row=per_row,
                gap=float(gap), n_actual=int(len(ix)), on_stripe=1.0,
                jitter_mode=mode, n_requested=int(n),
                count_exact=bool(len(ix) == n),
                note="no random fill; corridor is real")
    return ix, iy, meta


def corridor(N, n, gap_width, seed=0, angle=0.0):
    """Uniform random sensors everywhere EXCEPT one empty strip of the given
    physical width, oriented at `angle` radians.

    This is the clean one-knob probe of the "geometric residual" hypothesis:
    coverage radius grows as gap_width/2, but so does a single, purely
    directional feature, so it separates a scalar-h explanation from a
    corridor-width explanation without changing the sensor count.
    """
    rng = np.random.default_rng(seed)
    nx, ny = np.cos(angle), np.sin(angle)
    ix, iy = [], []
    seen = set()
    tries = 0
    while len(ix) < n and tries < 400:
        tries += 1
        xs = rng.uniform(0, 2 * np.pi, 4 * n)
        ys = rng.uniform(0, 2 * np.pi, 4 * n)
        # signed distance to the corridor centre line through (pi, pi)
        d = (xs - np.pi) * nx + (ys - np.pi) * ny
        d = np.mod(d + np.pi, 2 * np.pi) - np.pi
        keep = np.abs(d) > gap_width / 2
        a = np.round(xs[keep] * N / (2 * np.pi)).astype(int) % N
        b = np.round(ys[keep] * N / (2 * np.pi)).astype(int) % N
        for p, q in zip(a, b):
            k = int(p) * N + int(q)
            if k not in seen:
                seen.add(k)
                ix.append(int(p))
                iy.append(int(q))
                if len(ix) >= n:
                    break
    if len(ix) < n:
        raise RuntimeError(f"corridor: only placed {len(ix)}/{n}")
    meta = dict(family="corridor", gap_width=float(gap_width),
                angle=float(angle), n_actual=n)
    return np.asarray(ix[:n]), np.asarray(iy[:n]), meta


def lattice_m(N, m):
    """Exact m x m regular lattice (no padding).

    The padding in `nolab.configs.lattice` (it tops a floor(sqrt(n))^2 grid up
    to n with uniform random points) destroys the exact aliasing structure that
    makes the regular lattice analytically tractable, so the aliasing experiment
    must use this generator instead.
    """
    xs = (np.arange(m)[:, None] + 0.5) * N / m
    ys = (np.arange(m)[None, :] + 0.5) * N / m
    ix = np.round(np.broadcast_to(xs, (m, m)).ravel()).astype(int) % N
    iy = np.round(np.broadcast_to(ys, (m, m)).ravel()).astype(int) % N
    ix, iy = _dedup(ix, iy, N)
    meta = dict(family="lattice_m", m=int(m), n_actual=int(len(ix)),
                nyquist_K=(m - 1) // 2,
                note="resolves |k| <= (m-1)//2 exactly; aliases above it")
    return ix, iy, meta
