"""stripes_v2 -- the stripe generator with the count deficit spread instead of
truncated off the tail.

WHY THIS FILE EXISTS
--------------------
`nolab.configs_fix.stripes_exact(..., jitter_mode="rowshift", strict_count=True)`
lays out `p * thickness` rows of `per_row = ceil(ceil(n/p)/thickness)` sensors
each, which overshoots the requested `n`, and then enforces the count with

    ix, iy = ix[:n], iy[:n]

The overshoot is `deficit = p * thickness * per_row - n`, and the truncation
deletes the LAST `deficit` sensors, which all live in the last band at
CONSECUTIVE x positions.  So the layout that is advertised as "p identical
bands, count held fixed at 784" is actually "p-1 identical bands plus one band
with a hole `deficit * N / per_row` grid cells wide".

Measured at N = 128, n = 784, jitter_mode = "rowshift", seed 0:

    p    total  deficit   h (tail-trunc)   h (deficit spread)
    10     790        6      0.347             0.310
    12     792        8      0.405             0.264
    13     793        9      0.491             0.250
    14     784        0      0.250             0.250
    15     795       11      0.396             0.220
    16     784        0      0.202             0.202
    18     792        8      0.347             0.208

The hole dominates the covering radius (p = 13 and p = 15 look WORSE than
p = 12), and it dominates delta_K.  With K = 7 (nu = 2.5e-3, p* = 15):

    p     delta_K, tail-trunc     delta_K, deficit spread
    10        -0.0792                  -0.0770
    12        -0.0834                  -0.0215
    13        -0.1157                  -0.0064
    14        -0.0020                  -0.0020     (deficit 0: identical)
    15        -0.1133                  +0.0757     <- p* = 2 K_c + 1
    16        +0.0838                  +0.0838     (deficit 0: identical)
    18        +0.0049                  +0.0772

Tail truncation makes delta_K NON-MONOTONE in p and puts the sign flip in the
wrong place; the only two p values it leaves alone are exactly the two with
deficit 0 (p = 14 and 16).  Spreading the deficit restores monotonicity and puts
the sign flip EXACTLY on p*, at all three viscosities:

    K_c = 4, p* =  9:  p=8 -0.00000  ->  p=9  +0.3877
    K_c = 5, p* = 11:  p=10 -0.00103 ->  p=11 +0.2610
    K_c = 7, p* = 15:  p=14 -0.00199 ->  p=15 +0.0757

i.e. a two-to-five order of magnitude jump located at the parameter-free
threshold, which is a much sharper statement than the one currently in
RESULTS.md (which quotes +0.073 at p = 11 from the tail-truncated layouts).

The dynamics in results/37_stripe_nyquist_fixedn ran on the tail-truncated
layouts, so its plateaus are pessimistic and its delta_K column cannot be used
to order the family.  Rerun with this generator (scripts/41).

WHAT "SPREAD" DOES
------------------
It removes `deficit` sensors, at most one per row, from rows chosen round-robin
and at x indices strided by a coprime step, so that no two removals are adjacent
and no band loses more than one sensor.  For p values where the deficit is zero
the output is bit-identical to `stripes_exact(..., strict_count=True)`.
"""

import numpy as np

_STRIDE = 7919          # prime, coprime to any per_row used here


def stripes_v2(N, n, n_stripes, seed=0, jitter_mode="rowshift"):
    """p horizontal bands, exactly `n` sensors, deficit spread over rows.

    jitter_mode:
        "rowshift"  each occupied row gets an independent uniform x offset
                    (breaks row-to-row alignment, leaves the y comb intact,
                    creates no collisions).  This is the one to use.
        "none"      no jitter, perfectly aligned columns.
        "point"     NOT PROVIDED.  Per-sensor jitter collides and then loses
                    3-22% of the sensors to deduplication; use
                    nolab.configs_fix.stripes_exact if you need it for
                    bit-exact reproduction of results/19, /30, /31, /33.

    Returns (ix, iy, meta).  meta records everything a results file must carry
    for the layout to be reconstructible: thickness, per_row, deficit, the
    number of rows that lost a sensor, gap (from the TRUE band pitch, not from
    floor division), and the realised count.
    """
    if jitter_mode not in ("rowshift", "none"):
        raise ValueError(f"jitter_mode {jitter_mode!r} not supported here")
    rng = np.random.default_rng(seed)
    per_band = int(np.ceil(n / n_stripes))
    thickness = int(np.ceil(per_band / N))
    band_pitch = N / n_stripes                     # true pitch, not N // p
    if thickness >= band_pitch:
        raise ValueError(
            f"n={n} over {n_stripes} stripes needs thickness {thickness} but "
            f"the band pitch is only {band_pitch:.2f} rows: no corridor left")
    per_row = int(np.ceil(per_band / thickness))
    if per_row > N:
        raise ValueError(f"per_row={per_row} exceeds N={N}")

    base = np.round(np.linspace(0, N, per_row, endpoint=False)).astype(int) % N
    rows = []
    for j in range(n_stripes):
        r0 = int(round((j + 0.5) * band_pitch)) - thickness // 2
        for t in range(thickness):
            row = (r0 + t) % N
            off = int(rng.integers(0, N)) if jitter_mode == "rowshift" else 0
            rows.append((row, (base + off) % N))

    total = len(rows) * per_row
    deficit = total - n
    if deficit < 0:
        raise RuntimeError(f"internal: total {total} < n {n}")
    if deficit > len(rows):
        raise RuntimeError(
            f"deficit {deficit} exceeds the number of rows {len(rows)}: more "
            f"than one sensor per row would have to go.  Choose n closer to "
            f"{total} (e.g. n = {total}) or change n_stripes.")

    drop = {}
    for i in range(deficit):
        drop[i % len(rows)] = (i * _STRIDE) % per_row

    ix, iy = [], []
    for i, (row, xs) in enumerate(rows):
        keep = np.ones(per_row, bool)
        if i in drop:
            keep[drop[i]] = False
        ix.extend(xs[keep].tolist())
        iy.extend([row] * int(keep.sum()))
    ix = np.asarray(ix, int)
    iy = np.asarray(iy, int)
    if len(ix) != n:
        raise RuntimeError(f"internal: placed {len(ix)} != {n}")

    gap = (band_pitch - thickness) * 2 * np.pi / N
    meta = dict(family="stripes_v2", n_stripes=int(n_stripes),
                thickness=int(thickness), per_row=int(per_row),
                band_pitch_rows=float(band_pitch),
                gap=float(gap), deficit=int(deficit),
                rows_losing_one=int(len(drop)),
                rows_occupied=int(len(rows)),
                n_actual=int(len(ix)), count_exact=True,
                jitter_mode=jitter_mode, seed=int(seed),
                nyquist_K=(n_stripes - 1) // 2,
                note=("deficit spread one-per-row; identical to "
                      "stripes_exact(strict_count=True) when deficit == 0"))
    return ix, iy, meta
