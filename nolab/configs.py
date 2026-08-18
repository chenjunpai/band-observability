"""Sensor-configuration generators with EXACT counts.

v1 deduplicated after sampling, so clustered layouts silently ended up
with ~10% fewer sensors than claimed.  Here every generator retries until
the requested count is reached, and `lattice` provides the
covering-optimal (minimum-h) layout as a fifth reference point.
"""

import numpy as np


def _snap_round(N, xs, ys):
    ix = np.round(np.asarray(xs, float) * N / (2 * np.pi)).astype(int) % N
    iy = np.round(np.asarray(ys, float) * N / (2 * np.pi)).astype(int) % N
    return ix, iy


def _exact(N, n, draw, seed, max_rounds=2000):
    rng = np.random.default_rng(seed)
    seen, ix, iy = set(), [], []
    rounds = 0
    while len(ix) < n and rounds < max_rounds:
        xs, ys = draw(rng, n * 4 + 16)
        gx, gy = _snap_round(N, xs, ys)
        for a, b in zip(gx, gy):
            k = int(a) * N + int(b)
            if k not in seen:
                seen.add(k)
                ix.append(int(a))
                iy.append(int(b))
                if len(ix) >= n:
                    break
        rounds += 1
    if len(ix) < n:
        raise RuntimeError(f"could not draw {n} unique sensors "
                           f"(got {len(ix)})")
    return np.asarray(ix, int)[:n], np.asarray(iy, int)[:n]


def uniform(N, n, seed=0):
    """Independent uniform positions (the exp14 training distribution)."""
    return _exact(N, n,
                  lambda rng, m: (rng.uniform(0, 2 * np.pi, m),
                                  rng.uniform(0, 2 * np.pi, m)),
                  seed)


def ground_tracks(N, n, seed=0, n_tracks=8, slope=3.0, jitter=0.02):
    """Satellite-like swaths: dense along a few slanted lines."""
    def draw(rng, m):
        per = max(int(np.ceil(m / n_tracks)), 1)
        xs, ys = [], []
        for b in rng.uniform(0, 2 * np.pi, n_tracks):
            s = np.linspace(0, 2 * np.pi, per)
            xs.append(s + rng.normal(0, jitter, per))
            ys.append((b + slope * s) % (2 * np.pi)
                      + rng.normal(0, jitter, per))
        return np.concatenate(xs), np.concatenate(ys)
    return _exact(N, n, draw, seed)


def clustered(N, n, seed=0, n_clusters=12, width=0.25):
    """Station networks: tight clusters with large empty regions."""
    def draw(rng, m):
        per = max(int(np.ceil(m / n_clusters)), 1)
        cx = rng.uniform(0, 2 * np.pi, n_clusters)
        cy = rng.uniform(0, 2 * np.pi, n_clusters)
        xs = np.concatenate([c + rng.normal(0, width, per) for c in cx])
        ys = np.concatenate([c + rng.normal(0, width, per) for c in cy])
        return xs % (2 * np.pi), ys % (2 * np.pi)
    return _exact(N, n, draw, seed)


def blind_half(N, n, seed=0):
    """Half the domain is never observed at all."""
    return _exact(N, n,
                  lambda rng, m: (rng.uniform(0, np.pi, m),
                                  rng.uniform(0, 2 * np.pi, m)),
                  seed)


def lattice(N, n, seed=0):
    """Covering-optimal (minimum-h) layout: a regular grid plus padding."""
    m = int(np.floor(np.sqrt(n)))
    m = max(m, 1)
    if m * m > n:
        m -= 1
    X = (np.arange(m)[:, None] + 0.5) * 2 * np.pi / m
    Y = (np.arange(m)[None, :] + 0.5) * 2 * np.pi / m
    xs = np.broadcast_to(X, (m, m)).ravel()
    ys = np.broadcast_to(Y, (m, m)).ravel()
    ix = np.round(xs * N / (2 * np.pi)).astype(int) % N
    iy = np.round(ys * N / (2 * np.pi)).astype(int) % N
    if m * m < n:
        pad_ix, pad_iy = uniform(N, n - m * m, seed=seed + 12345)
        ix = np.concatenate([ix, pad_ix])
        iy = np.concatenate([iy, pad_iy])
    return ix[:n], iy[:n]


GENERATORS = dict(uniform=uniform, ground_tracks=ground_tracks,
                  clustered=clustered, blind_half=blind_half,
                  lattice=lattice)
