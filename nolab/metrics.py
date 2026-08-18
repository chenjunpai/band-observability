"""Synchronisation metrics.

v2 fixes versus v1:
  * `converged` no longer depends on the final error at time T (v1 used
    errs[-1] < 1e-3, which made N_c a function of the window);
  * rate = 0.0 no longer conflates "fit failed" with "no contraction":
    every result carries a status (converged / weak / no_decay / no_fit /
    fast_sync / plateau);
  * every rate has an R^2, a fit count and the fitted window, so a
    polyfit on a non-exponential curve cannot quietly fake a rate.
"""

import numpy as np


def sync_rate(ts, errs, floor=1e-9, c_min=0.05, r2_min=0.5,
              min_decay=0.5):
    """Fit e(t) ~ exp(-c t) on the transient.

    Returns dict(rate, r2, n_fit, status, converged, plateau, t_span).
    `converged` requires a genuine decay by at least `min_decay` relative
    to the initial error, a rate >= c_min, and an exponential fit with
    R^2 >= r2_min.  It never reads the last sample, so it does not drift
    with the time horizon.
    """
    e, t = np.asarray(errs, float), np.asarray(ts, float)
    empty = dict(rate=0.0, r2=0.0, n_fit=0, status="no_fit",
                 converged=False, plateau=None, t_span=None)
    if e.size < 6 or not np.all(np.isfinite(e)):
        return empty
    e0 = e[0]
    if e0 <= 0:
        return empty
    emin = float(e.min())
    plateau = max(3.0 * emin, floor)
    decayed = bool(emin <= min_decay * e0)
    if not decayed and e[-1] >= 0.9 * e0:
        return dict(rate=0.0, r2=0.0, n_fit=0, status="no_decay",
                    converged=False, plateau=plateau, t_span=float(t[-1]))
    m = (e < 0.9 * e0) & (e > plateau)
    if m.sum() < 5:
        if e[-1] < 1e-3:
            return dict(rate=0.0, r2=0.0, n_fit=0, status="fast_sync",
                        converged=True, plateau=plateau,
                        t_span=float(t[-1]))
        return dict(rate=0.0, r2=0.0, n_fit=0, status="no_fit",
                    converged=False, plateau=plateau, t_span=float(t[-1]))
    log_e = np.log(e[m])
    p = np.polyfit(t[m], log_e, 1)
    rate = float(max(-p[0], 0.0))
    pred = np.polyval(p, t[m])
    ss_res = float(np.sum((log_e - pred) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    converged = bool(decayed and rate >= c_min and r2 >= r2_min)
    status = ("converged" if converged else
              ("plateau" if emin <= max(3.0 * floor, 1e-12) else "weak"))
    return dict(rate=rate, r2=r2, n_fit=int(m.sum()), status=status,
                converged=converged, plateau=plateau,
                t_span=float(t[m][-1] - t[m][0]))


def bootstrap_Nc(per_K, M, c_star=0.75, nboot=2000, seed=0):
    """Bootstrap CI for N_c from per-K rate lists (exp13-style).
    per_K: {K: {"ndof": int, "rates": [...]}}.  Returns [lo, hi]."""
    rng = np.random.default_rng(seed)
    Ks = sorted(per_K)
    boots = []
    for _ in range(nboot):
        crossed = None
        for K in Ks:
            rates = np.asarray(per_K[K]["rates"], float)
            idx = rng.integers(0, len(rates), M)
            if np.mean(rates[idx]) > c_star:
                crossed = K
                break
        boots.append(per_K[crossed]["ndof"] if crossed is not None else np.nan)
    b = np.asarray(boots, float)
    if np.all(np.isnan(b)):
        return [None, None]
    return [float(np.nanpercentile(b, 5)), float(np.nanpercentile(b, 95))]


def first_consistent_K(per_K, c_star=0.75):
    """First K whose mean rate crosses c_star AND all larger K stay above.
    Returns (K, ndof, consistent) -- `consistent=False` flags a
    non-monotonic crossing, which must be reported, not hidden."""
    Ks = sorted(per_K)
    means = {K: float(np.mean(per_K[K]["rates"])) for K in Ks}
    for i, K in enumerate(Ks):
        if means[K] >= c_star:
            above = [means[K2] >= c_star for K2 in Ks[i + 1:]]
            consistent = bool(all(above))
            return K, per_K[K]["ndof"], consistent
    return None, None, False
