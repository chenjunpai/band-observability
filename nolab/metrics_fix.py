"""Rate estimation that does not depend on one fitting protocol, plus the
aggregation rules the v3 tables should have used.

Two problems in the v3 numbers this file addresses.

1.  Fit fragility.  In `results/17_config_ablation` the uniform n = 400 seed 0
    sweep gives rate 1.87 at mu = 10, 0.13 ("weak") at mu = 20 and 1.96 at
    mu = 50.  A factor 15 between neighbouring mu is either real physics or a
    broken fit window, and `sync_rate` cannot tell you which because it returns
    a single number from a single window.  `sync_rate_multi` runs the same curve
    through several protocols and reports the spread, so an unstable estimate is
    visible instead of silently entering a table.

2.  Selection bias.  Every v3 table reports max over the mu sweep.  With 4 mu
    values and a noisy estimator the maximum is biased upward, and it is the
    quantity used for both the layout ranking and the nudging-vs-LETKF gap.
    `aggregate_mu` reports the full curve, the median, and the max together, and
    flags the case that actually matters: the optimum sitting at the edge of the
    sweep (which is what happens to nudging at mu = 200 in
    `results/20_letkf_sweep` -- the reported "LETKF is 2.6x more accurate" is
    measured against a nudging optimum that was never reached).
"""

import numpy as np


def _fit(t, e, lo, hi, robust=False):
    m = (e < lo * e[0]) & (e > hi)
    if m.sum() < 5:
        return None
    x, y = t[m], np.log(e[m])
    if robust:                       # Theil-Sen slope: immune to a few outliers
        idx = np.linspace(0, len(x) - 1, min(len(x), 60)).astype(int)
        xs, ys = x[idx], y[idx]
        sl = [(ys[j] - ys[i]) / (xs[j] - xs[i])
              for i in range(len(xs)) for j in range(i + 1, len(xs))
              if xs[j] != xs[i]]
        slope = float(np.median(sl))
        inter = float(np.median(ys - slope * xs))
    else:
        p = np.polyfit(x, y, 1)
        slope, inter = float(p[0]), float(p[1])
    pred = slope * x + inter
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return dict(rate=float(max(-slope, 0.0)), r2=r2, n_fit=int(m.sum()))


def sync_rate_multi(ts, errs, floor=1e-9):
    """Fit the same error curve under several protocols and report the spread.

    Protocols:
      A  the v2 window                     e < 0.90 e0, e > 3 e_min
      B  a later window (skip the transient) e < 0.50 e0, e > 3 e_min
      C  a tighter floor                    e < 0.90 e0, e > 30 e_min
      D  Theil-Sen on window A              (robust to a few bad samples)

    Returns dict(rate, rate_spread, rates, stable, ...).  `stable` is False when
    max/min over the protocols exceeds 1.5, which is the signal that the number
    should not go into a table unexamined.
    """
    e = np.asarray(errs, float)
    t = np.asarray(ts, float)
    out = dict(rate=0.0, rate_spread=None, rates={}, stable=False,
               status="no_fit", r2=0.0)
    if e.size < 6 or not np.all(np.isfinite(e)) or e[0] <= 0:
        return out
    emin = float(e.min())
    plateau = max(3.0 * emin, floor)
    protocols = dict(
        A=_fit(t, e, 0.90, plateau),
        B=_fit(t, e, 0.50, plateau),
        C=_fit(t, e, 0.90, max(30.0 * emin, floor)),
        D=_fit(t, e, 0.90, plateau, robust=True),
    )
    got = {k: v["rate"] for k, v in protocols.items() if v is not None}
    if not got:
        return out
    vals = np.asarray(list(got.values()), float)
    pos = vals[vals > 1e-6]
    spread = float(pos.max() / pos.min()) if pos.size >= 2 else 1.0
    if not np.isfinite(spread):
        spread = None
    out.update(rate=float(np.median(vals)), rate_spread=spread,
               rates=got, stable=bool(spread is not None and spread <= 1.5),
               r2=float(protocols["A"]["r2"]) if protocols["A"] else 0.0,
               status="converged" if np.median(vals) > 0.05 else "weak")
    return out


def aggregate_mu(rows, key="mu", rate_key="rate"):
    """Summarise a mu sweep honestly.

    Returns dict(best, best_mu, median, curve, at_edge).  `at_edge` is True when
    the best mu is the smallest or largest value swept, i.e. the optimum was not
    bracketed and `best` is a lower bound on the achievable rate, not an optimum.
    """
    rows = sorted(rows, key=lambda r: r[key])
    mus = [float(r[key]) for r in rows]
    rates = [float(r[rate_key]) for r in rows]
    i = int(np.argmax(rates))
    return dict(best=float(rates[i]), best_mu=float(mus[i]),
                median=float(np.median(rates)),
                curve=list(zip(mus, rates)),
                at_edge=bool(i in (0, len(rates) - 1)),
                note=("optimum at sweep edge: extend the sweep before "
                      "quoting this number" if i in (0, len(rates) - 1) else ""))


def Nc_vs_cstar(per_K, c_stars=(0.25, 0.5, 0.75, 1.0, 1.25)):
    """N_c as a function of the (arbitrary) rate threshold c*.

    `N_c` in v3 is the first K whose mean rate crosses c* = 0.75.  That is an
    iso-rate contour, not a synchronisation threshold -- at nu = 1.5e-2 the
    K = 2 (25-mode) observation already has mean rate 0.25 > 0, i.e. the system
    synchronises far below the reported "critical" count.  Any claim about
    N_c(nu) must therefore be shown to be robust in c*, or restated as
    "modes required to reach rate c*".
    """
    out = {}
    Ks = sorted(int(k) for k in per_K)
    for c in c_stars:
        hit = None
        for K in Ks:
            if float(np.mean(per_K[str(K)]["rates"] if str(K) in per_K
                             else per_K[K]["rates"])) >= c:
                hit = K
                break
        rec = per_K[str(hit)] if (hit is not None and str(hit) in per_K) else (
            per_K[hit] if hit is not None else None)
        out[c] = dict(K_c=hit, N_c=(rec["ndof"] if rec else None))
    return out
