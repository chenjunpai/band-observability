"""Standard ranking statistics, replacing the ad-hoc counters in scripts 33/24/28.

WHY THIS FILE EXISTS
--------------------
Three headline numbers in v5 are computed by rules that a referee can reasonably
call arbitrary, and two of them change when the rule changes.

1.  `33_boundary_curves.rank_violations` counts a violation only when the layout
    that should be better plateaus MORE THAN 3x worse.  The 3 is a free knob.
    Reported: delta_K 46, h 94.  With the plain rule (every inversion counts)
    the same rows give delta_K 81, h 127 out of 325 pairs.  Same verdict, but
    the published number depends on the knob, so publish a standard statistic
    instead: Kendall's tau (rank correlation) and the AUC of the predictor
    against the binary SYNC label.  Both are parameter free.

2.  `24_mu_frontier` quotes three ratios (rate 2.1x, same-cost 1.4x, accuracy
    1.6x) that are each computed with a different aggregation.  Recomputed from
    the same rows with ONE stated rule (best configuration per method per seed,
    then mean over seeds) the numbers are rate 2.07x, M=8 1.21x, accuracy 2.37x
    -- the accuracy ratio doubles because the 1.6x figure silently restricts
    LETKF to loc = 0.8, while loc = 1.5 is strictly more accurate.  Use
    `frontier_ratios` and state `select` in the caption.

3.  `28_robustness_noise` builds its 27/27 ordering from `max(rate)`, and `rate`
    is the quantity `nolab.verdict` retires outside the SYNC band -- which is
    where most of the noisy runs sit.  `ordering_by` takes the metric as an
    argument so the check can be run on the plateau.  (Recomputed on the
    plateau the answer is still 27/27, so only the script needs fixing, not the
    conclusion.)

Nothing here re-runs a solver; all three are post-processing over existing rows.
"""

import itertools

import numpy as np


# --------------------------------------------------------------- rank quality

def kendall_tau(x, y):
    """Kendall's tau-b between predictor x and outcome y.

    Pass y = plateau_rel and x = delta_K with `sign=-1` folded in by the caller,
    or use `predictor_quality` which handles the orientation.  Ties are handled
    by the tau-b denominator, which matters here because several layouts share
    an h to three decimals.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    conc = disc = tx = ty = 0
    for i, j in itertools.combinations(range(n), 2):
        dx = np.sign(x[i] - x[j])
        dy = np.sign(y[i] - y[j])
        if dx == 0 and dy == 0:
            tx += 1
            ty += 1
        elif dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx == dy:
            conc += 1
        else:
            disc += 1
    n0 = conc + disc
    den = np.sqrt((n0 + tx) * (n0 + ty))
    return float((conc - disc) / den) if den > 0 else 0.0


def auc(scores, labels):
    """Area under the ROC curve of `scores` against the binary `labels`.

    Rank-based (Mann-Whitney) so it needs no threshold.  Use labels = 1 for
    SYNC.  AUC = 1 means the predictor separates SYNC from non-SYNC perfectly;
    0.5 means it carries no information.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for k in np.flatnonzero(cnt > 1):
        m = inv == k
        ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def inversion_count(x, y, sign, tol=1.0):
    """Pairs where the predictor is wrong.  `tol=1` is the plain rule; the v5
    scripts used tol=3 (only count when the outcome is >3x worse).  Both are
    returned by `predictor_quality` so the sensitivity to the knob is visible.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    bad = 0
    tot = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        if x[i] == x[j] or y[i] == y[j]:
            continue
        tot += 1
        pred_i_better = (x[i] > x[j]) if sign > 0 else (x[i] < x[j])
        worse_ratio = (y[i] / y[j]) if pred_i_better else (y[j] / y[i])
        if worse_ratio > tol:
            bad += 1
    return dict(inversions=bad, pairs=tot, tol=float(tol))


def predictor_quality(rows, predictor, sign, outcome="best_plateau_rel",
                      sync_key="best_verdict", tols=(1.0, 3.0)):
    """Everything Fig 1 needs to justify its x axis, for one candidate x axis.

    sign = +1 when larger predictor should mean lower plateau (delta_K),
    sign = -1 when smaller predictor should mean lower plateau (h).
    """
    rs = [r for r in rows if r.get(outcome) is not None
          and r.get(predictor) is not None]
    x = [r[predictor] for r in rs]
    y = [r[outcome] for r in rs]
    lab = [1 if r.get(sync_key) == "SYNC" else 0 for r in rs]
    out = dict(predictor=predictor, n=len(rs),
               tau=kendall_tau([sign * v for v in x], [-v for v in y]),
               auc=auc([sign * v for v in x], lab))
    for t in tols:
        out[f"inversions_tol{t:g}"] = inversion_count(x, y, sign, t)["inversions"]
    out["pairs"] = inversion_count(x, y, sign, 1.0)["pairs"]
    return out


# ------------------------------------------------------------------- frontier

def frontier_ratios(rows, method_key="method", nudging="nudging",
                    baseline="LETKF", seed_key="seed",
                    rate_key="rate", err_key="final", cost_key="solver_steps",
                    select=None, require_bracketed=True):
    """LETKF-vs-nudging ratios under ONE aggregation rule.

    Rule: within each method (optionally restricted by `select`, a predicate on
    a row), take the best row per seed -- best rate for the rate ratio, lowest
    final error for the accuracy ratio -- then average over seeds.  Report the
    two ratios plus the cost ratio of the configurations that produced them.

    `require_bracketed` checks that the nudging optimum is interior to the mu
    sweep; if it is not, the ratio is a lower bound and the flag says so.  (In
    results/24 the optimum IS interior: rate peaks at mu = 400-800 and mu = 1600
    diverges, so this passes -- unlike results/20, where it did not.)
    """
    def sub(m):
        rs = [r for r in rows if r.get(method_key) == m and not r.get("diverged")]
        return [r for r in rs if select is None or select(r)]

    def per_seed(rs, key, best="max"):
        out = {}
        for r in rs:
            v = r.get(key)
            if v is None:
                continue
            s = r.get(seed_key)
            cur = out.get(s)
            if cur is None or (v > cur[0] if best == "max" else v < cur[0]):
                out[s] = (v, r)
        return out

    nud, base = sub(nudging), sub(baseline)
    if not nud or not base:
        return dict(error="one side is empty after `select`")
    nr, br = per_seed(nud, rate_key, "max"), per_seed(base, rate_key, "max")
    ne, be = per_seed(nud, err_key, "min"), per_seed(base, err_key, "min")
    seeds = sorted(set(nr) & set(br) & set(ne) & set(be))
    if not seeds:
        return dict(error="no seed present on both sides")

    rate_n = float(np.mean([nr[s][0] for s in seeds]))
    rate_b = float(np.mean([br[s][0] for s in seeds]))
    err_n = float(np.mean([ne[s][0] for s in seeds]))
    err_b = float(np.mean([be[s][0] for s in seeds]))
    cost_n = float(np.mean([nr[s][1].get(cost_key, np.nan) for s in seeds]))
    cost_b = float(np.mean([br[s][1].get(cost_key, np.nan) for s in seeds]))

    bracketed = None
    if require_bracketed:
        mus = sorted({r["mu"] for r in nud if "mu" in r})
        best_mus = {nr[s][1].get("mu") for s in seeds}
        bracketed = bool(mus and all(m not in (mus[0], mus[-1])
                                     for m in best_mus if m is not None))

    return dict(seeds=seeds,
                rate_nudging=rate_n, rate_baseline=rate_b,
                rate_ratio=rate_b / rate_n if rate_n else None,
                err_nudging=err_n, err_baseline=err_b,
                accuracy_ratio=err_n / err_b if err_b else None,
                cost_ratio=cost_b / cost_n if cost_n else None,
                nudging_optimum_bracketed=bracketed,
                rule="best config per seed, then mean over seeds")


# ------------------------------------------------------------------- ordering

def ordering_by(rows, layouts, metric, sign, condition_keys):
    """Layout ranking per condition, using whatever metric you pass.

    sign = -1 for plateau (smaller is better), +1 for rate (larger is better).
    Returns dict(orders, baseline, unchanged, total, differing).
    """
    conds = sorted({tuple(r[k] for k in condition_keys) for r in rows})
    orders = {}
    for c in conds:
        sub = {}
        for name in layouts:
            v = [r[metric] for r in rows
                 if r.get("layout") == name
                 and tuple(r[k] for k in condition_keys) == c
                 and r.get(metric) is not None]
            if not v:
                continue
            sub[name] = (max(v) if sign > 0 else min(v))
        orders[c] = [k for k, _ in sorted(sub.items(),
                                          key=lambda kv: -sign * kv[1])]
    base = orders[conds[0]]
    diff = {c: o for c, o in orders.items() if o != base}
    return dict(orders={str(k): v for k, v in orders.items()},
                baseline=base, unchanged=len(orders) - len(diff),
                total=len(orders),
                differing={str(k): v for k, v in diff.items()})


# ------------------------------------------------------ necessity of delta_K>0

def necessity_audit(rows, predictor="delta_K_floor0", sync_key="best_verdict",
                    threshold=0.0):
    """How many layouts with predictor <= threshold nonetheless reached SYNC.

    This is the statement v5 is currently UNDER-claiming.  Its retraction table
    still lists "delta_K < 0 implies failure" as refuted by uniform n=300/400,
    but under the plateau criterion those two are PARTIAL, not SYNC, so the
    counterexamples are gone: measured over results/19 (57 rows) and results/33
    (26 rows) there is no layout with delta_K < 0 that synchronises.
    """
    below = [r for r in rows if r.get(predictor) is not None
             and r[predictor] <= threshold]
    viol = [r for r in below if r.get(sync_key) == "SYNC"]
    above = [r for r in rows if r.get(predictor) is not None
             and r[predictor] > threshold]
    sync_above = [r for r in above if r.get(sync_key) == "SYNC"]
    pos = [r[predictor] for r in rows if r.get(sync_key) == "SYNC"
           and r.get(predictor) is not None]
    return dict(threshold=float(threshold),
                n_below=len(below), sync_below=len(viol),
                violations=[r.get("layout", r.get("family")) for r in viol],
                n_above=len(above), sync_above=len(sync_above),
                min_delta_K_among_sync=(float(min(pos)) if pos else None))
