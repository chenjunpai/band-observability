"""The convergence criterion the project has been missing.

WHY THIS FILE EXISTS
--------------------
`nolab.metrics.sync_rate` decides `converged` from

    converged = decayed and rate >= 0.05 and r2 >= 0.5
    decayed   = (e.min() <= 0.5 * e[0])

i.e. "the error halved at some point, and the log-linear fit over the window
0.9*e0 > e > 3*e_min has a negative slope".  It never looks at where the error
ENDS UP.  The v2 docstring says this was deliberate ("converged no longer
depends on the final error at time T"), but that removed the only test that
actually measures synchronisation.

Measured consequence, from `results/27_rate_stability` (nu = 5e-3, T = 8, no
observation noise, no model error, i.e. the idealised setting where a genuinely
synchronising observer reaches 1e-5):

    uniform n=784  mu=50            e0=1.31  e_min=1.2e-5  e_end=1.2e-5   rate 1.49  "converged"
    uniform n=300  mu=150 seed 1    e0=1.23  e_min=2.1e-1  e_end=4.5e-1   rate 6.27  "converged"
    uniform n=300  mu=200 init 2    e0=1.19  e_min=2.1e-1  e_end=5.0e-1   rate 26.3  "converged"
    uniform n=400  mu=200 init 2    e0=1.17  e_min=1.6e-1  e_end=2.5e-1   rate 17.9  "converged"

The last three never synchronise: the error drops by one order of magnitude and
then sits on a plateau at 0.05-0.5 for the rest of the run.  `sync_rate` fits
the short steep segment between e = 0.9*e0 and e = 3*e_min -- a sliver of the
initial transient -- and extrapolates its slope into a "rate" of 26.

Everything downstream inherits this.  In particular the two counterexamples
that `POSITIONING_v4.md` builds its main line on (§1.1 "uniform n=300 converges
at the same h where n=200 fails", §1.2 "uniform n=400 converges where lattice
m=10 with smaller h fails") are both artefacts: n=300 plateaus at 5.8e-2 and
n=400 at 1.2e-1.  Neither converges.  Replacements that survive this criterion
are in AUDIT_v4.md §2.

WHAT TO USE INSTEAD
-------------------
A three-way verdict on the PLATEAU, not on the transient slope:

    SYNC     e_inf / e_0 <  sync_tol      (default 1e-2)
    PARTIAL  e_inf / e_0 <  partial_tol   (default 0.15)
    FAIL     otherwise
    DIVERGED non-finite, or the error grew past the divergence factor

and `rate` is reported only when the verdict is SYNC.  In the PARTIAL and FAIL
bands the exponential model does not describe the curve, so the slope of any
window through it is not a rate of anything.

PARTIAL is not a hedge; it is a real state.  The observer has locked the large
scales and not the small ones, and the plateau height is the quantity that
distinguishes layouts inside that band.  It deserves its own column.

CHOOSING sync_tol
-----------------
With perfect observations, a perfect model and assimilation at every step, a
synchronising observer reaches 1e-5 (measured, uniform n = 784).  1e-2 is
therefore a loose threshold with two orders of margin.  It is NOT loose once
noise or model error is present, because those set a physical floor:

    observation noise 1%      plateau ~ 6e-4      (results/28)
    observation noise 5%      plateau ~ 3e-3      (results/28)
    model error nu +- 10%     plateau ~ 1.2e-2    (results/28)
    dt_obs = 0.1              plateau ~ 2e-2      (results/28)

Under any of those, pass `sync_tol` explicitly at a few times the floor, or use
`floor_from_conditions` below.  Comparing a noisy run against the default 1e-2
would call model error a failure of the layout.
"""

import numpy as np

SYNC_TOL = 1e-2
PARTIAL_TOL = 0.15
E_REF_DEFAULT = 1.3      # measured e[0] over the whole repository: 1.11 - 1.38


def _tail_median(e, tail_frac=0.25):
    """Median of the last `tail_frac` of the curve.

    The median, not the minimum: `e.min()` is a single sample and on a noisy
    plateau it is routinely 2-4x below the plateau level (see the curves in
    results/27_rate_stability, where the plateau oscillates by a factor of 3).
    Using min() to decide convergence reintroduces exactly the optimism this
    module exists to remove.
    """
    e = np.asarray(e, float)
    if e.size == 0:
        return None
    k = max(int(round(tail_frac * e.size)), 1)
    return float(np.median(e[-k:]))


def outcome(ts, errs, e_ref=None, sync_tol=SYNC_TOL, partial_tol=PARTIAL_TOL,
            tail_frac=0.25, diverge_factor=1e6):
    """Verdict for a run whose full error curve is available.  USE THIS in every
    new script; store its output next to `rate`, never instead of the curve.

    Returns dict(verdict, plateau, plateau_rel, e0, rate_is_meaningful, n).
    """
    e = np.asarray(errs, float)
    out = dict(verdict="NO_DATA", plateau=None, plateau_rel=None, e0=None,
               rate_is_meaningful=False, n=int(e.size))
    if e.size < 4:
        return out
    if not np.all(np.isfinite(e)) or float(e.max()) > diverge_factor:
        out.update(verdict="DIVERGED")
        return out
    e0 = float(e[0]) if e_ref is None else float(e_ref)
    if e0 <= 0:
        return out
    tail = _tail_median(e, tail_frac)
    rel = tail / e0
    v = ("SYNC" if rel < sync_tol else
         ("PARTIAL" if rel < partial_tol else "FAIL"))
    out.update(verdict=v, plateau=tail, plateau_rel=float(rel), e0=e0,
               rate_is_meaningful=bool(v == "SYNC"))
    return out


def verdict_from_final(final, e_ref=E_REF_DEFAULT, sync_tol=SYNC_TOL,
                       partial_tol=PARTIAL_TOL, diverged=False,
                       diverge_factor=1e6):
    """Verdict for a LEGACY result that stored only `final` (the error at T).

    Every results file written before this module stored `final` but discarded
    the curve, so this is the only reclassification possible without rerunning.
    It is weaker than `outcome` in one specific way: `final` is a single sample
    of an oscillating plateau, so a row can be misgraded by roughly the plateau's
    own oscillation amplitude (a factor of ~3 in the measured curves).

    That matters only inside the bands.  `29_reclassify.py` therefore marks any
    row whose plateau_rel lands within a factor `AMBIGUITY` of either threshold
    as needing a curve rerun, and `33_fig1_curves.py` reruns those.
    """
    out = dict(verdict="NO_DATA", plateau=None, plateau_rel=None,
               e_ref=float(e_ref), rate_is_meaningful=False)
    if diverged:
        out.update(verdict="DIVERGED")
        return out
    if final is None:
        return out                       # NO_DATA: the run stored no error
    if not np.isfinite(final):
        out.update(verdict="DIVERGED")
        return out
    f = float(final)
    if f > diverge_factor:
        out.update(verdict="DIVERGED")
        return out
    rel = f / float(e_ref)
    v = ("SYNC" if rel < sync_tol else
         ("PARTIAL" if rel < partial_tol else "FAIL"))
    out.update(verdict=v, plateau=f, plateau_rel=float(rel),
               rate_is_meaningful=bool(v == "SYNC"))
    return out


AMBIGUITY = 2.0     # half the measured plateau oscillation amplitude (~3x)


def is_ambiguous(plateau_rel, sync_tol=SYNC_TOL, partial_tol=PARTIAL_TOL,
                 factor=AMBIGUITY):
    """True when a single-sample `final` could have landed on either side of a
    threshold given the plateau's own oscillation."""
    if plateau_rel is None:
        return True
    for t in (sync_tol, partial_tol):
        if t / factor <= plateau_rel <= t * factor:
            return True
    return False


def floor_from_conditions(noise=0.0, nu_err=0.0, dt_obs=None):
    """Order-of-magnitude plateau floor implied by the run conditions, from the
    measurements in results/28_robustness_noise.  Use `3 * floor` as sync_tol
    when any of these is nonzero; the default 1e-2 is only valid for the
    idealised (noise = 0, exact model, m_assim = 1) setting.
    """
    f = 1e-5
    if noise > 0:
        f = max(f, 0.06 * noise)          # 1% -> 6e-4, 5% -> 3e-3 (measured)
    if abs(nu_err) > 0:
        f = max(f, 1.2e-2)                # +-10% in nu (measured)
    if dt_obs is not None and dt_obs > 0.01:
        f = max(f, 2e-2 * (dt_obs / 0.1))
    return float(f)


def best_over(records, key="plateau_rel"):
    """Best (lowest plateau) record of a sweep, plus whether it sits at an edge.

    Replaces `max over mu of rate`, which is upward-biased by the noise in the
    rate estimator and, worse, maximises a quantity that is not defined outside
    the SYNC band.
    """
    rs = [r for r in records if r.get(key) is not None]
    if not rs:
        return dict(best=None, best_index=None, at_edge=None, n=0)
    vals = [r[key] for r in rs]
    i = int(np.argmin(vals))
    return dict(best=float(vals[i]), best_index=i, best_record=rs[i],
                at_edge=bool(i in (0, len(vals) - 1)), n=len(vals))


VERDICT_ORDER = {"DIVERGED": 0, "FAIL": 1, "PARTIAL": 2, "SYNC": 3,
                 "NO_DATA": -1}


def best_verdict(verdicts):
    """The most favourable verdict in a sweep (SYNC > PARTIAL > FAIL > DIVERGED)."""
    if not verdicts:
        return "NO_DATA"
    return max(verdicts, key=lambda v: VERDICT_ORDER.get(v, -1))
