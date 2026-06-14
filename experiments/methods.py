"""Two-sample tests built on the dKS statistic.

Naming (item 1 — used everywhere, code and plot legends):

  * dks.exact  -> "exact-sample dKS"  (EXACT_LABEL)
        Brute-force O(n^2) dKS on the pooled sample. It is exact for the SAMPLE,
        not for the population. Never call it "baseline" or plain "exact dKS".

  * dks.approx -> "Sample-Sketch-Solve dKS"  (SSS_LABEL, short tag "SSS-dKS")
        Jeff's framework name for the O(n log n) grid approximation.

Procedures:

  * permutation_test(P, Q, stat_fn, ...) -- relabels the pooled sample to build
    the null distribution of `stat_fn`; exactly valid for any B, needs no
    analytic constant. This is the rigorous reference. Used two ways: with
    stat_fn = exact_stat ("exact-sample dKS") and stat_fn = sss_stat ("SSS-dKS").

  * sss_direct_test -- B-free: reject when dks.approx exceeds the clean analytic
    threshold tau = 2*sqrt(ln(1/delta)/n). Labeled "SSS-dKS direct (PENDING
    Peter's C2)" -- see sss_direct_threshold for the open constant question.

The older threshold_d2 / direct_test below are kept for reference: they encode
the paper's Section 6.2 form (with an extra ln(2n) factor). run_calibration.py
is what decides empirically which constant is right.
"""
import time

import numpy as np
import dks

# --- item 1: canonical names, used in code and plot legends -----------------
EXACT_LABEL = "exact-sample dKS"
SSS_LABEL = "Sample-Sketch-Solve dKS"
SSS_TAG = "SSS-dKS"
SSS_DIRECT_LABEL = "SSS-dKS direct (PENDING Peter's C2)"

# stable method ids (CSV / plot keys) and their display labels
METHOD_EXACT = "exact_sample"
METHOD_SSS = "sss"
METHOD_SSS_DIRECT = "sss_direct"
LABELS = {
    METHOD_EXACT: EXACT_LABEL,
    METHOD_SSS: SSS_LABEL,
    METHOD_SSS_DIRECT: SSS_DIRECT_LABEL,
}
LEGEND = {  # short legend tags
    METHOD_EXACT: EXACT_LABEL,
    METHOD_SSS: SSS_TAG,
    METHOD_SSS_DIRECT: "SSS-dKS direct",
}


# --- the two statistics -----------------------------------------------------
def exact_stat(P, Q):
    """exact-sample dKS: brute-force O(n^2) dKS on the given samples."""
    return dks.exact(P, Q)


def sss_stat(P, Q, eps=-1.0):
    """Sample-Sketch-Solve dKS (SSS-dKS): O(n log n) grid approximation.

    eps <= 0 uses the default 2*sqrt(n) grid.
    """
    return dks.approx(P, Q, eps)


# --- item 2: permutation test driven by an arbitrary statistic --------------
def permutation_test(P, Q, stat_fn, B, delta, rng, randomized=False):
    """Permutation two-sample test for an arbitrary dKS statistic `stat_fn`.

    T_obs = stat_fn(P, Q).  Pool the 2n points; for b in 1..B-1 shuffle the pool,
    split into the first n / last n, and recompute T_b = stat_fn(., .).  Together
    with T_obs that is B exchangeable values under H0.

        conservative p = (1 + #{b: T_b >= T_obs}) / B        # default, level <= delta
        reject         = p <= delta

    Randomized tie-breaking (randomized=True): flip a coin when T_obs sits
    exactly on the (1 - delta) quantile, which gives EXACTLY level delta
    (Hemerik & Goeman, 2018, "Exact testing with random permutations"):

        p_rand = (#{T_b > T_obs} + U * (1 + #{T_b == T_obs})) / B,   U ~ Uniform(0,1)

    The conservative form is the safe default for reported numbers; randomized is
    only useful when ties are common (discrete statistics).

    Returns (reject, p_value, T_obs, elapsed_seconds); the whole call is timed.
    """
    t0 = time.perf_counter()
    P = np.ascontiguousarray(P, dtype=float)
    Q = np.ascontiguousarray(Q, dtype=float)
    n = len(P)
    pool = np.vstack([P, Q])
    N = pool.shape[0]

    T_obs = stat_fn(P, Q)

    T_perm = np.empty(B - 1)
    for b in range(B - 1):
        idx = rng.permutation(N)
        T_perm[b] = stat_fn(pool[idx[:n]], pool[idx[n:]])

    if randomized:
        gt = int(np.count_nonzero(T_perm > T_obs))
        eq = int(np.count_nonzero(T_perm == T_obs))
        p_value = (gt + rng.random() * (1.0 + eq)) / B
    else:
        ge = int(np.count_nonzero(T_perm >= T_obs))
        p_value = (1.0 + ge) / B

    reject = bool(p_value <= delta)
    elapsed = time.perf_counter() - t0
    return reject, float(p_value), float(T_obs), elapsed


# --- item 2: the direct (B-free) SSS method ---------------------------------
def sss_direct_threshold(n, delta):
    """Clean closed-form rejection threshold for the SSS-dKS direct test (d=2).

    Invert the population tail bound  delta = exp(-n * eps^2 / 4):

        tau = 2 * sqrt( ln(1/delta) / n )

    This is the CLEAN form (no ln(2n) factor).  The paper's Section 6.2 constant
    carries an extra ln(2n) (see threshold_d2 below).  Which one is correct is
    exactly what run_calibration.py settles empirically -- if the empirical null
    tail hugs exp(-n eps^2 / 4), the clean form is the right inversion.

    PENDING Peter's C2: he may use a different direct variant / exact constant;
    everything here is parameterized so the confirmed form drops straight in.
    """
    return 2.0 * np.sqrt(np.log(1.0 / delta) / int(n))


def sss_direct_test(P, Q, delta, eps=-1.0):
    """SSS-dKS direct (PENDING Peter's C2): B-free two-sample test.

    reject  iff  dks.approx(P, Q) > tau(n, delta),  tau = 2*sqrt(ln(1/delta)/n).

    No permutations, so it costs a single approx evaluation -- the cheap method.
    Returns (reject, stat, tau, elapsed_seconds); the whole call is timed.
    """
    t0 = time.perf_counter()
    P = np.ascontiguousarray(P, dtype=float)
    Q = np.ascontiguousarray(Q, dtype=float)
    n = len(P)
    tau = sss_direct_threshold(n, delta)
    stat = dks.approx(P, Q, eps)
    reject = bool(stat > tau)
    elapsed = time.perf_counter() - t0
    return reject, float(stat), float(tau), elapsed


# ---------------------------------------------------------------------------
# Kept for reference: the paper's Section 6.2 form (extra ln(2n) factor) and a
# convenience wrapper.  Not used by the new power experiment, but retained so
# the two candidate constants can be compared head-to-head.
# ---------------------------------------------------------------------------
def dks_stat(P, Q, exact=True, eps=-1.0):
    """dKS statistic. exact=True -> O(n^2) exact; else O(n log n) approx."""
    return dks.exact(P, Q) if exact else dks.approx(P, Q, eps)


def threshold_d2(n, delta, C2=None):
    """PROVISIONAL Section 6.2 threshold eps(n, delta) for d = 2 (extra ln(2n)).

    Default C2 = 4*ln(2n); functional form sqrt((C2 + ln(2/delta)) / (2n)).
    Contrast with sss_direct_threshold (the clean 2*sqrt(ln(1/delta)/n) form).
    """
    n = int(n)
    if C2 is None:
        C2 = 4.0 * np.log(2.0 * n)
    return float(np.sqrt((C2 + np.log(2.0 / delta)) / (2.0 * n)))


def direct_test(P, Q, delta, C2=None, exact=True, eps=-1.0):
    """Direct (B-free) test with the Section 6.2 threshold_d2 constant."""
    P = np.ascontiguousarray(P, dtype=float)
    Q = np.ascontiguousarray(Q, dtype=float)
    n = min(len(P), len(Q))
    thr = threshold_d2(n, delta, C2)
    stat = dks_stat(P, Q, exact, eps)
    return {"stat": stat, "threshold": thr, "reject": stat >= thr}
