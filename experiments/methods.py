"""Two-sample tests built on the (2D) dKS statistic.

Naming (constants below; used everywhere, code and plot legends):

  * dks.exact  -> "exact-sample dKS"  (EXACT_LABEL)
        The DIRECT, brute-force O(n^2) computation of dKS on the pooled sample.
        It is "exact for the sample" -- NOT a theoretically-optimal exact 2D
        algorithm, and not a population quantity.  We deliberately keep this
        O(n^2) brute force as the baseline; the experiments show how close (and
        how much faster) the approximation is relative to computing it directly.

  * dks.approx -> "Sample-Sketch-Solve dKS"  (SSS_LABEL, short tag "SSS-dKS")
        Jeff's framework name for the O(n log n) grid approximation.  It is a
        DETERMINISTIC function of the data (fixed 2*sqrt(n) grid, no random
        sub-sampling) -- verified by calling it repeatedly on fixed input -- so
        it is a valid permutation-test statistic with no extra seeding.

Procedures:

  * permutation_test(P, Q, stat_fn, ...) -- relabels the pooled sample to build
    the null. VALID / CONSERVATIVE at level delta for any B; dKS is discrete so
    ties occur (so "conservative", not "exact"). Optional randomized tie-breaking
    (Hemerik & Goeman 2018) attains EXACTLY level delta. This is the validity
    anchor for everything else.

  * sss_direct_test(..., tau_fn) -- B-free: reject when dks.approx exceeds an
    analytic threshold.  TWO candidate thresholds are provided, tau_clean and
    tau_union; NEITHER is claimed valid without proof (PENDING Peter's C2). The
    calibration experiment is the empirical check on which one is honest.
"""
import time

import numpy as np
import dks

# --- canonical names, used in code and plot legends ------------------------
EXACT_LABEL = "exact-sample dKS"
SSS_LABEL = "Sample-Sketch-Solve dKS"
SSS_TAG = "SSS-dKS"
SSS_DIRECT_CLEAN_LABEL = "SSS-dKS direct (clean, PENDING Peter's C2)"
SSS_DIRECT_UNION_LABEL = "SSS-dKS direct (union/Sec-6.2, PENDING Peter's C2)"

# stable method ids (CSV / plot keys) and their display labels
METHOD_EXACT = "exact_sample"
METHOD_SSS = "sss"
METHOD_DIRECT_CLEAN = "sss_direct_clean"
METHOD_DIRECT_UNION = "sss_direct_union"
LABELS = {
    METHOD_EXACT: EXACT_LABEL,
    METHOD_SSS: SSS_LABEL,
    METHOD_DIRECT_CLEAN: SSS_DIRECT_CLEAN_LABEL,
    METHOD_DIRECT_UNION: SSS_DIRECT_UNION_LABEL,
}
LEGEND = {  # short legend tags
    METHOD_EXACT: EXACT_LABEL,
    METHOD_SSS: SSS_TAG,
    METHOD_DIRECT_CLEAN: "SSS-dKS direct (clean)",
    METHOD_DIRECT_UNION: "SSS-dKS direct (union)",
}


# --- the two statistics -----------------------------------------------------
def exact_stat(P, Q):
    """exact-sample dKS: DIRECT brute-force O(n^2) dKS on the given samples."""
    return dks.exact(P, Q)


def sss_stat(P, Q, eps=-1.0):
    """Sample-Sketch-Solve dKS (SSS-dKS): deterministic O(n log n) grid approx.

    eps <= 0 uses the default 2*sqrt(n) grid.
    """
    return dks.approx(P, Q, eps)


def dks_stat(P, Q, exact=True, eps=-1.0):
    """Convenience: exact=True -> O(n^2) exact; else O(n log n) approx."""
    return dks.exact(P, Q) if exact else dks.approx(P, Q, eps)


# --- permutation test (the validity anchor) ---------------------------------
def permutation_test(P, Q, stat_fn, B, delta, rng, randomized=False):
    """Permutation two-sample test for an arbitrary dKS statistic `stat_fn`.

    T_obs = stat_fn(P, Q).  Pool the 2n points; for b in 1..B-1 shuffle the pool,
    split into the first n / last n, and recompute T_b = stat_fn(., .).  With
    T_obs that is B exchangeable values under H0.

        conservative p = (1 + #{b: T_b >= T_obs}) / B     # default
        reject         = p <= delta

    This is VALID / CONSERVATIVE at level delta for any B.  dKS is a discrete
    statistic, so ties (T_b == T_obs) occur and the conservative p-value can be
    strictly below delta -- report it as "valid / conservative level delta", not
    "exact".  randomized=True breaks ties with a coin flip at the (1 - delta)
    quantile to attain EXACTLY level delta (Hemerik & Goeman, 2018):

        p_rand = (#{T_b > T_obs} + U * (1 + #{T_b == T_obs})) / B,  U ~ U(0,1)

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


# --- two candidate direct (B-free) thresholds -------------------------------
# Both inversions are PROVISIONAL (PENDING Peter's C2). The permutation test is
# the validity anchor; run_calibration.py is the empirical check on which of
# these the null tail actually respects.
def tau_clean(n, delta):
    """Clean threshold: invert  delta = exp(-n eps^2 / 4)  ->  2*sqrt(ln(1/delta)/n).

    No union / VC factor.  The optimistic candidate.
    """
    return 2.0 * np.sqrt(np.log(1.0 / delta) / int(n))


def tau_union(n, delta):
    """Union threshold (paper's d=2 formula):  sqrt( (4 ln(2n) / n) * ln(1/delta) ).

    The ln(1/delta) term comes from the Chernoff bound, so it is a NATURAL log
    (Jeff confirmed in the draft) -- not log2.  Carries the extra ln(2n) factor
    from a union bound over the grid corners; larger than tau_clean, so more
    conservative (the safe candidate).
    """
    n = int(n)
    return float(np.sqrt((4.0 * np.log(2.0 * n) / n) * np.log(1.0 / delta)))


def sss_direct_test(P, Q, delta, tau_fn, eps=-1.0):
    """SSS-dKS direct (B-free): reject iff dks.approx(P, Q) > tau_fn(n, delta).

    tau_fn is tau_clean or tau_union.  NEITHER is claimed valid without proof
    (PENDING Peter's C2).  No permutations, so it costs a single approx eval --
    the cheap method.  Returns (reject, stat, tau, elapsed_seconds).
    """
    t0 = time.perf_counter()
    P = np.ascontiguousarray(P, dtype=float)
    Q = np.ascontiguousarray(Q, dtype=float)
    n = len(P)
    tau = tau_fn(n, delta)
    stat = dks.approx(P, Q, eps)
    reject = bool(stat > tau)
    elapsed = time.perf_counter() - t0
    return reject, float(stat), float(tau), elapsed


# --- the matching theoretical tail bounds (for the calibration plot) --------
def bound_clean(eps, n):
    """Tail bound inverted by tau_clean:  exp(-n eps^2 / 4)."""
    eps = np.asarray(eps, dtype=float)
    return np.exp(-n * eps ** 2 / 4.0)


def bound_union(eps, n):
    """Tail bound inverted by tau_union:  exp(-n eps^2 / (4 ln(2n)))."""
    eps = np.asarray(eps, dtype=float)
    return np.exp(-n * eps ** 2 / (4.0 * np.log(2.0 * n)))
