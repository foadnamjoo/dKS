"""Two-sample tests built on the dKS statistic.

Two procedures, mirroring the paper's comparison:

  * permutation_test  -- the "B-Bootstrap Exact Test". Self-contained and exactly
    valid for any B: it builds the null distribution by relabeling the pooled
    sample, so it needs no analytic constant. This is the rigorous reference.

  * direct_test       -- "Our Method": reject when the statistic exceeds an
    analytic threshold eps(n, delta), so NO bootstrap is needed (B-free).
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ THRESHOLD CONSTANT IS PROVISIONAL.  threshold_d2() below uses the log-n   │
    │ candidate C2 = 4*ln(2n).  Confirm the exact d=2 constant / closed form    │
    │ with Peter (the paper's Section 6.2 C_2) before using these numbers in    │
    │ the paper. Everything is parameterized so the final form drops straight   │
    │ in: pass C2=..., or replace threshold_d2 entirely.                        │
    └─────────────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import dks


def dks_stat(P, Q, exact=True, eps=-1.0):
    """dKS statistic. exact=True -> O(n^2) exact; else O(n log n) approx."""
    return dks.exact(P, Q) if exact else dks.approx(P, Q, eps)


def permutation_test(P, Q, B, rng, level=0.05, exact=True, eps=-1.0):
    """Permutation (bootstrap) two-sample test.

    p-value = (1 + #{permuted stat >= observed}) / (B + 1), which is exactly
    valid (controls type-I error at `level`) for any B. Reject if p <= level.
    """
    P = np.ascontiguousarray(P, dtype=float)
    Q = np.ascontiguousarray(Q, dtype=float)
    n_p, n_q = len(P), len(Q)
    pooled = np.vstack([P, Q])
    N = n_p + n_q

    observed = dks_stat(P, Q, exact, eps)
    ge = 0
    for _ in range(B):
        idx = rng.permutation(N)
        Pp = pooled[idx[:n_p]]
        Qp = pooled[idx[n_p:]]
        if dks_stat(Pp, Qp, exact, eps) >= observed:
            ge += 1
    pvalue = (1.0 + ge) / (B + 1.0)
    return {"stat": observed, "pvalue": pvalue, "reject": pvalue <= level}


def threshold_d2(n, delta, C2=None):
    """PROVISIONAL analytic rejection threshold eps(n, delta) for d = 2.

    Default uses the log-n constant C2 = 4*ln(2n). The functional form here
    (sqrt((C2 + ln(2/delta)) / (2n))) follows the DKW/Chernoff-style inversion
    of the sample-complexity bound and is a placeholder pending Peter's exact
    Section 6.2 expression. Override by passing C2 or replacing this function.
    """
    n = int(n)
    if C2 is None:
        C2 = 4.0 * np.log(2.0 * n)
    return float(np.sqrt((C2 + np.log(2.0 / delta)) / (2.0 * n)))


def direct_test(P, Q, delta, C2=None, exact=True, eps=-1.0):
    """Direct (B-free) test: reject if dKS(P, Q) >= eps(n, delta)."""
    P = np.ascontiguousarray(P, dtype=float)
    Q = np.ascontiguousarray(Q, dtype=float)
    n = min(len(P), len(Q))
    thr = threshold_d2(n, delta, C2)
    stat = dks_stat(P, Q, exact, eps)
    return {"stat": stat, "threshold": thr, "reject": stat >= thr}
