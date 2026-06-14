"""Correctness checks for the dks Python binding.

Run after `pip install .`:
    python bindings/python/test_dks_py.py
"""
import numpy as np
import dks


def brute(P, Q):
    """O(n^3) NumPy reference for dKS (small n only)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    npts, nq = len(P), len(Q)
    if npts == 0 or nq == 0:
        return 0.0
    xs = np.concatenate([P[:, 0], Q[:, 0]])
    ys = np.concatenate([P[:, 1], Q[:, 1]])
    best = 0.0
    for xt in xs:
        for yt in ys:
            cp = np.count_nonzero((P[:, 0] <= xt) & (P[:, 1] <= yt))
            cq = np.count_nonzero((Q[:, 0] <= xt) & (Q[:, 1] <= yt))
            best = max(best, abs(cp / npts - cq / nq))
    return best


def main():
    rng = np.random.default_rng(0)
    fails = 0

    # identity -> 0
    A = rng.random((20, 2))
    assert abs(dks.exact(A, A)) < 1e-12, "identity exact"
    assert abs(dks.approx(A, A)) < 1e-12, "identity approx"

    # shifted/disjoint -> 1
    P = np.array([[0.0, 0.0], [0.1, 0.1]])
    Q = np.array([[1.0, 1.0], [1.1, 1.1]])
    assert abs(dks.exact(P, Q) - 1.0) < 1e-12, "shifted exact"

    # exact == brute on random instances (incl. unequal sizes)
    for _ in range(100):
        P = rng.random((rng.integers(1, 25), 2))
        Q = rng.random((rng.integers(1, 25), 2))
        e, b = dks.exact(P, Q), brute(P, Q)
        if abs(e - b) > 1e-12:
            print(f"FAIL exact vs brute: {e} vs {b}")
            fails += 1
            break

    # approx within eps of exact
    eps = 0.05
    worst = 0.0
    for _ in range(20):
        P = rng.random((2000, 2))
        Q = rng.random((2000, 2))
        worst = max(worst, abs(dks.exact(P, Q) - dks.approx(P, Q, eps)))
    print(f"approx vs exact worst |diff| = {worst:.4f} (eps={eps})")
    if worst > eps + 1e-9:
        print("FAIL approx exceeded eps")
        fails += 1

    print("ALL PYTHON TESTS PASSED" if fails == 0 else f"{fails} FAILURE(S)")
    raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
