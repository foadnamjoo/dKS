"""Null-CALIBRATION experiment for 2D dKS.

Under H0 (P = Q = Uniform([-1, 1]^2)) draw Z independent fresh samples and record
the null dKS value with BOTH statistics:

  * exact-sample dKS (dks.exact)  -- the direct O(n^2) value; this is the one the
    theory bounds, so its empirical tail is what we compare to the bounds.
  * SSS-dKS         (dks.approx)  -- recorded too, so the approximation's
    downward bias (the grid under-counts the true max gap) is visible.

plots.py overlays the empirical tails P_hat(dKS >= eps) on the two candidate
bounds (clean = exp(-n eps^2/4), union = 2^(-n eps^2/(4 ln 2n))) -- the visual
answer to which direct threshold (tau_clean vs tau_union) the null actually
respects.

Verification default: n = 2000, Z = 500 (fast even with O(n^2) exact).
FINAL config:           n = 10000, Z = 1000  (exact at n=10k is a one-time
                        ~10-15 min run; pass --n 10000 --Z 1000).

results/calibration.csv columns:  statistic, value, n, Z
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generators as gen
import methods as M


HERE = os.path.dirname(os.path.abspath(__file__))


def run(n, Z, eps, seed, outfile):
    seedseq = np.random.SeedSequence(seed)
    v_exact = np.empty(Z)
    v_approx = np.empty(Z)
    t0 = time.time()
    for i in range(Z):
        rng = np.random.default_rng(seedseq.spawn(1)[0])
        P = gen.uniform_square(n, rng)       # H0: P and Q identically distributed
        Q = gen.uniform_square(n, rng)
        v_exact[i] = M.exact_stat(P, Q)
        v_approx[i] = M.sss_stat(P, Q, eps)
        if (i + 1) % max(1, Z // 10) == 0:
            print(f"  {i+1}/{Z}  [{time.time()-t0:.0f}s]")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "value", "n", "Z"])
        for v in v_exact:
            w.writerow(["exact", float(v), n, Z])
        for v in v_approx:
            w.writerow(["approx", float(v), n, Z])

    _diagnose("exact-sample dKS (dks.exact)", v_exact, n)
    _diagnose("SSS-dKS (dks.approx)", v_approx, n)
    bias = float(np.mean(v_exact - v_approx))
    print(f"\napprox downward bias (mean exact - mean approx) = {bias:+.4f}  "
          f"({100*bias/np.mean(v_exact):+.1f}% of mean exact)")
    print(f"saved {outfile}  ({2*Z} rows: {Z} exact + {Z} approx)")
    return v_exact, v_approx


def _diagnose(name, vals, n):
    print(f"\n{name}:  n={n} Z={len(vals)}  "
          f"min={vals.min():.4f} median={np.median(vals):.4f} "
          f"mean={vals.mean():.4f} max={vals.max():.4f}")
    print(f"  {'eps':>8}{'P_hat(>=eps)':>14}{'clean bound':>14}{'union bound':>14}")
    for e in np.quantile(vals, [0.5, 0.9, 0.95, 0.99]):
        emp = float((vals >= e).mean())
        bc = float(M.bound_clean(e, n))
        bu = float(M.bound_union(e, n))
        print(f"  {e:>8.4f}{emp:>14.4f}{bc:>14.4f}{bu:>14.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="sample size per draw")
    ap.add_argument("--Z", type=int, default=500, help="number of null redraws")
    ap.add_argument("--eps", type=float, default=-1.0,
                    help="SSS grid resolution; <=0 uses the 2*sqrt(n) grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "calibration.csv"))
    args = ap.parse_args()

    print(f"calibration: n={args.n} Z={args.Z} eps={args.eps} seed={args.seed}")
    run(args.n, args.Z, args.eps, args.seed, args.out)


if __name__ == "__main__":
    main()
