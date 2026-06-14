"""Null-CALIBRATION experiment for dKS (item 5).

Under H0 (P and Q identically distributed), draw Z independent fresh uniform
redraws and record the Sample-Sketch-Solve dKS statistic (dks.approx) each time.
These Z null values let plots.py compare the empirical tail

    P_hat(dKS >= eps)

against the analytic bound exp(-n * eps^2 / 4).  If the empirical tail sits
below and close to the bound, the clean threshold tau = 2*sqrt(ln(1/delta)/n)
used by the SSS-dKS direct test is the right inversion (vs the Section 6.2 form
with an extra ln(2n)) -- i.e. this experiment settles which direct constant is
correct, and shows the test controls its size without being wildly conservative.

We use n large (default 10000) so the tail bound is in its meaningful regime.

Quick verification run:
    python experiments/run_calibration.py --n 10000 --Z 1000
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
    vals = np.empty(Z)
    t0 = time.time()
    for i in range(Z):
        rng = np.random.default_rng(seedseq.spawn(1)[0])
        P = gen.uniform_square(n, rng)      # H0: P and Q identically distributed
        Q = gen.uniform_square(n, rng)
        vals[i] = M.sss_stat(P, Q, eps)     # statistic = dks.approx (SSS-dKS)
        if (i + 1) % max(1, Z // 10) == 0:
            print(f"  {i+1}/{Z}  [{time.time()-t0:.0f}s]")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "n", "dks_approx"])
        for i, v in enumerate(vals):
            w.writerow([i, n, float(v)])

    # quick console diagnostic: empirical tail vs bound at a few thresholds
    print(f"\nn={n}  Z={Z}  statistic=SSS-dKS (dks.approx)")
    print(f"  null dKS: min={vals.min():.4f}  median={np.median(vals):.4f}  "
          f"mean={vals.mean():.4f}  max={vals.max():.4f}")
    print(f"  {'eps':>8}{'P_hat(>=eps)':>16}{'exp(-n eps^2/4)':>18}{'below?':>9}")
    qs = np.quantile(vals, [0.5, 0.9, 0.95, 0.99])
    for e in qs:
        emp = float((vals >= e).mean())
        bound = float(np.exp(-n * e * e / 4.0))
        print(f"  {e:>8.4f}{emp:>16.4f}{bound:>18.4f}{str(emp <= bound):>9}")
    print(f"saved {outfile}  ({Z} null values)")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000, help="sample size per draw")
    ap.add_argument("--Z", type=int, default=1000, help="number of null redraws")
    ap.add_argument("--eps", type=float, default=-1.0,
                    help="SSS grid resolution; <=0 uses the 2*sqrt(n) grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "calibration.csv"))
    args = ap.parse_args()

    print(f"calibration: n={args.n} Z={args.Z} eps={args.eps} seed={args.seed}")
    run(args.n, args.Z, args.eps, args.seed, args.out)


if __name__ == "__main__":
    main()
