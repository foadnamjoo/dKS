"""Null-calibration experiment for dKS (Jeff's plot).

Under H0 (P and Q identically distributed), draw many independent uniform
samples, compute the dKS statistic each time, and compare the empirical tail
probability  P(D_n >= eps)  against the theoretical bound  exp(-n * eps^2 / 4).

If the empirical curve sits at or below the theoretical bound, the calibrated
test controls its size -- directly answering the reviewer concern that the
C_d constants might make the test miscalibrated / overly conservative. The gap
between the two curves visualizes exactly how conservative it is.

Run:  python experiments/run_calibration.py
      python experiments/run_calibration.py --n 200 --reps 2000
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generators as gen
import methods as M


def run(n, reps, seed, exact, outfile):
    rng = np.random.default_rng(seed)

    stats = np.empty(reps)
    for i in range(reps):
        P = gen.uniform(n, rng)        # P and Q identically distributed: H0
        Q = gen.uniform(n, rng)
        stats[i] = M.dks_stat(P, Q, exact=exact)
    stats.sort()

    # empirical exceedance P(D_n >= eps) over a grid of eps
    eps_grid = np.linspace(stats.min() * 0.9, stats.max() * 1.05, 300)
    emp = np.array([(stats >= e).mean() for e in eps_grid])
    bound = np.exp(-n * eps_grid**2 / 4.0)          # theoretical tail bound

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(eps_grid, emp, color="#1f77b4", lw=2,
            label=f"empirical  $P(D_n \\geq \\varepsilon)$  ({reps} reps)")
    ax.plot(eps_grid, np.clip(bound, 0, 1), color="#d62728", ls="--", lw=2,
            label=r"theoretical bound  $e^{-n\varepsilon^2/4}$")
    ax.set_xlabel(r"threshold $\varepsilon$")
    ax.set_ylabel("tail probability")
    ax.set_title(f"dKS null calibration  ($d=2$, $n={n}$, H$_0$: $P=Q$ uniform)")
    ax.set_yscale("log")
    ax.set_ylim(1.0 / reps / 2, 1.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print(f"n={n} reps={reps}  max empirical stat={stats.max():.4f}  "
          f"median={np.median(stats):.4f}")
    print(f"saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="calibration.pdf")
    ap.add_argument("--approx", action="store_true")
    args = ap.parse_args()
    run(args.n, args.reps, args.seed, exact=not args.approx, outfile=args.out)


if __name__ == "__main__":
    main()
