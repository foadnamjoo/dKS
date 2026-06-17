"""Single-dKS RUNTIME + OBSERVED-ERROR experiment (the fig:plots / fig 6 panels).

This is a pure algorithmic micro-benchmark of ONE dKS evaluation -- NOT a
permutation test (that lives in run_power.py, whose avg_runtime times a whole
B-permutation test).  Here we time a single call to each algorithm and read off
its returned value.

Setup: P = Q = Uniform([0, 1]^2) (the NULL), so the true dKS distance is 0 and
each algorithm's RETURNED value IS its observed error (finite-sample sup
discrepancy).  Both methods see the SAME P, Q per rep, so their error values are
paired.

  * dks.exact  -> "Baseline"   O(n^2) brute force.  Clipped at --baseline-max
                  (exact is quadratic, so large n is infeasible).
  * dks.approx -> "Our Algo"   O(n log n) deterministic 2*sqrt(n) grid.  Run on
                  the full NS sweep.

For each (method, n) we average runtime (seconds) and observed error over --reps
reps.  results/runtime.csv columns:

    method, n, reps, runtime, obs_err        (method in {exact, approx})

Usage:
    python experiments/run_runtime.py --baseline-max 16384 --reps 20 --seed 0
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generators as gen
import dks

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

METHOD_EXACT = "exact"     # Baseline
METHOD_APPROX = "approx"   # Our Algo


def run(ns, reps, baseline_max, seed, outfile):
    seedseq = np.random.SeedSequence(seed)
    rows = []
    t_start = time.time()

    for n in ns:
        rt = {METHOD_APPROX: 0.0, METHOD_EXACT: 0.0}
        er = {METHOD_APPROX: 0.0, METHOD_EXACT: 0.0}
        do_exact = n <= baseline_max
        for _ in range(reps):
            rng = np.random.default_rng(seedseq.spawn(1)[0])
            P = gen.uniform_square(n, rng)
            Q = gen.uniform_square(n, rng)

            # Our Algo: single O(n log n) approx eval (default 2*sqrt(n) grid)
            t0 = time.perf_counter()
            v = dks.approx(P, Q)
            rt[METHOD_APPROX] += time.perf_counter() - t0
            er[METHOD_APPROX] += v

            # Baseline: single O(n^2) exact eval on the SAME P, Q (paired error)
            if do_exact:
                t0 = time.perf_counter()
                v = dks.exact(P, Q)
                rt[METHOD_EXACT] += time.perf_counter() - t0
                er[METHOD_EXACT] += v

        methods = [METHOD_APPROX] + ([METHOD_EXACT] if do_exact else [])
        for m in methods:
            rows.append({
                "method": m, "n": n, "reps": reps,
                "runtime": rt[m] / reps, "obs_err": er[m] / reps,
            })
        ex = (f"  exact={rt[METHOD_EXACT]/reps*1e3:.1f}ms "
              f"err={er[METHOD_EXACT]/reps:.4f}") if do_exact else "  exact=skipped"
        print(f"n={n:<7d} approx={rt[METHOD_APPROX]/reps*1e3:.2f}ms "
              f"err={er[METHOD_APPROX]/reps:.4f}{ex}  [{time.time()-t_start:.0f}s]")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {outfile}  ({len(rows)} rows)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=20, help="reps per (method, n)")
    ap.add_argument("--baseline-max", type=int, default=16384,
                    help="largest n at which to run the O(n^2) exact Baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "runtime.csv"))
    args = ap.parse_args()

    print(f"runtime: NS={NS} reps={args.reps} baseline_max={args.baseline_max} "
          f"seed={args.seed}")
    run(NS, args.reps, args.baseline_max, args.seed, args.out)


if __name__ == "__main__":
    main()
