"""Single-dKS RUNTIME + OBSERVED-ERROR experiment (the fig:plots / fig 6 panels).

This is a pure algorithmic micro-benchmark of ONE dKS evaluation -- NOT a
permutation test (that lives in run_power.py, whose avg_runtime times a whole
B-permutation test).  Here we time a single call to each algorithm and read off
its returned value.

Setup: P = Q = Uniform([0, 1]^2) (the NULL), so the true dKS distance is 0 and
each algorithm's RETURNED value IS its observed error (finite-sample sup
discrepancy).  The per-(n, rep) RNG is seeded deterministically from
SeedSequence([seed, n_index, rep]), so the exact Baseline and the approx Our Algo
see the SAME P, Q at each (n, rep) even though we run all Our Algo cells first --
their error values stay paired.

  * dks.approx -> "Our Algo"   O(n log n) deterministic 2*sqrt(n) grid.  reps=20
                  at EVERY n in NS (cheap; run first).
  * dks.exact  -> "Baseline"   O(n^2) brute force.  reps=20 for n <= 16384, reps=1
                  for n >= 32768 (one quadratic eval is already minutes+).  Run in
                  increasing n; PROBE-STOP after any Baseline cell whose single-run
                  wall time exceeds ~40 min (hard ceiling n = 524288 regardless).

For each (method, n) we record mean runtime (seconds, ONE dKS call) and mean
observed error over its reps.  results/runtime.csv columns:

    method, n, reps, runtime, obs_err        (method in {exact, approx})

The CSV is opened once, header flushed up front, and each cell's row is written +
flushed as it finishes, so partial progress survives an interrupt.

Usage:
    python experiments/run_runtime.py            # full sweep (~1-2 hrs)
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
NS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 786432, 1048576]

METHOD_EXACT = "exact"     # Baseline (O(n^2))
METHOD_APPROX = "approx"   # Our Algo (O(n log n))

BASELINE_FULL_REPS_MAX = 16384   # n <= this: Baseline uses full reps; above: 1 rep
PROBE_STOP_SECONDS = 18000.0     # ~5 hr: stop pushing Baseline past this single-run cost


def _cell(fn, n, n_idx, reps, seed):
    """Mean (runtime_seconds_per_single_call, obs_err) over `reps` reps.

    The rng for each rep is SeedSequence([seed, n_idx, rep]) -- deterministic and
    independent of which method runs, so approx and exact share P, Q per (n, rep).
    """
    rt_sum = 0.0
    er_sum = 0.0
    for rep in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence([seed, n_idx, rep]))
        P = gen.uniform_square(n, rng)
        Q = gen.uniform_square(n, rng)
        t0 = time.perf_counter()
        v = fn(P, Q)
        rt_sum += time.perf_counter() - t0
        er_sum += v
    return rt_sum / reps, er_sum / reps


def run(ns, reps, seed, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    t_start = time.time()
    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "reps", "runtime", "obs_err"])
        w.writeheader()
        f.flush()

        def emit(method, n, r, runtime, obs_err):
            w.writerow({"method": method, "n": n, "reps": r,
                        "runtime": runtime, "obs_err": obs_err})
            f.flush()
            print(f"{method:<6} n={n:<7d} reps={r:<3d} "
                  f"runtime={runtime:.4f}s err={obs_err:.5f}  "
                  f"[{time.time()-t_start:.0f}s]")

        # --- Our Algo: all n first (cheap) -------------------------------------
        for n_idx, n in enumerate(ns):
            runtime, obs_err = _cell(lambda P, Q: dks.approx(P, Q), n, n_idx, reps, seed)
            emit(METHOD_APPROX, n, reps, runtime, obs_err)

        # --- Baseline: increasing n, with probe-stop ---------------------------
        for n_idx, n in enumerate(ns):
            reps_b = reps if n <= BASELINE_FULL_REPS_MAX else 1
            runtime, obs_err = _cell(lambda P, Q: dks.exact(P, Q), n, n_idx, reps_b, seed)
            emit(METHOD_EXACT, n, reps_b, runtime, obs_err)
            if runtime > PROBE_STOP_SECONDS:
                print(f"baseline hit ~40min at n={n}, stopping")
                break

    print(f"\nsaved {outfile}")
    print("RUNTIME RUN DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=20,
                    help="reps for Our Algo (all n) and Baseline (n <= 16384)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "runtime.csv"))
    args = ap.parse_args()

    print(f"runtime: NS={NS} reps={args.reps} seed={args.seed} "
          f"(Baseline reps=1 for n>={2*BASELINE_FULL_REPS_MAX})")
    run(NS, args.reps, args.seed, args.out)


if __name__ == "__main__":
    main()
