"""Parallel Baseline (exact O(n^2)) power/runtime runner.

dks.exact is single-threaded C++ -- so on a many-core laptop we run independent
(n, alpha, trial) permutation tests across worker processes, each pinned to one
core (BLAS threads forced to 1).  This turns a 10 h desk-work window (plugged in)
into ~7x throughput, enough for the full alpha=0.03 Baseline power curve out to
n=100k WITH real power (not just a Z=1 runtime stamp).

Each completed trial is flushed to the raw CSV immediately, so a crash/stop keeps
all finished trials (never lose work, never hide data).  Aggregate (power +
avg_runtime) is recomputed from ALL raw trials at the end.

  .venv/bin/python experiments/run_par_baseline.py --workers 7
"""
import os
# force single-threaded math in every worker BEFORE numpy/dks import (spawn re-runs this)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generators as gen
import methods as M

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# schedule: alpha -> {n: target trials}.  alpha=0.03 is the headline contamination
# (saturates ~150k for dKS-Sketch); alpha=0.1 saturates by 10k.
SCHEDULE = {
    0.03: {1000: 10, 2000: 10, 5000: 10, 10000: 10, 20000: 10, 50000: 10, 100000: 5},
    0.1:  {1000: 10, 2000: 10, 5000: 10, 10000: 10},
}

B = 100
DELTA = 0.05
SEED = 0
RANDOMIZED = False        # conservative/valid p-value (methods.py default)
PER_TEST_10K = 139.0      # measured s/test at n=10k, B=100 (single core, unloaded)


def one_trial(task):
    """Run ONE permutation test. task=(n, a, z). Seed matches run_scan.run_cell."""
    n, a, z = task
    aint = int(round(a * 1000))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, aint, n, z]))
    P = gen.uniform_square(n, rng)
    Q = gen.huber_mixture(n, a, rng)
    r, _, _, e = M.permutation_test(P, Q, M.exact_stat, B, DELTA, rng, RANDOMIZED)
    return (n, a, z, int(r), float(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7,
                    help="parallel processes (leave a few cores for your desk work)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "baseline_par.csv"))
    args = ap.parse_args()
    raw_path = args.out[:-4] + "_raw.csv"

    # build task list; longest (largest n) first so the slow 100k tests start immediately
    tasks = []
    for a, sched in SCHEDULE.items():
        for n, Z in sched.items():
            for z in range(Z):
                tasks.append((n, a, z))
    tasks.sort(key=lambda t: -t[0])
    est = sum(PER_TEST_10K * (n / 10000.0) ** 2 for n, _, _ in tasks)
    print(f"tasks={len(tasks)}  workers={args.workers}  "
          f"single-core work={est/3600:.1f} core-h  "
          f"ideal wall~{est/3600/args.workers:.1f} h (longer under full-load throttle)",
          flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    fr = open(raw_path, "w", newline="")
    wr = csv.DictWriter(fr, fieldnames=["method", "n", "alpha", "B", "delta",
                                        "trial", "reject", "runtime"])
    wr.writeheader(); fr.flush()

    results = []
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for (n, a, z, rej, elp) in pool.imap_unordered(one_trial, tasks):
            results.append((n, a, z, rej, elp))
            wr.writerow({"method": "exact_sample", "n": n, "alpha": a, "B": B,
                         "delta": DELTA, "trial": z, "reject": rej, "runtime": elp})
            fr.flush()
            done += 1
            print(f"  [{done}/{len(tasks)}] a={a:<5} n={n:<7} z={z} "
                  f"rej={rej} rt={elp:.1f}s  [{(time.time()-t0)/60:.1f}min]", flush=True)
    fr.close()

    # aggregate from ALL raw trials
    fields = ["method", "n", "alpha", "B", "delta", "Z",
              "rejection_rate", "ci_low", "ci_high", "avg_runtime"]
    cells = {}
    for n, a, z, rej, elp in results:
        cells.setdefault((a, n), []).append((rej, elp))
    fa = open(args.out, "w", newline="")
    wa = csv.DictWriter(fa, fieldnames=fields); wa.writeheader()
    for (a, n) in sorted(cells, key=lambda k: (k[0], k[1])):
        rr = [x[0] for x in cells[(a, n)]]
        rt = [x[1] for x in cells[(a, n)]]
        Z = len(rr)
        p = sum(rr) / Z
        se = (p * (1 - p) / Z) ** 0.5
        wa.writerow({"method": "exact_sample", "n": n, "alpha": a, "B": B,
                     "delta": DELTA, "Z": Z, "rejection_rate": p,
                     "ci_low": max(0.0, p - se), "ci_high": min(1.0, p + se),
                     "avg_runtime": sum(rt) / Z})
    fa.close()
    print(f"DONE {(time.time()-t0)/3600:.2f} h -> {args.out}  (+raw {raw_path})", flush=True)


if __name__ == "__main__":
    main()
