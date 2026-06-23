"""Parallel dKS-Sketch (sss) power scan at a high trial count W, to stabilize the
noisy curves (esp. alpha=0.01) per Jeff's "increase W until stable, push as far as
possible". The Sketch is cheap and its trials are independent, so we run all
(alpha, n, trial) across cores. Writes scan_w<W>.csv (+ _raw).

  .venv/bin/python experiments/run_par_sketch.py 200

Note: power is exact under parallelism; the Sketch RUNTIME here is mildly
contention-inflated (negligible on the figure -- Sketch is flat near 0 on a log
axis to 14000 s). For the FINAL paper number we can take a clean single-core
runtime pass; for stability-checking this is fine.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, csv, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generators as gen
import methods as M

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
B, DELTA, SEED, EPS = 100, 0.05, 0, -1.0

# same n-grid per alpha as the current figure
SCHED = {
    0.1:  [1000, 2000, 5000, 10000, 20000],
    0.03: [1000, 2000, 5000, 10000, 20000, 50000, 100000, 150000, 200000,
           300000, 400000, 500000],
    0.01: [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 300000,
           400000, 500000, 600000, 700000],
}


def one_trial(task):
    n, a, z = task
    aint = int(round(a * 1000))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, aint, n, z]))
    P = gen.uniform_square(n, rng)
    Q = gen.huber_mixture(n, a, rng)
    r, _, _, e = M.permutation_test(P, Q, lambda A, Bp: M.sss_stat(A, Bp, EPS),
                                    B, DELTA, rng, False)
    return (n, a, z, int(r), float(e))


def main():
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out = os.path.join(RESULTS, f"scan_w{W}.csv")
    raw = out[:-4] + "_raw.csv"
    tasks = [(n, a, z) for a, ns in SCHED.items() for n in ns for z in range(W)]
    tasks.sort(key=lambda t: -t[0])          # longest n first
    print(f"sketch W={W}: {len(tasks)} trials on {workers} cores -> {out}", flush=True)

    cells = {}   # (a,n) -> list of (reject, runtime)
    fr = open(raw, "w", newline="")
    wr = csv.writer(fr)
    wr.writerow(["method", "n", "alpha", "B", "delta", "trial", "reject", "runtime"])
    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for (n, a, z, rej, elp) in pool.imap_unordered(one_trial, tasks):
            cells.setdefault((a, n), []).append((rej, elp))
            wr.writerow(["sss", n, a, B, DELTA, z, rej, elp]); fr.flush()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)}  [{(time.time()-t0)/60:.1f} min]", flush=True)
    fr.close()

    fa = open(out, "w", newline="")
    wa = csv.DictWriter(fa, fieldnames=["method", "n", "alpha", "B", "delta", "Z",
                                        "rejection_rate", "ci_low", "ci_high", "avg_runtime"])
    wa.writeheader()
    for (a, n) in sorted(cells, key=lambda k: (k[0], k[1])):
        rr = [x[0] for x in cells[(a, n)]]; rt = [x[1] for x in cells[(a, n)]]
        Z = len(rr); p = sum(rr) / Z; se = (p * (1 - p) / Z) ** 0.5
        wa.writerow({"method": "sss", "n": n, "alpha": a, "B": B, "delta": DELTA,
                     "Z": Z, "rejection_rate": p, "ci_low": max(0.0, p - se),
                     "ci_high": min(1.0, p + se), "avg_runtime": sum(rt) / Z})
    fa.close()
    print(f"DONE {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
