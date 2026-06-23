"""Run ONE Baseline power cell in parallel and append to baseline_par_raw.csv.
Used to add the missing alpha=0.1 @ n=20,000 point (Sketch has it; Baseline didn't).
Power is exact under parallelism; runtime is alpha-independent and taken from the
clean overnight n=20,000 measurement, so this run is power-only."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, csv
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_par_baseline as rp

A, N, Z = 0.1, 20000, 10

if __name__ == "__main__":
    tasks = [(N, A, z) for z in range(Z)]
    with Pool(min(Z, 10)) as pool:
        results = list(pool.imap_unordered(rp.one_trial, tasks))
    raw = os.path.join(rp.RESULTS, "baseline_par_raw.csv")
    with open(raw, "a", newline="") as f:
        w = csv.writer(f)
        for (n, a, z, rej, elp) in sorted(results, key=lambda r: r[2]):
            w.writerow(["exact_sample", n, a, rp.B, rp.DELTA, z, rej, elp])
    k = sum(r[3] for r in results)
    print(f"DONE a={A} n={N}: appended {Z} trials -> {k}/{Z} = {k/Z:.2f}")
