"""Top up the alpha=0.03 @ n=100,000 Baseline POWER cell from Z=5 to Z=10.
Adds trials z=5..9 (z=0..4 already in baseline_par_raw.csv, distinct seeds) and
appends them, so the loader then aggregates Z=10. Parallel: 5 tests on 5 cores,
~4 h wall (each 100k test ~3.9 h). Power is exact under parallelism."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, csv
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_par_baseline as rp

A, N, ZSTART, ZEND = 0.03, 100000, 5, 10

if __name__ == "__main__":
    tasks = [(N, A, z) for z in range(ZSTART, ZEND)]
    with Pool(len(tasks)) as pool:
        results = list(pool.imap_unordered(rp.one_trial, tasks))
    raw = os.path.join(rp.RESULTS, "baseline_par_raw.csv")
    with open(raw, "a", newline="") as f:
        w = csv.writer(f)
        for (n, a, z, rej, elp) in sorted(results, key=lambda r: r[2]):
            w.writerow(["exact_sample", n, a, rp.B, rp.DELTA, z, rej, elp])
    print(f"DONE 100k top-up: added trials {ZSTART}..{ZEND-1}; cell now Z=10")
