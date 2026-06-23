"""dKS-Sketch alpha-scan.

For each contamination alpha, increase n until the permutation-test POWER
saturates (then run ONE more n to confirm), to locate where each alpha reaches
full power.  dKS-Sketch (sss) ONLY -- it is fast, so this is cheap and tells us,
WITHOUT paying for the O(n^2) exact Baseline, which alpha needs a large n (and
how large) for the paper's runtime-gap experiment.

Writes results/scan_sss.csv incrementally (one flushed row per (n, alpha) cell),
so partial results survive interruption.  Same schema as power.csv.

  .venv/bin/python experiments/run_scan.py --alpha 0.005 0.01 0.03 0.05 0.07 0.1 0.2 0.3 \
      --B 500 --Z 100 --randomized
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generators as gen
import methods as M

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson_ci(k, Z, z=1.96):
    if Z == 0:
        return 0.0, 0.0
    phat = k / Z
    denom = 1.0 + z * z / Z
    center = (phat + z * z / (2.0 * Z)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / Z + z * z / (4 * Z * Z))
    return max(0.0, center - half), min(1.0, center + half)


def run_cell(n, a, B, Z, delta, eps, randomized, seed, method="sss"):
    """Z trials of the permutation test at (n, a); deterministic.
    method='sss' (dKS-Sketch) or 'exact_sample' (O(n^2) Baseline)."""
    rej = np.zeros(Z, dtype=bool)
    elp = np.zeros(Z)
    aint = int(round(a * 1000))
    if method == "exact_sample":
        stat = M.exact_stat
    else:
        stat = lambda A, Bp: M.sss_stat(A, Bp, eps)
    for z in range(Z):
        rng = np.random.default_rng(np.random.SeedSequence([seed, aint, n, z]))
        P = gen.uniform_square(n, rng)
        Q = gen.huber_mixture(n, a, rng)
        r, _, _, e = M.permutation_test(P, Q, stat, B, delta, rng, randomized)
        rej[z], elp[z] = r, e
    return rej, elp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, nargs="+", required=True)
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[1000, 2000, 5000, 10000, 20000, 50000,
                             100000, 200000, 500000])
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--Z", type=int, default=100)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomized", action="store_true")
    ap.add_argument("--sat", type=float, default=0.99,
                    help="rejection_rate at/above which an alpha is 'saturated'")
    ap.add_argument("--method", default="sss", choices=["sss", "exact_sample"],
                    help="'sss' (dKS-Sketch) or 'exact_sample' (O(n^2) Baseline)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "scan_sss.csv"))
    ap.add_argument("--raw", default=None,
                    help="per-trial raw output (default <out>_raw.csv): every trial's "
                         "reject(0/1)+runtime, so the plan/Z/band can change w/o re-running")
    args = ap.parse_args()
    if args.raw is None:
        args.raw = (args.out[:-4] + "_raw.csv" if args.out.endswith(".csv")
                    else args.out + "_raw")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["method", "n", "alpha", "B", "delta", "Z",
              "rejection_rate", "ci_low", "ci_high", "avg_runtime"]
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
        f.flush()
    raw_fields = ["method", "n", "alpha", "B", "delta", "trial", "reject", "runtime"]
    raw_new = not os.path.exists(args.raw)
    fraw = open(args.raw, "a", newline="")
    wraw = csv.DictWriter(fraw, fieldnames=raw_fields)
    if raw_new:
        wraw.writeheader()
        fraw.flush()

    print(f"scan[{args.method}]: alpha={args.alpha} ns<=({args.ns[-1]}) B={args.B} "
          f"Z={args.Z} randomized={args.randomized} -> {args.out}  (+raw {args.raw})",
          flush=True)
    t0 = time.time()
    for a in args.alpha:
        sat_idx = None
        for idx, n in enumerate(args.ns):
            tc = time.time()
            rej, elp = run_cell(n, a, args.B, args.Z, args.delta, args.eps,
                                args.randomized, args.seed, args.method)
            k = int(rej.sum())
            lo, hi = wilson_ci(k, args.Z)
            w.writerow({"method": args.method, "n": n, "alpha": a, "B": args.B,
                        "delta": args.delta, "Z": args.Z,
                        "rejection_rate": k / args.Z, "ci_low": lo, "ci_high": hi,
                        "avg_runtime": float(elp.mean())})
            f.flush()
            for z in range(args.Z):       # raw per-trial: re-aggregate any way later
                wraw.writerow({"method": args.method, "n": n, "alpha": a, "B": args.B,
                               "delta": args.delta, "trial": z,
                               "reject": int(rej[z]), "runtime": float(elp[z])})
            fraw.flush()
            print(f"  a={a:<5.3f} n={n:<7d} power={k/args.Z:.3f} "
                  f"rt/test={elp.mean():.3f}s cell={time.time()-tc:.0f}s "
                  f"[{(time.time()-t0)/60:.1f}min]", flush=True)
            if k / args.Z >= args.sat:
                if sat_idx is None:
                    sat_idx = idx          # first saturation -> run exactly one more n
                else:
                    break
            elif sat_idx is not None:
                break
        where = (f"saturates by n={args.ns[sat_idx]}" if sat_idx is not None
                 else f"NOT saturated by n={args.ns[-1]}")
        print(f"--- alpha={a}: {where} ---", flush=True)
    f.close()
    fraw.close()
    print(f"DONE in {(time.time()-t0)/60:.1f} min -> {args.out}  (+raw {args.raw})",
          flush=True)


if __name__ == "__main__":
    main()
