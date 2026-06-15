"""2D Huber two-sample POWER experiment.

P = Uniform([-1, 1]^2) (clean);  Q_alpha = Huber mixture with a truncated central
Gaussian bump at contamination level alpha.  alpha = 0 is the null -- those rows
ARE the empirical size / Type-I check (should be ~ delta within the CI for the
permutation tests); alpha > 0 is power.

For each (n, alpha) we run Z trials, each with a fresh per-trial seed, and four
procedures:

  * exact-sample dKS  -- permutation_test(stat_fn = exact_stat)        [O(n^2) baseline]
  * SSS-dKS           -- permutation_test(stat_fn = sss_stat), same B  [the main one]
  * SSS-dKS direct (clean)  -- B-free, threshold tau_clean            [PENDING C2]
  * SSS-dKS direct (union)  -- B-free, threshold tau_union            [PENDING C2]

We compare ONLY exact-sample dKS vs Sample-Sketch-Solve dKS (+ the two direct
variants).  No MMD / energy / Bickel: this is an algorithmic comparison (same
dKS statistic, exact vs sketched vs analytic-threshold), not a divergence
bake-off.

Per (method, n, alpha) we record rejection_rate, a 95% Wilson binomial CI, and
avg_runtime; results/power.csv columns:
    method, n, alpha, B, delta, Z, rejection_rate, ci_low, ci_high, avg_runtime

Quick verification (a few minutes):
    python experiments/run_power.py
Scale up for the paper:
    python experiments/run_power.py --n 50 100 200 400 800 --B 300 --Z 300
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
METHOD_ORDER = [M.METHOD_EXACT, M.METHOD_SSS,
                M.METHOD_DIRECT_CLEAN, M.METHOD_DIRECT_UNION]


def wilson_ci(k, Z, z=1.96):
    """95% Wilson score interval for a binomial proportion k / Z."""
    if Z == 0:
        return 0.0, 0.0
    phat = k / Z
    denom = 1.0 + z * z / Z
    center = (phat + z * z / (2.0 * Z)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / Z + z * z / (4.0 * Z * Z))
    return max(0.0, center - half), min(1.0, center + half)


def run(ns, alphas, B, Z, delta, eps, seed, outfile, randomized):
    seedseq = np.random.SeedSequence(seed)
    rows = []
    t_start = time.time()

    for n in ns:
        for a in alphas:
            rej = {m: np.zeros(Z, dtype=bool) for m in METHOD_ORDER}
            elp = {m: np.zeros(Z) for m in METHOD_ORDER}
            for z in range(Z):
                rng = np.random.default_rng(seedseq.spawn(1)[0])
                P = gen.uniform_square(n, rng)
                Q = gen.huber_mixture(n, a, rng)

                r, _, _, e = M.permutation_test(
                    P, Q, M.exact_stat, B, delta, rng, randomized)
                rej[M.METHOD_EXACT][z], elp[M.METHOD_EXACT][z] = r, e

                r, _, _, e = M.permutation_test(
                    P, Q, lambda A, Bp: M.sss_stat(A, Bp, eps),
                    B, delta, rng, randomized)
                rej[M.METHOD_SSS][z], elp[M.METHOD_SSS][z] = r, e

                r, _, _, e = M.sss_direct_test(P, Q, delta, M.tau_clean, eps)
                rej[M.METHOD_DIRECT_CLEAN][z], elp[M.METHOD_DIRECT_CLEAN][z] = r, e

                r, _, _, e = M.sss_direct_test(P, Q, delta, M.tau_union, eps)
                rej[M.METHOD_DIRECT_UNION][z], elp[M.METHOD_DIRECT_UNION][z] = r, e

            for m in METHOD_ORDER:
                k = int(rej[m].sum())
                lo, hi = wilson_ci(k, Z)
                rows.append({
                    "method": m, "n": n, "alpha": a, "B": B, "delta": delta,
                    "Z": Z, "rejection_rate": k / Z,
                    "ci_low": lo, "ci_high": hi,
                    "avg_runtime": float(elp[m].mean()),
                })
            print(f"n={n:<4d} a={a:<4.2f} | "
                  f"exact={rej[M.METHOD_EXACT].mean():.2f} "
                  f"SSS={rej[M.METHOD_SSS].mean():.2f} "
                  f"dir_clean={rej[M.METHOD_DIRECT_CLEAN].mean():.2f} "
                  f"dir_union={rej[M.METHOD_DIRECT_UNION].mean():.2f} | "
                  f"t: exact={elp[M.METHOD_EXACT].mean()*1e3:6.1f}ms "
                  f"SSS={elp[M.METHOD_SSS].mean()*1e3:5.1f}ms "
                  f"[{time.time()-t_start:.0f}s]")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _print_summary(rows, alphas, delta)
    print(f"\nsaved {outfile}  ({len(rows)} rows)")
    return rows


def _print_summary(rows, alphas, delta):
    def cell(r):
        return (f"{r['rejection_rate']:.3f} "
                f"[{r['ci_low']:.3f},{r['ci_high']:.3f}] "
                f"{r['avg_runtime']*1e3:.2f}ms")

    print(f"\n=== EMPIRICAL SIZE  (alpha = 0; target ~ delta = {delta}, "
          f"within 95% Wilson CI) ===")
    hdr = f"{'method':<40}{'n':>6}{'reject [95% CI]  runtime':>30}"
    print(hdr); print("-" * len(hdr))
    for m in METHOD_ORDER:
        for r in [r for r in rows if r["method"] == m and r["alpha"] == 0.0]:
            ok = "OK" if (r["ci_low"] <= delta <= r["ci_high"]) else "**"
            print(f"{M.LABELS[m]:<40}{r['n']:>6}  {cell(r):>26} {ok}")

    print("\n=== POWER  (alpha > 0:  rejection_rate [95% CI]  avg_runtime) ===")
    for a in [x for x in alphas if x > 0.0]:
        print(f"\n-- alpha = {a} --")
        hdr = f"{'method':<40}{'n':>6}{'reject [95% CI]  runtime':>30}"
        print(hdr)
        for m in METHOD_ORDER:
            for r in [r for r in rows if r["method"] == m and r["alpha"] == a]:
                print(f"{M.LABELS[m]:<40}{r['n']:>6}  {cell(r):>26}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.15, 0.30])
    ap.add_argument("--B", type=int, default=100, help="permutations per test")
    ap.add_argument("--Z", type=int, default=100, help="trials per (n, alpha) cell")
    ap.add_argument("--delta", type=float, default=0.05, help="nominal level")
    ap.add_argument("--eps", type=float, default=-1.0,
                    help="SSS grid resolution; <=0 uses the 2*sqrt(n) grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomized", action="store_true",
                    help="randomized permutation tie-breaking (exact level delta)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "power.csv"))
    args = ap.parse_args()

    print(f"power: n={args.n} alpha={args.alpha} B={args.B} Z={args.Z} "
          f"delta={args.delta} eps={args.eps} seed={args.seed}")
    run(args.n, args.alpha, args.B, args.Z, args.delta, args.eps,
        args.seed, args.out, args.randomized)


if __name__ == "__main__":
    main()
