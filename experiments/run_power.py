"""Huber two-sample POWER experiment (item 3).

P = Uniform([-1, 1]^2) (clean);  Q_alpha = Huber mixture with a central Gaussian
bump at contamination level alpha.  alpha = 0 is the null (rejection rate ~ delta,
a size check); alpha > 0 is power.

For each (n, alpha) we run Z independent trials, each with a fresh per-trial seed,
and three procedures:

  * exact-sample dKS  -- permutation_test(stat_fn = exact_stat)        [reference]
  * SSS-dKS           -- permutation_test(stat_fn = sss_stat), same B  [the main one]
  * SSS-dKS direct    -- B-free threshold test (PENDING Peter's C2)    [the cheap one]

We compare ONLY exact-sample dKS vs Sample-Sketch-Solve dKS (+ the direct
variant) -- no MMD / energy / Bickel: the point is the algorithmic framing
(same statistic, exact vs sketched), not a kernel-vs-dKS bake-off.

Per (method, n, alpha) we record rejection_rate = mean(reject) and
avg_runtime = mean(elapsed), and write results/power.csv.

Quick verification run (small defaults, a few minutes):
    python experiments/run_power.py
Scale up for the paper:
    python experiments/run_power.py --n 50 100 200 400 800 --B 300 --Z 300
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
METHOD_ORDER = [M.METHOD_EXACT, M.METHOD_SSS, M.METHOD_SSS_DIRECT]


def run(ns, alphas, B, Z, delta, eps, seed, outfile, randomized):
    # one fresh child seed per trial -> reproducible given `seed`
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

                # exact-sample dKS permutation test
                r, _, _, e = M.permutation_test(
                    P, Q, M.exact_stat, B, delta, rng, randomized)
                rej[M.METHOD_EXACT][z], elp[M.METHOD_EXACT][z] = r, e

                # SSS-dKS permutation test (same B)
                r, _, _, e = M.permutation_test(
                    P, Q, lambda A, Bp: M.sss_stat(A, Bp, eps),
                    B, delta, rng, randomized)
                rej[M.METHOD_SSS][z], elp[M.METHOD_SSS][z] = r, e

                # SSS-dKS direct (B-free)
                r, _, _, e = M.sss_direct_test(P, Q, delta, eps)
                rej[M.METHOD_SSS_DIRECT][z], elp[M.METHOD_SSS_DIRECT][z] = r, e

            for m in METHOD_ORDER:
                rows.append({
                    "method": m,
                    "label": M.LABELS[m],
                    "n": n,
                    "alpha": a,
                    "rejection_rate": float(rej[m].mean()),
                    "avg_runtime": float(elp[m].mean()),
                    "Z": Z, "B": B, "delta": delta,
                })
            print(f"n={n:<4d} alpha={a:<4.2f} "
                  f"| exact={rej[M.METHOD_EXACT].mean():.3f}@{elp[M.METHOD_EXACT].mean()*1e3:7.2f}ms "
                  f"| SSS={rej[M.METHOD_SSS].mean():.3f}@{elp[M.METHOD_SSS].mean()*1e3:7.2f}ms "
                  f"| direct={rej[M.METHOD_SSS_DIRECT].mean():.3f}@{elp[M.METHOD_SSS_DIRECT].mean()*1e3:6.3f}ms "
                  f"[{time.time()-t_start:.0f}s]")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _print_summary(rows, delta)
    print(f"\nsaved {outfile}  ({len(rows)} rows)")
    return rows


def _print_summary(rows, delta):
    print("\n=== POWER SUMMARY (rejection_rate | avg_runtime ms) ===")
    header = f"{'method':<26}{'n':>6}{'alpha':>7}{'reject':>9}{'runtime_ms':>13}"
    print(header)
    print("-" * len(header))
    for m in METHOD_ORDER:
        for r in [r for r in rows if r["method"] == m]:
            flag = "  <- null size" if r["alpha"] == 0.0 else ""
            print(f"{M.LABELS[m]:<26}{r['n']:>6}{r['alpha']:>7.2f}"
                  f"{r['rejection_rate']:>9.3f}{r['avg_runtime']*1e3:>13.3f}{flag}")
    print(f"(null target ~ delta = {delta})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100, 200],
                    help="sample sizes (default: small verification grid)")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.15, 0.30],
                    help="Huber contamination levels (alpha=0 is the null)")
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
