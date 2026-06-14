"""Power experiment (empirical rejection rate) for the dKS two-sample test.

For each contamination level alpha and sample size n, draw many independent
(P, Q) pairs -- P clean uniform, Q Huber-contaminated at alpha -- and record how
often each method rejects H0. alpha = 0 is the null (rejection rate should sit
near the nominal level = a size check); alpha > 0 is power.

Methods compared (one curve each, matching the mockup):
  * Our Method        -- direct analytic threshold (B-free)
  * B-Bootstrap Exact -- permutation test

Output: one panel per alpha, x = n, y = empirical rejection rate, saved as a PDF.

Run:  python experiments/run_power.py            # quick demo settings
      python experiments/run_power.py --full     # full settings from the plan
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generators as gen
import methods as M


def run(alphas, ns, trials, B, delta, seed, exact, outfile):
    rng = np.random.default_rng(seed)
    # results[method][alpha] -> list of rejection rates over ns
    methods = ["Our Method", "B-Bootstrap Exact"]
    rates = {m: {a: np.zeros(len(ns)) for a in alphas} for m in methods}

    t0 = time.time()
    for a in alphas:
        for j, n in enumerate(ns):
            rej_direct = 0
            rej_perm = 0
            for _ in range(trials):
                P = gen.uniform(n, rng)
                Q = gen.huber_contaminated(n, rng, alpha=a)
                rej_direct += M.direct_test(P, Q, delta=delta, exact=exact)["reject"]
                rej_perm += M.permutation_test(P, Q, B=B, rng=rng, level=delta,
                                               exact=exact)["reject"]
            rates["Our Method"][a][j] = rej_direct / trials
            rates["B-Bootstrap Exact"][a][j] = rej_perm / trials
            print(f"alpha={a:.2f} n={n:<4d} "
                  f"direct={rates['Our Method'][a][j]:.3f} "
                  f"boot={rates['B-Bootstrap Exact'][a][j]:.3f} "
                  f"[{time.time()-t0:.1f}s]")

    # ---- plot: one panel per alpha ----
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 3.6),
                             sharey=True)
    if len(alphas) == 1:
        axes = [axes]
    styles = {"Our Method": dict(marker="o", color="#1f77b4"),
              "B-Bootstrap Exact": dict(marker="s", color="#d62728", ls="--")}
    for ax, a in zip(axes, alphas):
        for m in methods:
            ax.plot(ns, rates[m][a], label=m, **styles[m])
        if a == 0.0:
            ax.axhline(delta, color="gray", ls=":", lw=1, label=f"level = {delta}")
            ax.set_title(r"$\alpha = 0$  (null / size)")
        else:
            ax.set_title(rf"$\alpha = {a}$")
        ax.set_xlabel("sample size $n$")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("empirical rejection rate")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"dKS two-sample test: empirical rejection "
                 f"(trials={trials}, B={B}, $\\delta$={delta})")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print(f"\nsaved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="full plan settings (slow); default is a quick demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="power.pdf")
    ap.add_argument("--exact", action="store_true",
                    help="use the exact O(n^2) statistic (slow; only for small n). "
                         "Default is the fast O(n log n) approximation.")
    args = ap.parse_args()

    if args.full:
        alphas = [0.0, 0.05, 0.10, 0.20]
        ns = [20, 50, 100, 200, 500]
        trials, B = 1000, 1000
    else:
        alphas = [0.0, 0.10, 0.20]
        ns = [20, 50, 100, 200]
        trials, B = 200, 200

    run(alphas, ns, trials=trials, B=B, delta=0.05, seed=args.seed,
        exact=args.exact, outfile=args.out)


if __name__ == "__main__":
    main()
