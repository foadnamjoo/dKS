"""Plots for the 2D Huber power + null-calibration experiments.

Reads results/power.csv and results/calibration.csv, writes PDFs (and PNGs) to
experiments/figures/:

    fig_runtime_vs_n.pdf       x=n,            y=avg_runtime (log-y)   exact vs SSS
    fig_power_vs_n.pdf         x=n,            y=rejection_rate (+CI)
    fig_power_vs_runtime.pdf   x=avg_runtime,  y=rejection_rate (+CI)  HEADLINE (log-x)
    fig_calibration.pdf        x=eps,          y=tail prob (log-y)

Styling (shared):
    Baseline (exact)       dashed + triangles
    Our Algo (approx)      solid  + circles
    Our Algo direct clean  dotted, faint
    Our Algo direct union  dash-dot, faint
One color per alpha; 95% Wilson CI shaded on rejection-rate curves.
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import methods as M

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")

STYLE = {
    M.METHOD_EXACT:        dict(linestyle="--", marker="^", lw=1.8, markersize=6),
    M.METHOD_SSS:          dict(linestyle="-",  marker="o", lw=1.8, markersize=6),
    M.METHOD_DIRECT_CLEAN: dict(linestyle=":",  marker="", lw=1.5, alpha=0.5),
    M.METHOD_DIRECT_UNION: dict(linestyle="-.", marker="", lw=1.5, alpha=0.5),
}
PERM = [M.METHOD_EXACT, M.METHOD_SSS]
DIRECT = [M.METHOD_DIRECT_CLEAN, M.METHOD_DIRECT_UNION]


def _read_power():
    rows = []
    with open(os.path.join(RESULTS, "power.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({
                "method": r["method"], "n": int(r["n"]), "alpha": float(r["alpha"]),
                "rejection_rate": float(r["rejection_rate"]),
                "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
                "avg_runtime": float(r["avg_runtime"]), "delta": float(r["delta"]),
            })
    return rows


def _alpha_colors(alphas):
    cmap = plt.cm.viridis
    return {a: cmap(0.12 + 0.76 * i / max(1, len(alphas) - 1))
            for i, a in enumerate(alphas)}


def _ser(rows, m, a):
    pts = sorted([r for r in rows if r["method"] == m and r["alpha"] == a],
                 key=lambda r: r["n"])
    return {k: np.array([p[k] for p in pts]) for k in
            ("n", "avg_runtime", "rejection_rate", "ci_low", "ci_high")}


def _legend(ax, alphas, colors, methods):
    mh = [Line2D([0], [0], color="0.25", label=M.LEGEND[m],
                 **{k: v for k, v in STYLE[m].items() if k != "alpha"})
          for m in methods]
    ah = [Line2D([0], [0], color=colors[a], lw=3, label=rf"$\alpha={a:g}$")
          for a in alphas]
    leg1 = ax.legend(handles=mh, title="method", fontsize=7.5, title_fontsize=8,
                     loc="upper left", framealpha=0.92)
    ax.add_artist(leg1)
    ax.legend(handles=ah, title="contamination", fontsize=8, title_fontsize=8,
              loc="lower right", framealpha=0.92)


def fig_runtime_vs_n(rows):
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors(alphas)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in PERM:
        for a in alphas:
            s = _ser(rows, m, a)
            ax.plot(s["n"], s["avg_runtime"] * 1e3, color=colors[a], **STYLE[m])
    ax.set_xlabel("sample size  $n$")
    ax.set_ylabel("avg runtime per test  (ms)")
    ax.set_yscale("log")
    ax.set_title("Runtime vs $n$: Baseline ($O(n^2)$) vs Our Algo ($O(n\\log n)$)")
    ax.grid(alpha=0.3, which="both")
    _legend(ax, alphas, colors, PERM)
    _save(fig, "fig_runtime_vs_n")


def fig_power_vs_n(rows):
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors(alphas)
    delta = rows[0]["delta"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in DIRECT:                       # faint, behind
        for a in alphas:
            s = _ser(rows, m, a)
            ax.plot(s["n"], s["rejection_rate"], color=colors[a], **STYLE[m])
    for m in PERM:
        for a in alphas:
            s = _ser(rows, m, a)
            ax.fill_between(s["n"], s["ci_low"], s["ci_high"],
                            color=colors[a], alpha=0.13, lw=0)
            ax.plot(s["n"], s["rejection_rate"], color=colors[a], **STYLE[m])
    ax.axhline(delta, color="0.55", ls=(0, (1, 1)), lw=1)
    ax.text(ax.get_xlim()[1], delta, f" $\\delta={delta:g}$", va="center",
            ha="left", fontsize=7, color="0.4")
    ax.set_xlabel("sample size  $n$")
    ax.set_ylabel("rejection rate  (95% Wilson CI)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Power vs $n$  ($\\alpha=0$ row is the empirical size / Type-I)")
    ax.grid(alpha=0.3)
    _legend(ax, alphas, colors, PERM + DIRECT)
    _save(fig, "fig_power_vs_n")


def fig_power_vs_runtime(rows):
    """HEADLINE: rejection probability (Y) vs runtime (X, log), per Jeff."""
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors(alphas)
    delta = rows[0]["delta"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in PERM:
        for a in alphas:
            s = _ser(rows, m, a)
            x = s["avg_runtime"] * 1e3
            ax.fill_between(x, s["ci_low"], s["ci_high"],
                            color=colors[a], alpha=0.13, lw=0)
            ax.plot(x, s["rejection_rate"], color=colors[a], **STYLE[m])
    ax.axhline(delta, color="0.55", ls=(0, (1, 1)), lw=1)
    ax.set_xlabel("avg runtime per test  (ms, log scale)")
    ax.set_ylabel("rejection probability  (95% Wilson CI)")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Power vs runtime  (left = cheaper, up = more powerful)")
    ax.grid(alpha=0.3, which="both")
    _legend(ax, alphas, colors, PERM)
    _save(fig, "fig_power_vs_runtime")


def fig_power_vs_alpha(rows):
    """Rejection rate vs contamination alpha at a FIXED n (the single n present;
    if several, the largest, noted in the title).  One line per method in
    PERM + DIRECT, with 95% Wilson CI bands on the permutation methods (matching
    fig_power_vs_n) and the M.LABELS legend.
    """
    ns = sorted({r["n"] for r in rows})
    n_fixed = ns[-1]
    sub = [r for r in rows if r["n"] == n_fixed]
    delta = rows[0]["delta"]
    methods = PERM + DIRECT
    mcol = dict(zip(methods, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]))

    def ser(m):
        pts = sorted([r for r in sub if r["method"] == m], key=lambda r: r["alpha"])
        return {k: np.array([p[k] for p in pts]) for k in
                ("alpha", "rejection_rate", "ci_low", "ci_high")}

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in DIRECT:                       # faint, behind (no CI band)
        s = ser(m)
        if len(s["alpha"]):
            ax.plot(s["alpha"], s["rejection_rate"], color=mcol[m], **STYLE[m])
    for m in PERM:                         # CI band + line
        s = ser(m)
        if len(s["alpha"]):
            ax.fill_between(s["alpha"], s["ci_low"], s["ci_high"],
                            color=mcol[m], alpha=0.13, lw=0)
            ax.plot(s["alpha"], s["rejection_rate"], color=mcol[m], **STYLE[m])
    ax.axhline(delta, color="0.55", ls=(0, (1, 1)), lw=1)
    ax.text(ax.get_xlim()[1], delta, f" $\\delta={delta:g}$", va="center",
            ha="left", fontsize=7, color="0.4")
    ax.set_xlabel(r"contamination  $\alpha$")
    ax.set_ylabel("rejection rate  (95% Wilson CI)")
    ax.set_ylim(-0.03, 1.03)
    note = f" (largest of {ns})" if len(ns) > 1 else ""
    ax.set_title(rf"Power vs contamination $\alpha$  (fixed $n={n_fixed}${note})")
    ax.grid(alpha=0.3)
    present = [m for m in methods if len(ser(m)["alpha"])]   # only methods with data
    handles = [Line2D([0], [0], color=mcol[m], label=M.LABELS[m],
                      **{k: v for k, v in STYLE[m].items() if k != "alpha"})
               for m in present]
    ax.legend(handles=handles, title="method", fontsize=7.5, title_fontsize=8,
              loc="upper left", framealpha=0.92)
    _save(fig, "fig_power_vs_alpha")


def fig_calibration():
    rows = {"exact": [], "approx": []}
    n = None
    with open(os.path.join(RESULTS, "calibration.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["statistic"]].append(float(r["value"]))
            n = int(r["n"])
    ve = np.sort(np.asarray(rows["exact"]))
    va = np.sort(np.asarray(rows["approx"]))
    Z = len(ve)

    def surv(vals):
        return vals, 1.0 - np.arange(len(vals)) / len(vals)

    lo = min(ve.min(), va.min()) * 0.8
    hi = max(ve.max(), va.max()) * 1.25
    eps = np.linspace(lo, hi, 500)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    xe, ye = surv(ve)
    xa, ya = surv(va)
    ax.step(xe, ye, where="post", color="#1f77b4", lw=2.2,
            label=r"empirical Baseline (exact)  $\hat P(\geq\varepsilon)$")
    ax.step(xa, ya, where="post", color="#1f77b4", lw=1.8, ls="--",
            label=r"empirical Our Algo (approx)")
    ax.plot(eps, np.clip(M.bound_clean(eps, n), 0, 1), color="#d62728", lw=2,
            label=r"clean bound  $e^{-n\varepsilon^2/4}$")
    ax.plot(eps, np.clip(M.bound_union(eps, n), 0, 1), color="#ff7f0e", lw=2,
            ls="-.", label=r"union bound  $e^{-n\varepsilon^2/(4\ln 2n)}$")

    # programmatic verdict over the OPERATIVE tail -- the small-probability
    # region (P_hat <= 0.2) where a level-delta test actually sets its threshold.
    # (Near the body the exponential bound is not meant to hold and is ignored.)
    emp_e = np.array([(ve >= e).mean() for e in eps])
    tail = emp_e <= 0.2
    below_clean = bool(np.all(emp_e[tail] <= M.bound_clean(eps[tail], n) + 1e-12))
    below_union = bool(np.all(emp_e[tail] <= M.bound_union(eps[tail], n) + 1e-12))
    if below_clean:
        verdict = ("exact tail (P<=0.2) sits below the CLEAN bound\n"
                   r"$\Rightarrow$ tau_clean is supported empirically")
    elif below_union:
        verdict = ("exact tail sits below UNION but crosses clean\n"
                   r"$\Rightarrow$ the Sec-6.2 ln(2n) factor is needed")
    else:
        verdict = "exact tail exceeds both bounds in this regime"
    bias = float(np.mean(ve) - np.mean(va))

    ax.set_xlabel(r"threshold  $\varepsilon$  (dKS value)")
    ax.set_ylabel("tail probability")
    ax.set_yscale("log")
    ax.set_ylim(0.5 / Z, 1.5)
    ax.set_title(rf"Null calibration  ($d=2$, $n={n}$, $Z={Z}$, H$_0$: $P=Q$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    ax.annotate(verdict + f"\napprox downward bias $\\approx${bias:+.4f}",
                xy=(0.03, 0.04), xycoords="axes fraction", fontsize=8,
                color="0.2", va="bottom")
    _save(fig, "fig_calibration")


def fig_calibration_cdf():
    """Empirical CDF P(D <= eps) under H0 for Baseline (exact) and Our Algo
    (approx), with the theoretical bound 1 - exp(-n eps^2 / (4 ln 2n)).
    Companion to fig_calibration (which shows the tail P(D >= eps)).
    """
    rows = {"exact": [], "approx": []}
    n = None
    with open(os.path.join(RESULTS, "calibration.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["statistic"]].append(float(r["value"]))
            n = int(r["n"])
    ve = np.sort(np.asarray(rows["exact"]))
    va = np.sort(np.asarray(rows["approx"]))
    Z = len(ve)

    def cdf(vals):
        return vals, np.arange(len(vals)) / len(vals)    # P(D <= eps)

    lo = min(ve.min(), va.min()) * 0.8
    hi = max(ve.max(), va.max()) * 1.25
    eps = np.linspace(lo, hi, 500)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    xe, ye = cdf(ve)
    xa, ya = cdf(va)
    ax.step(xe, ye, where="post", color="#1f77b4", lw=2.2,
            label=r"empirical Baseline (exact)  $\hat P(\leq\varepsilon)$")
    ax.step(xa, ya, where="post", color="#1f77b4", lw=1.8, ls="--",
            label=r"empirical Our Algo (approx)")
    ax.plot(eps, np.clip(1.0 - M.bound_union(eps, n), 0.0, 1.0),
            color="#ff7f0e", lw=2, ls="-.",
            label=r"bound  $1 - e^{-n\varepsilon^2/(4\ln 2n)}$")
    ax.set_xlabel(r"threshold  $\varepsilon$  (dKS value)")
    ax.set_ylabel(r"CDF  $P(D \leq \varepsilon)$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(rf"Null calibration (CDF)  ($d=2$, $n={n}$, $Z={Z}$, H$_0$: $P=Q$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig_calibration_cdf")


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--calibration", action="store_true")
    args = ap.parse_args()
    do_all = not (args.power or args.calibration)
    if do_all or args.power:
        rows = _read_power()
        fig_runtime_vs_n(rows)
        fig_power_vs_n(rows)
        fig_power_vs_runtime(rows)
        fig_power_vs_alpha(rows)
    if do_all or args.calibration:
        fig_calibration()
        fig_calibration_cdf()


if __name__ == "__main__":
    main()
