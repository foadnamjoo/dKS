"""Plots for the Huber power + null-calibration experiments (item 4).

Reads results/power.csv and results/calibration.csv, writes PDFs (and PNGs) to
experiments/figures/:

    fig_runtime_vs_n.pdf       x = n,            y = avg_runtime   (log-y)
    fig_power_vs_n.pdf         x = n,            y = rejection_rate
    fig_power_vs_runtime.pdf   x = avg_runtime,  y = rejection_rate (log-x)   HEADLINE
    fig_calibration.pdf        x = eps,          y = tail prob (log-y)

Method styling is fixed and shared across the power figures:
    exact-sample dKS    dashed + triangle
    SSS-dKS             solid  + circle
    SSS-dKS direct      dotted + x
One curve per alpha (color), three per method.
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

# fixed per-method style (item 4)
STYLE = {
    M.METHOD_EXACT:      dict(linestyle="--", marker="^"),
    M.METHOD_SSS:        dict(linestyle="-",  marker="o"),
    M.METHOD_SSS_DIRECT: dict(linestyle=":",  marker="x"),
}
METHOD_ORDER = [M.METHOD_EXACT, M.METHOD_SSS, M.METHOD_SSS_DIRECT]


def _read_power():
    rows = []
    with open(os.path.join(RESULTS, "power.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({
                "method": r["method"], "n": int(r["n"]),
                "alpha": float(r["alpha"]),
                "rejection_rate": float(r["rejection_rate"]),
                "avg_runtime": float(r["avg_runtime"]),
                "delta": float(r["delta"]),
            })
    return rows


def _alpha_colors(alphas):
    cmap = plt.cm.viridis
    return {a: cmap(0.12 + 0.76 * i / max(1, len(alphas) - 1))
            for i, a in enumerate(alphas)}


def _series(rows, method, alpha, xkey, ykey):
    pts = sorted([r for r in rows if r["method"] == method and r["alpha"] == alpha],
                 key=lambda r: r["n"])
    return [p[xkey] for p in pts], [p[ykey] for p in pts]


def _method_alpha_legend(ax, alphas, colors, delta=None):
    method_handles = [Line2D([0], [0], color="0.25", **STYLE[m], label=M.LEGEND[m])
                      for m in METHOD_ORDER]
    alpha_handles = [Line2D([0], [0], color=colors[a], lw=3,
                            label=rf"$\alpha={a:g}$") for a in alphas]
    leg1 = ax.legend(handles=method_handles, title="method", fontsize=8,
                     title_fontsize=8, loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=alpha_handles, title="contamination", fontsize=8,
              title_fontsize=8, loc="lower right", framealpha=0.9)


def fig_runtime_vs_n(rows):
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors(alphas)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in METHOD_ORDER:
        for a in alphas:
            x, y = _series(rows, m, a, "n", "avg_runtime")
            ax.plot(x, np.array(y) * 1e3, color=colors[a], **STYLE[m],
                    markersize=6, lw=1.8)
    ax.set_xlabel("sample size  $n$")
    ax.set_ylabel("avg runtime per test  (ms)")
    ax.set_yscale("log")
    ax.set_title("Runtime vs sample size  (exact-sample dKS is $O(n^2)$ per call)")
    ax.grid(alpha=0.3, which="both")
    _method_alpha_legend(ax, alphas, colors)
    _save(fig, "fig_runtime_vs_n")


def fig_power_vs_n(rows):
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors(alphas)
    delta = rows[0]["delta"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in METHOD_ORDER:
        for a in alphas:
            x, y = _series(rows, m, a, "n", "rejection_rate")
            ax.plot(x, y, color=colors[a], **STYLE[m], markersize=6, lw=1.8)
    ax.axhline(delta, color="0.55", ls=(0, (1, 1)), lw=1)
    ax.text(ax.get_xlim()[1], delta, f" level $\\delta={delta:g}$",
            va="center", ha="left", fontsize=7, color="0.4")
    ax.set_xlabel("sample size  $n$")
    ax.set_ylabel("rejection rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Power vs sample size  ($\\alpha=0$ is the null / size check)")
    ax.grid(alpha=0.3)
    _method_alpha_legend(ax, alphas, colors)
    _save(fig, "fig_power_vs_n")


def fig_power_vs_runtime(rows):
    """HEADLINE: rejection probability (Y) vs runtime (X, log) -- per Jeff."""
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors(alphas)
    delta = rows[0]["delta"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in METHOD_ORDER:
        for a in alphas:
            # ordered by n: each curve traces increasing sample size
            x, y = _series(rows, m, a, "avg_runtime", "rejection_rate")
            ax.plot(np.array(x) * 1e3, y, color=colors[a], **STYLE[m],
                    markersize=6, lw=1.8)
    ax.axhline(delta, color="0.55", ls=(0, (1, 1)), lw=1)
    ax.set_xlabel("avg runtime per test  (ms, log scale)")
    ax.set_ylabel("rejection probability")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Power vs runtime  (left = cheaper; up = more powerful)")
    ax.grid(alpha=0.3, which="both")
    _method_alpha_legend(ax, alphas, colors)
    _save(fig, "fig_power_vs_runtime")


def fig_calibration():
    vals, n = [], None
    with open(os.path.join(RESULTS, "calibration.csv")) as f:
        for r in csv.DictReader(f):
            vals.append(float(r["dks_approx"]))
            n = int(r["n"])
    vals = np.sort(np.asarray(vals))
    Z = len(vals)

    eps_emp = vals
    tail_emp = 1.0 - np.arange(Z) / Z            # P_hat(dKS >= eps) as a step
    eps_dense = np.linspace(vals.min() * 0.8, vals.max() * 1.25, 400)
    bound = np.exp(-n * eps_dense ** 2 / 4.0)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.step(eps_emp, tail_emp, where="post", color="#1f77b4", lw=2,
            label=rf"empirical  $\hat P(\mathrm{{dKS}}\geq\varepsilon)$  ($Z={Z}$)")
    ax.plot(eps_dense, np.clip(bound, 0, 1), color="#d62728", ls="--", lw=2,
            label=r"bound  $e^{-n\varepsilon^2/4}$")
    ax.set_xlabel(r"threshold  $\varepsilon$  (dKS value)")
    ax.set_ylabel("tail probability")
    ax.set_yscale("log")
    ax.set_ylim(0.5 / Z, 1.5)
    ax.set_title(rf"Null calibration of SSS-dKS  ($d=2$, $n={n}$, H$_0$: $P=Q$ uniform)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.annotate("empirical tail sits below and close to the bound\n"
                r"$\Rightarrow$ clean threshold $\tau=2\sqrt{\ln(1/\delta)/n}$ is calibrated",
                xy=(0.03, 0.04), xycoords="axes fraction", fontsize=8,
                color="0.25", va="bottom")
    _save(fig, "fig_calibration")


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true", help="only power figures")
    ap.add_argument("--calibration", action="store_true", help="only calibration")
    args = ap.parse_args()
    do_all = not (args.power or args.calibration)

    if do_all or args.power:
        rows = _read_power()
        fig_runtime_vs_n(rows)
        fig_power_vs_n(rows)
        fig_power_vs_runtime(rows)
    if do_all or args.calibration:
        fig_calibration()


if __name__ == "__main__":
    main()
