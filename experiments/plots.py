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
from matplotlib.ticker import FuncFormatter
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


def _read_power(fname="power.csv"):
    rows = []
    with open(os.path.join(RESULTS, fname)) as f:
        for r in csv.DictReader(f):
            rows.append({
                "method": r["method"], "n": int(r["n"]), "alpha": float(r["alpha"]),
                "rejection_rate": float(r["rejection_rate"]),
                "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
                "avg_runtime": float(r["avg_runtime"]), "delta": float(r["delta"]),
                "Z": int(r["Z"]),
            })
    return rows


def _alpha_colors(alphas):
    cmap = plt.cm.viridis
    return {a: cmap(0.12 + 0.76 * i / max(1, len(alphas) - 1))
            for i, a in enumerate(alphas)}


# Curated palette for the CSR panels: vivid, lively, maximally distinguishable
# (Material-accent hues) -- bright blue / green / orange / magenta as
# contamination rises.  Kept separate from _alpha_colors so the other finalized
# panels (still on viridis) are untouched.  Falls back to viridis if there are
# more alphas than palette entries.
_CSR_PALETTE = ["#2979FF", "#00C853", "#FF6D00", "#F50057", "#AA00FF"]
# Colours for the CSR contamination curves specifically.  With alpha in
# {0.1,0.2,0.3} this gives blue / green / vivid-pink (#F50057 -- the hue we
# liked from the dropped alpha=0.4).  Kept separate from _CSR_PALETTE so the
# method colours used elsewhere (Baseline orange, dKS-Sketch blue) are intact.
_CSR_ALPHA_COLORS = ["#2979FF", "#00C853", "#F50057", "#FF6D00", "#AA00FF"]


def _alpha_colors_csr(alphas):
    if len(alphas) <= len(_CSR_ALPHA_COLORS):
        return {a: _CSR_ALPHA_COLORS[i] for i, a in enumerate(alphas)}
    return _alpha_colors(alphas)


def _ser(rows, m, a):
    pts = sorted([r for r in rows if r["method"] == m and r["alpha"] == a],
                 key=lambda r: r["n"])
    return {k: np.array([p[k] for p in pts]) for k in
            ("n", "avg_runtime", "rejection_rate", "ci_low", "ci_high", "Z")}


def _legend(ax, alphas, colors, methods, rows):
    have = {r["method"] for r in rows}             # only legend methods with data
    methods = [m for m in methods if m in have]
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
    _legend(ax, alphas, colors, PERM, rows)
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
    _legend(ax, alphas, colors, PERM + DIRECT, rows)
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
    _legend(ax, alphas, colors, PERM, rows)
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
    """Empirical CDF P(D <= eps) under H0 for Baseline and dKS-Sketch,
    with the theoretical bound 1 - exp(-n eps^2 / (4 ln 2n)).
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

    # dense grid over the plotted x-range; the direct union bound is vacuous
    # (flat 0) below eps~0.044, then climbs to ~1 by eps~0.05
    eps = np.linspace(0.010, 0.055, 500)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    xe, ye = cdf(ve)
    xa, ya = cdf(va)
    ax.step(xe, ye, where="post", color=_CSR_PALETTE[2], lw=2.0, ls=(0, (6, 4)),
            label="empirical Baseline")
    ax.step(xa, ya, where="post", color=_CSR_PALETTE[0], lw=2.2,
            label="empirical dKS-Sketch")
    ax.plot(eps, np.clip(1.0 - M.bound_union_direct(eps, n), 0.0, 1.0),
            color="0.45", lw=2, ls=":",
            label=r"bound  $1-(32/\varepsilon^2)e^{-n\varepsilon^2/2}$")
    ax.set_xlabel(r"threshold  $\varepsilon$  (dKS value)")
    ax.set_ylabel(r"CDF  $P(D \leq \varepsilon)$")
    ax.set_xlim(0.010, 0.055)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(rf"Null calibration (CDF)  ($d=2$, $n={n}$, $Z={Z}$, H$_0$: $P=Q$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig_calibration_cdf")


# --- CSR (computational-statistical) panels: Baseline (exact, dashed) vs
# --- Our Algo (approx, solid).  Read the small-n sweep in power_smalln.csv;
# --- larger fonts + explicit dash/solid method key; each method is plotted only
# --- over the n it has (Baseline clipped at 1000, Our Algo to 2000).
_CSR_CSV = "power_smalln.csv"
_CSR_LBL = {M.METHOD_EXACT: "dKS-Baseline", M.METHOD_SSS: "dKS-Sketch"}
# CSR-only marker/line styles (the shared global STYLE -- used by the other,
# already-finalized panels -- is left untouched).  Square for Baseline reads
# more clearly than the old triangle; explicit dash tuple for a crisp dashed
# curve both in-axes and in the legend key.
_CSR_STYLE = {
    M.METHOD_EXACT: dict(linestyle=(0, (6, 4)), marker="s", lw=1.9, markersize=6),
    M.METHOD_SSS:   dict(linestyle="-",         marker="o", lw=1.9, markersize=6),
}


def _legend_csr(ax, alphas, colors, methods, rows, fs=10, mloc="upper left",
                aloc="lower right"):
    have = {r["method"] for r in rows}
    methods = [m for m in methods if m in have]
    # Legend KEY readability: one marker per key (numpoints=1), but a long handle
    # + a tight dash so several dash segments span the key -- the single centered
    # marker covers ~1, leaving at least three dashes visible on Baseline; Our
    # Algo stays solid.
    _LEG_LS = {M.METHOD_EXACT: (0, (4, 3))}
    mh = []
    for m in methods:
        st = {k: v for k, v in _CSR_STYLE[m].items() if k != "alpha"}
        st["lw"] = 2.3
        if m in _LEG_LS:
            st["linestyle"] = _LEG_LS[m]
        mh.append(Line2D([0], [0], color="0.15",
                         label=_CSR_LBL.get(m, M.LABELS[m]), **st))
    ah = [Line2D([0], [0], color=colors[a], lw=3, label=rf"$\alpha={a:g}$")
          for a in alphas]
    leg1 = ax.legend(handles=mh, fontsize=fs,
                     title_fontsize=fs, loc=mloc, framealpha=0.92,
                     handlelength=4.8, handletextpad=0.7, numpoints=1)
    ax.add_artist(leg1)
    ax.legend(handles=ah, title="contamination", fontsize=fs, title_fontsize=fs,
              loc=aloc, framealpha=0.92)


def fig_power_vs_n_csr():
    rows = [r for r in _read_power(_CSR_CSV) if r["alpha"] < 0.35]  # Jeff: alpha in {.1,.2,.3}
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors_csr(alphas)
    delta = rows[0]["delta"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in PERM:
        for a in alphas:
            s = _ser(rows, m, a)
            if not len(s["n"]):
                continue
            # band = mean rejection rate +/- 1 standard error (std dev of the
            # estimate, sqrt(p(1-p)/Z)); -> 0 at saturation, wider with fewer
            # trials.  Shows ALL data; not a Wilson CI.
            p = s["rejection_rate"]
            se = np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / s["Z"])
            ax.fill_between(s["n"], p - se, p + se,
                            color=colors[a], alpha=0.16, lw=0)
            ax.plot(s["n"], p, color=colors[a], **_CSR_STYLE[m])
    ax.axhline(delta, color="0.55", ls=(0, (1, 1)), lw=1)
    ax.set_xlabel("sample size  $n$", fontsize=13)
    ax.set_ylabel(r"rejection rate  (mean $\pm$ 1 s.e.)", fontsize=13)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(r"Power vs $n$", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    _legend_csr(ax, alphas, colors, PERM, rows, mloc="center right")
    _save(fig, "fig_power_vs_n_csr")


def fig_runtime_vs_n_csr():
    rows = [r for r in _read_power(_CSR_CSV) if r["alpha"] < 0.35]  # Jeff: alpha in {.1,.2,.3}
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors_csr(alphas)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in PERM:
        for a in alphas:
            s = _ser(rows, m, a)
            if not len(s["n"]):
                continue
            ax.plot(s["n"], s["avg_runtime"], color=colors[a], **_CSR_STYLE[m])
    ax.set_xlabel("sample size  $n$", fontsize=13)
    ax.set_ylabel("avg runtime per test  (s)", fontsize=13)
    ax.set_yscale("log")
    ax.set_title(r"Runtime vs $n$", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format(y, ",g")))
    _legend_csr(ax, alphas, colors, PERM, rows)
    _save(fig, "fig_runtime_vs_n_csr")


def fig_power_vs_runtime_csr():
    rows = [r for r in _read_power(_CSR_CSV) if r["alpha"] < 0.35]  # Jeff: alpha in {.1,.2,.3}
    alphas = sorted({r["alpha"] for r in rows})
    colors = _alpha_colors_csr(alphas)
    delta = rows[0]["delta"]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for m in PERM:
        for a in alphas:
            s = _ser(rows, m, a)
            if not len(s["n"]):
                continue
            x = s["avg_runtime"]
            p = s["rejection_rate"]
            se = np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / s["Z"])
            ax.fill_between(x, p - se, p + se,
                            color=colors[a], alpha=0.16, lw=0)
            ax.plot(x, p, color=colors[a], **_CSR_STYLE[m])
    ax.axhline(delta, color="0.55", ls=(0, (1, 1)), lw=1)
    ax.set_xlabel("avg runtime per test  (s, log scale)", fontsize=13)
    ax.set_ylabel(r"rejection probability  (mean $\pm$ 1 s.e.)", fontsize=13)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("CSR power vs runtime", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: format(v, ",g")))
    _legend_csr(ax, alphas, colors, PERM, rows, mloc="center right")
    _save(fig, "fig_power_vs_runtime_csr")


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf / .png")


# === fig:plots (fig 6): single-dKS RUNTIME + OBSERVED-ERROR panels ===========
# Standalone from the power panels: these read results/runtime.csv (written by
# run_runtime.py -- a single dKS eval per rep under the null P=Q, so each
# algorithm's returned value IS its observed error).  method in {exact, approx}.
# Baseline = dashed+square (orange); dKS-Sketch = solid+circle (blue).
_RT_CSV = "runtime.csv"
_RT_ORDER = ["exact", "approx"]
_RT_LBL = {"exact": "dKS-Baseline", "approx": "dKS-Sketch"}
_RT_COLOR = {"exact": _CSR_PALETTE[2], "approx": _CSR_PALETTE[0]}  # orange / blue
_RT_STYLE = {
    "exact":  dict(linestyle=(0, (6, 4)), marker="s", lw=1.9, markersize=6),
    "approx": dict(linestyle="-",         marker="o", lw=1.9, markersize=6),
}


def _read_runtime(fname=_RT_CSV):
    rows = []
    with open(os.path.join(RESULTS, fname)) as f:
        for r in csv.DictReader(f):
            rows.append({"method": r["method"], "n": int(r["n"]),
                         "reps": int(r["reps"]), "runtime": float(r["runtime"]),
                         "obs_err": float(r["obs_err"])})
    return rows


def _rt_ser(rows, m):
    pts = sorted([r for r in rows if r["method"] == m], key=lambda r: r["n"])
    return {k: np.array([p[k] for p in pts]) for k in ("n", "runtime", "obs_err")}


def _rt_legend(ax, loc, bbox=None):
    # Custom handles so the Baseline key shows several clear dashes: a long handle
    # + tight dash pattern, single centered marker (same trick as the CSR legends).
    _LEG_LS = {"exact": (0, (4, 3)), "approx": "-"}
    handles = [Line2D([0], [0], color=_RT_COLOR[m], label=_RT_LBL[m], lw=2.3,
                      marker=_RT_STYLE[m]["marker"],
                      markersize=_RT_STYLE[m]["markersize"], linestyle=_LEG_LS[m])
               for m in _RT_ORDER]
    ax.legend(handles=handles, fontsize=10, loc=loc, framealpha=0.92,
              handlelength=4.8, handletextpad=0.7, numpoints=1, bbox_to_anchor=bbox)


def fig_rt_runtime_vs_n():
    rows = _read_runtime()
    fig, ax = plt.subplots(figsize=(6.8, 4.25))
    for m in _RT_ORDER:
        s = _rt_ser(rows, m)
        if not len(s["n"]):
            continue
        ax.plot(s["n"], s["runtime"] / 3600.0, color=_RT_COLOR[m],
                label=_RT_LBL[m], **_RT_STYLE[m])
    ax.set_xlabel("sample size  $n$", fontsize=13)
    ax.set_ylabel("runtime per dKS eval  (hours)", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))  # 1,000,000
    ax.set_title(r"Runtime vs $n$", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3)
    # dKS-Sketch's runtime (<=0.2 s) is ~0 on an hours axis -- point it out explicitly
    ax.annotate(r"dKS-Sketch: $\leq 0.2$ s",
                xy=(915000, 0.0), xytext=(815000, 1.18),
                fontsize=14, fontweight="bold", color=_RT_COLOR["approx"],
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=_RT_COLOR["approx"],
                                mutation_scale=22, connectionstyle="arc3,rad=-0.25"))
    _rt_legend(ax, "upper left")
    _save(fig, "fig_rt_runtime_vs_n")


def fig_rt_error_vs_n():
    rows = _read_runtime()
    fig, ax = plt.subplots(figsize=(6.8, 4.25))
    for m in _RT_ORDER:
        s = _rt_ser(rows, m)
        if not len(s["n"]):
            continue
        ax.plot(s["n"], s["obs_err"], color=_RT_COLOR[m],
                label=_RT_LBL[m], **_RT_STYLE[m])
    ax.set_xlabel("sample size  $n$", fontsize=13)
    ax.set_ylabel(r"observed error  (dKS value under $P=Q$)", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))  # 1,000,000
    ax.set_title(r"Observed error vs $n$", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3)
    _rt_legend(ax, "upper right")
    _save(fig, "fig_rt_error_vs_n")


def fig_rt_runtime_vs_error():
    rows = _read_runtime()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for m in _RT_ORDER:
        s = _rt_ser(rows, m)
        if not len(s["n"]):
            continue
        ax.plot(s["runtime"] / 3600.0, s["obs_err"], color=_RT_COLOR[m],
                label=_RT_LBL[m], **_RT_STYLE[m])
    ax.set_xlabel("runtime per dKS eval  (hours)", fontsize=13)
    ax.set_ylabel("observed error", fontsize=13)
    ax.set_title("Runtime vs observed error", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3)
    _rt_legend(ax, "lower right", bbox=(0.985, 0.17))
    # zoom inset: low-runtime / low-error corner -- dKS-Sketch reaches small error
    # at ~0 runtime while the Baseline needs far more
    axins = ax.inset_axes([0.44, 0.40, 0.53, 0.56])
    for m in _RT_ORDER:
        s = _rt_ser(rows, m)
        axins.plot(s["runtime"] / 3600.0, s["obs_err"], color=_RT_COLOR[m], **_RT_STYLE[m])
    axins.set_xlim(-0.0004, 0.005)
    axins.set_ylim(0, 0.022)
    axins.set_xticks([0, 0.001, 0.002, 0.003, 0.004, 0.005])
    axins.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    axins.tick_params(labelsize=8)
    axins.grid(alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=1.2)
    _save(fig, "fig_rt_runtime_vs_error")


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
