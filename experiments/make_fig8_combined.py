"""Combined 3-panel version of Figure 7 in ONE figure (1x3):
  power vs n  |  runtime vs n  |  CSR power vs runtime.
Reuses the data + sample-SD bands from make_fig8_rerun.py (single source of truth) and
writes figures/fig8_combined.{png,pdf}.  The paper uses the 3 separate panels; this is a
single-image version / preview of the whole of Figure 7.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import make_fig8_rerun as M   # loads base/sketch + helpers (also (re)writes the standalone panels)

base, sketch, COL, PLAN_A = M.base, M.sketch, M.COL, M.PLAN_A
sd_bounds, two_legends, _xaxis_n, FIGS = M.sd_bounds, M.two_legends, M._xaxis_n, M.FIGS


def panel_power_vs_n(ax):
    for a in PLAN_A:
        c = COL[a]
        if a in sketch:
            s = sketch[a]; lo, hi = sd_bounds(s)
            ax.fill_between(s["n"], lo, hi, color=c, alpha=0.25, lw=0)
            ax.plot(s["n"], s["p"], "-o", color=c, ms=5, lw=2.3)
        if a in base:
            b = base[a]; lo, hi = sd_bounds(b)
            ax.fill_between(b["n"], lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(b["n"], b["p"], "--s", color=c, ms=5, lw=2.0, mfc="white")
    ax.axhline(1.0, color="0.65", ls=":", lw=1); ax.axhline(0.05, color="0.65", ls=":", lw=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("sample size  $n$", fontsize=12)
    ax.set_ylabel(r"rejection rate  (mean $\pm$ 1 sample SD)", fontsize=12)
    ax.set_title("Power vs $n$", fontsize=13)
    _xaxis_n(ax); ax.grid(alpha=0.3)
    two_legends(ax)


def panel_runtime_vs_n(ax):
    for a in reversed(PLAN_A):
        c = COL[a]
        if a in sketch:
            ax.plot(sketch[a]["n"], sketch[a]["rt"], "-o", color=c, ms=5, lw=2.3)
        if a in base:
            ax.plot(base[a]["n"], base[a]["rt"], "--s", color=c, ms=5, lw=2.0, mfc="white")
    ax.set_yscale("log")
    ax.set_xlabel("sample size  $n$", fontsize=12)
    ax.set_ylabel("avg runtime per test  (s, log scale)", fontsize=12)
    ax.set_title("Runtime vs $n$", fontsize=13)
    _xaxis_n(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else f"{v:g}"))
    ax.grid(alpha=0.3, which="both")
    two_legends(ax)


def panel_power_vs_runtime(ax):
    for a in PLAN_A:
        c = COL[a]
        if a in sketch:
            s = sketch[a]; o = np.argsort(s["rt"]); lo, hi = sd_bounds(s)
            ax.fill_between(s["rt"][o], lo[o], hi[o], color=c, alpha=0.25, lw=0)
            ax.plot(s["rt"], s["p"], "-o", color=c, ms=5, lw=2.3)
        if a in base:
            b = base[a]; o = np.argsort(b["rt"]); lo, hi = sd_bounds(b)
            ax.fill_between(b["rt"][o], lo[o], hi[o], color=c, alpha=0.18, lw=0)
            ax.plot(b["rt"], b["p"], "--s", color=c, ms=5, lw=2.0, mfc="white")
    ax.set_xscale("log")
    ax.axhline(1.0, color="0.65", ls=":", lw=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("avg runtime per test  (s, log scale)", fontsize=12)
    ax.set_ylabel(r"rejection rate  (mean $\pm$ 1 sample SD)", fontsize=12)
    ax.set_title("CSR power vs runtime", fontsize=13)
    ax.grid(alpha=0.3, which="both")
    two_legends(ax)


fig, axes = plt.subplots(1, 3, figsize=(19.8, 4.1))
panel_power_vs_n(axes[0])
panel_runtime_vs_n(axes[1])
panel_power_vs_runtime(axes[2])
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig8_combined.png"), dpi=150)
fig.savefig(os.path.join(FIGS, "fig8_combined.pdf"))
print("wrote figures/fig8_combined.png /.pdf")
