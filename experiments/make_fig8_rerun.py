"""Fig-8 re-run DRAFT: Baseline (exact O(n^2), dashed) vs dKS-Sketch (sss, solid)
for the plan alphas {0.1, 0.03, 0.01}, B=100.  THREE SEPARATE figures (one per panel,
for the paper's 3-up \includegraphics, no suptitle -- that info goes in the caption):
  (1) fig8_power_vs_n  (2) fig8_runtime_vs_n  (3) fig8_power_vs_runtime.

Baseline from results/wk_baseline.csv (W=5); Sketch from results/scan_all_z20.csv
(W=20).  W = number of trials per cell (Jeff/paper notation).  Bands are mean
+/- 1 sample standard deviation = sqrt(p(1-p)) (Jeff's 1/W form), as the TRUE symmetric
interval p +/- SD (NOT clipped to [0,1]).  Runtime uses W=1
(deterministic).  CSV column header on disk stays "Z" (= W) to keep data loading.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")

PLAN_A = [0.1, 0.03, 0.01]
COL = {0.1: "#2979FF", 0.03: "#00C853", 0.01: "#F50057"}   # blue / green / red

# Sketch trial count (W) toggle. Both files are kept on disk (W=20 also backed up as
# scan_all_z20.BACKUP.csv) -- switch back to W=20 by changing just this one line:
SKETCH_CSV = "scan_w100.csv"       # "scan_z10.csv" (W=10) | "scan_all_z20.csv" (W=20) | "scan_w100.csv" (W=100)

# two legends, like before: line STYLE = method, COLOR = contamination alpha
STYLE_HANDLES = [
    Line2D([0], [0], color="0.25", ls="--", marker="s", mfc="white", ms=6, lw=2.0,
           label="Baseline"),
    Line2D([0], [0], color="0.25", ls="-", marker="o", ms=6, lw=2.3,
           label="dKS-Sketch"),
]
COLOR_HANDLES = [Line2D([0], [0], color=COL[a], lw=3, label=rf"$\alpha={a:g}$")
                 for a in PLAN_A]


def two_legends(ax, sloc=None, cloc=None):
    """Both legends stacked in the BOTTOM-RIGHT corner (method above contamination)."""
    lc = ax.legend(handles=COLOR_HANDLES, loc="lower right",
                   bbox_to_anchor=(1.0, 0.0), fontsize=8.5, framealpha=0.95,
                   title="contamination", title_fontsize=8.5)
    ax.add_artist(lc)
    ax.legend(handles=STYLE_HANDLES, loc="lower right",
              bbox_to_anchor=(1.0, 0.30), fontsize=8.5, framealpha=0.95,
              handlelength=4.0)   # longer handle so the Baseline dashes (cuts) show


def load(path, method):
    rows = [r for r in csv.DictReader(open(path)) if r.get("method") == method]
    out = {}
    for a in PLAN_A:
        sub = sorted([r for r in rows if round(float(r["alpha"]), 3) == a],
                     key=lambda r: int(r["n"]))
        if not sub:
            continue
        out[a] = dict(
            n=np.array([int(r["n"]) for r in sub]),
            p=np.array([float(r["rejection_rate"]) for r in sub]),
            W=np.array([int(r["Z"]) for r in sub]),
            rt=np.array([float(r["avg_runtime"]) for r in sub]),
        )
    return out


def load_baseline():
    """Baseline POWER from the parallel run ONLY: its trials 0..9 already INCLUDE
    the overnight trials 0..4 (identical seeds), so pooling both would double-count.
    Baseline RUNTIME from CLEAN single-core runs (overnight n<=50k + clean 100k),
    never the contention-inflated parallel runtimes."""
    # --- POWER: parallel raw only ---
    pw = {}
    p_par = os.path.join(RESULTS, "baseline_par_raw.csv")
    if os.path.exists(p_par):
        for r in csv.DictReader(open(p_par)):
            if r["method"] == "exact_sample":
                a = round(float(r["alpha"]), 3)
                if a in PLAN_A:
                    pw.setdefault((a, int(r["n"])), []).append(int(r["reject"]))
    # --- RUNTIME: clean single-core, alpha-INDEPENDENT (O(n^2) work depends only on n) ---
    rt = {}   # n -> clean seconds
    p_wk = os.path.join(RESULTS, "wk_baseline.csv")
    if os.path.exists(p_wk):
        for r in csv.DictReader(open(p_wk)):
            if round(float(r["alpha"]), 3) in PLAN_A:
                rt[int(r["n"])] = float(r["avg_runtime"])
    clean100k = None                       # clean single-core 100k, when ready
    p_c = os.path.join(RESULTS, "clean_rt_100k.csv")
    if os.path.exists(p_c):
        rows = list(csv.DictReader(open(p_c)))
        if rows:
            clean100k = float(rows[0]["avg_runtime"])
    par_rt = {}                            # n -> seconds, fallback only (inflated)
    p_pa = os.path.join(RESULTS, "baseline_par.csv")
    if os.path.exists(p_pa):
        for r in csv.DictReader(open(p_pa)):
            if round(float(r["alpha"]), 3) in PLAN_A:
                par_rt[int(r["n"])] = float(r["avg_runtime"])

    out = {}
    for a in PLAN_A:
        ns = sorted(n for (aa, n) in pw if aa == a)
        if not ns:
            continue
        nn, pp, WW, rr = [], [], [], []
        for n in ns:
            rej = pw[(a, n)]
            if n in rt:
                r_s = rt[n]                             # clean overnight (alpha-independent)
            elif n == 100000 and clean100k is not None:
                r_s = clean100k                         # clean single-core 100k
            else:
                r_s = par_rt.get(n, float("nan"))       # fallback
            nn.append(n); pp.append(sum(rej) / len(rej)); WW.append(len(rej)); rr.append(r_s)
        out[a] = dict(n=np.array(nn), p=np.array(pp), W=np.array(WW),
                      rt=np.array(rr, dtype=float))
    return out


def load_sketch_z10():
    """dKS-Sketch from the FIRST 10 trials only (per request: W=10 throughout).
    Reads scan_all_z20_raw.csv (W=20 on disk) and keeps trials 0..9 per cell."""
    cells = {}
    p = os.path.join(RESULTS, "scan_all_z20_raw.csv")
    for r in csv.DictReader(open(p)):
        if r["method"] != "sss" or int(r["trial"]) >= 10:
            continue
        a = round(float(r["alpha"]), 3)
        if a in PLAN_A:
            cells.setdefault((a, int(r["n"])), []).append(
                (int(r["reject"]), float(r["runtime"])))
    out = {}
    for a in PLAN_A:
        ns = sorted(n for (aa, n) in cells if aa == a)
        if not ns:
            continue
        out[a] = dict(
            n=np.array(ns),
            p=np.array([np.mean([t[0] for t in cells[(a, n)]]) for n in ns]),
            W=np.array([len(cells[(a, n)]) for n in ns]),
            rt=np.array([np.mean([t[1] for t in cells[(a, n)]]) for n in ns]),
        )
    return out


base = load_baseline()
# dKS-Sketch: prefer the fresh W=10 re-run (scan_z10.csv); fall back to the complete
# W=20 scan until that finishes. (Subsampling W=20->10 from raw was impossible -- the
# raw only had a couple alpha -- so we re-ran the scan cleanly at W=10.)
_sk_csv = os.path.join(RESULTS, SKETCH_CSV)
if not os.path.exists(_sk_csv):
    _sk_csv = os.path.join(RESULTS, "scan_all_z20.csv")   # fallback until W=10 re-run lands
sketch = load(_sk_csv, "sss")

# Sketch RUNTIME from the CLEAN sequential scan (scan_all_z20.csv): the high-W power
# run is parallel, so its runtimes are contention-inflated (~30% at 100k). Runtime is
# deterministic and W-independent, so the clean values are the honest ones to report.
_clean_rt = {}
for _r in csv.DictReader(open(os.path.join(RESULTS, "scan_all_z20.csv"))):
    if _r["method"] == "sss":
        _clean_rt[(round(float(_r["alpha"]), 3), int(_r["n"]))] = float(_r["avg_runtime"])
for _a in sketch:
    sketch[_a]["rt"] = np.array([_clean_rt.get((_a, int(_n)), _rt)
                                 for _n, _rt in zip(sketch[_a]["n"], sketch[_a]["rt"])])

# Plotted-range choice (data on disk untouched): stop alpha=0.03 Sketch at 300k --
# drop the redundant flat-1.0 400k/500k tail. Nothing hidden (flat past 150k); the
# large-n "stays cheap" story is still shown by alpha=0.01 out to 700k.
if 0.03 in sketch:
    m = sketch[0.03]["n"] <= 300000
    sketch[0.03] = {k: v[m] for k, v in sketch[0.03].items()}


def sd_band(s):
    """Sample standard deviation of the W binary {0,1} reject indicators in a cell, using
    Jeff's denominator W:  sd = sqrt( (1/W) * sum_j (v_j - p)^2 ) = sqrt(p(1-p)) for 0/1
    data (identically np.std(values, ddof=0)).  This is the sample SD itself --
    do NOT divide by W or by sqrt(W) (it is deliberately not shrunk by the trial count)."""
    return np.sqrt(np.clip(s["p"] * (1 - s["p"]), 0.0, None))


def sd_bounds(s):
    """(lower, upper) for the +/-1 sample-SD band as the TRUE symmetric interval p +/- sd.
    Deliberately NOT clipped to [0,1]: this is a standard deviation, not a confidence
    interval, so the band may extend below 0 or above 1."""
    sd = sd_band(s)
    return s["p"] - sd, s["p"] + sd


def w_summary(d):
    """Readable 'which W for which n' string, pooled across alpha (W can vary by n)."""
    from collections import defaultdict
    wmap = defaultdict(set)
    for a in d:
        for n, w in zip(d[a]["n"], d[a]["W"]):
            wmap[int(w)].add(int(n))
    if len(wmap) == 1:
        return f"$W{{=}}{next(iter(wmap))}$ (all $n$)"
    parts = []
    for w in sorted(wmap, reverse=True):
        ns = sorted(wmap[w])
        rng = f"$n{{=}}{ns[0]:,}$" if len(ns) == 1 else f"$n\\leq{ns[-1]:,}$"
        parts.append(f"$W{{=}}{w}$ ({rng})")
    return ", ".join(parts)


def _xaxis_n(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xticks([0, 100000, 200000, 300000, 400000, 500000, 600000, 700000])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


PANELS = []   # (canonical stem, figure, also_save_sd_copy) -- one standalone figure per panel

# ---- (1) power vs n ----  (+/-1 sample-SD bands on rejection rate)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
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
ax.set_xlabel("sample size  $n$", fontsize=12)          # LINEAR n (paper convention)
ax.set_ylabel(r"rejection rate  (mean $\pm$ 1 sample SD)", fontsize=12)
ax.set_title("Power vs $n$", fontsize=13)
_xaxis_n(ax); ax.grid(alpha=0.3)
two_legends(ax)
fig.tight_layout()
PANELS.append(("fig8_power_vs_n", fig, True))

# ---- (2) runtime vs n (LINEAR n, runtime in SECONDS on LOG scale) ----
fig, ax = plt.subplots(figsize=(5.4, 4.6))
for a in reversed(PLAN_A):   # draw red(0.01) first -> green -> blue(0.1) on top, so each
    c = COL[a]               # color shows in its own n-range (runtimes overlap: alpha-independent)
    if a in sketch:
        s = sketch[a]
        ax.plot(s["n"], s["rt"], "-o", color=c, ms=5, lw=2.3)
    if a in base:
        b = base[a]
        ax.plot(b["n"], b["rt"], "--s", color=c, ms=5, lw=2.0, mfc="white")
ax.set_yscale("log")
ax.set_xlabel("sample size  $n$", fontsize=12)
ax.set_ylabel("avg runtime per test  (s, log scale)", fontsize=12)
ax.set_title("Runtime vs $n$", fontsize=13)
_xaxis_n(ax)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else f"{v:g}"))
ax.grid(alpha=0.3, which="both")
two_legends(ax)
fig.tight_layout()
PANELS.append(("fig8_runtime_vs_n", fig, False))

# ---- (3) power vs runtime ----  (+/-1 sample-SD bands on rejection rate; Jeff's key panel)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
for a in PLAN_A:
    c = COL[a]
    if a in sketch:
        s = sketch[a]
        o = np.argsort(s["rt"])                         # sort by runtime for a clean fill
        lo, hi = sd_bounds(s)
        ax.fill_between(s["rt"][o], lo[o], hi[o], color=c, alpha=0.25, lw=0)
        ax.plot(s["rt"], s["p"], "-o", color=c, ms=5, lw=2.3)
    if a in base:
        b = base[a]
        o = np.argsort(b["rt"])
        lo, hi = sd_bounds(b)
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
fig.tight_layout()
PANELS.append(("fig8_power_vs_runtime", fig, True))

written = []
for _name, _f, _sd_copy in PANELS:   # canonical files for Overleaf + optional _sd checking copies
    for stem in [_name] + ([_name + "_sd"] if _sd_copy else []):
        _f.savefig(os.path.join(FIGS, stem + ".png"), dpi=150)
        _f.savefig(os.path.join(FIGS, stem + ".pdf"))   # vector PDF for Overleaf / the paper
        written.append(stem)
print("wrote:", ", ".join(written), "(each .png + .pdf)")
print("caption facts -> dKS-Sketch trials:", w_summary(sketch), "| Baseline trials:", w_summary(base))
for a in PLAN_A:
    bs = f"Baseline n->{base[a]['n'][-1]:,} ({base[a]['rt'][-1]:.0f}s)" if a in base else "Baseline: NONE"
    sk = f"Sketch n->{sketch[a]['n'][-1]:,} ({sketch[a]['rt'][-1]:.1f}s)" if a in sketch else "Sketch: NONE"
    print(f"  a={a:<5}: {bs:<32} | {sk}")
