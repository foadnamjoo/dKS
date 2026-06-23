"""Real-data demo: human height & weight, female vs male adults (NHANES 2017-2018).

(1) dKS two-sample permutation test detects the difference between the sexes.
(2) Unit-invariance: the SAME person is "178 cm, 82 kg" in metric and "70 in, 181 lb"
    in US units. dKS depends only on per-axis order, so it gives the IDENTICAL answer
    for every metric/US unit choice; a Gaussian-kernel MMD -- which blends the axes
    through a Euclidean metric -- shifts when you switch cm<->in or kg<->lb.
This is the height/weight motivation from the introduction, on real data.
"""
import os, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dks
import methods as M

DATA = os.path.join(HERE, "data", "nhanes.csv")
FIGS = os.path.join(HERE, "figures")


def load_sex():
    """Return (female, male) arrays of [height_cm, weight_kg] (NHANES adults >= 18)."""
    F, Mn = [], []
    for r in csv.DictReader(open(DATA)):
        pt = [float(r["height_cm"]), float(r["weight_kg"])]
        (Mn if r["sex"] == "M" else F).append(pt)
    return np.array(F), np.array(Mn)


def mmd(P, Q):
    """Gaussian-kernel MMD, unbiased estimator, median-distance bandwidth on pooled data."""
    def sq(X, Y):
        return np.maximum((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2 * X @ Y.T, 0.0)
    Z = np.vstack([P, Q])
    s2 = np.median(sq(Z, Z)[np.triu_indices(len(Z), 1)]) or 1.0
    k = lambda X, Y: np.exp(-sq(X, Y) / s2)
    m, n = len(P), len(Q)
    Kpp, Kqq, Kpq = k(P, P), k(Q, Q), k(P, Q)
    v = ((Kpp.sum() - np.trace(Kpp)) / (m * (m - 1))
         + (Kqq.sum() - np.trace(Kqq)) / (n * (n - 1)) - 2 * Kpq.mean())
    return float(np.sqrt(max(v, 0.0)))


F, Mn = load_sex()
print(f"Female n={len(F)}, Male n={len(Mn)}")

# (1) dKS two-sample permutation test (dKS-Sketch statistic)
rng = np.random.default_rng(0)
rej, p, v_obs, _ = M.permutation_test(F, Mn, M.sss_stat, 500, 0.05, rng, False)
print(f"dKS(Female,Male) = {v_obs:.3f}   permutation p = {p:.4g}   reject={rej}")

# (2) unit-invariance under realistic metric/US unit choices.
# columns: 0 = height (cm), 1 = weight (kg).  in = cm/2.54,  lb = kg*2.2046226.
CM_PER_IN, LB_PER_KG = 2.54, 2.2046226
COMBOS = [  # (tick label top=height bottom=weight, height scale from cm, weight scale from kg)
    ("cm\nkg", 1.0,          1.0),         # metric (reference)
    ("in\nkg", 1 / CM_PER_IN, 1.0),
    ("cm\nlb", 1.0,          LB_PER_KG),
    ("in\nlb", 1 / CM_PER_IN, LB_PER_KG),  # US
]
dks_u, mmd_u = [], []
for _, hs, ws in COMBOS:
    s = np.array([hs, ws])
    dks_u.append(dks.approx(F * s, Mn * s))
    mmd_u.append(mmd(F * s, Mn * s))
dks_u, mmd_u = np.array(dks_u), np.array(mmd_u)
print(f"dKS-Sketch over unit choices: {dks_u.min():.4f}..{dks_u.max():.4f} "
      f"(spread {dks_u.max()-dks_u.min():.1e} -> invariant)")
print(f"MMD over unit choices: {mmd_u.min():.3f}..{mmd_u.max():.3f} "
      f"({mmd_u.max()/mmd_u.min()-1:+.0%} swing metric <-> US)")

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
ax1.scatter(F[:, 0], F[:, 1], s=10, c="#FF6D00", alpha=0.35, lw=0, label=f"female ($n={len(F)}$)")
ax1.scatter(Mn[:, 0], Mn[:, 1], s=10, c="#00C853", alpha=0.35, lw=0, label=f"male ($n={len(Mn)}$)")
ax1.set_xlabel("height (cm)"); ax1.set_ylabel("weight (kg)")
ax1.set_title(rf"Height & weight by sex — dKS test $p={p:.2g}$")
leg = ax1.legend(frameon=True, markerscale=2.2)
for h in leg.legend_handles:
    h.set_alpha(1)
ax1.grid(alpha=0.3)

labels = [c[0] for c in COMBOS]
x = np.arange(len(COMBOS))
dks_rel, mmd_rel = dks_u / dks_u[0], mmd_u / mmd_u[0]
ax2.axhline(1.0, color="0.8", ls=":", lw=1, zorder=0)
ax2.plot(x, dks_rel, "-o", color="#2979FF", ms=8, lw=2.5, label="dKS-Sketch", zorder=3)
ax2.plot(x, mmd_rel, "-s", color="#D50000", ms=8, lw=2.5, mfc="white", mew=2, label="MMD", zorder=3)
for xi, v in zip(x, mmd_rel):
    ax2.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, -15),
                 ha="center", fontsize=8.5, color="#D50000")
ax2.annotate("exactly 1.00 — invariant", (x[-1], 1.0), textcoords="offset points",
             xytext=(-4, 9), ha="right", fontsize=8.5, color="#2979FF")
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_xlim(-0.35, len(COMBOS) - 0.65)
ax2.set_ylim(0.5, 1.1)
ax2.set_xlabel("unit choice   (top = height,  bottom = weight)")
ax2.set_ylabel("distance  /  value in metric (cm, kg)")
ax2.set_title("Unit-invariance")
ax2.legend(frameon=True, loc="lower left"); ax2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_heightweight.png"), dpi=150)
fig.savefig(os.path.join(FIGS, "fig_heightweight.pdf"))
print("wrote figures/fig_heightweight.png /.pdf")
