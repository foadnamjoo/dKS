"""Real-data demo on the Palmer penguins: flipper length (mm) vs body mass (g),
Adelie vs Gentoo.

(1) dKS two-sample permutation test detects the difference between the species.
(2) Unit-invariance: writing body mass in grams vs kilograms (or flipper length in
    mm vs cm) leaves dKS EXACTLY unchanged (it depends only on per-axis order), while
    a Gaussian-kernel MMD -- which relies on a Euclidean ground metric -- shifts ~20%.
This closes the loop with the introduction's height/weight, temp/pressure motivation.
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

DATA = os.path.join(HERE, "data", "penguins.csv")
FIGS = os.path.join(HERE, "figures")


def load(species, cols=("flipper_length_mm", "body_mass_g")):
    pts = []
    for r in csv.DictReader(open(DATA)):
        if r["species"] == species and all(r[c] for c in cols):
            pts.append([float(r[c]) for c in cols])
    return np.array(pts)


def mmd(P, Q):
    """Gaussian-kernel MMD, UNBIASED estimator (drops the k(x,x) diagonal), with the
    median-distance bandwidth recomputed on the pooled data (so it adapts to scale)."""
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


A, G = load("Adelie"), load("Gentoo")
print(f"Adelie n={len(A)}, Gentoo n={len(G)}")

# (1) dKS two-sample permutation test (natural units)
rng = np.random.default_rng(0)
rej, p, v_obs, _ = M.permutation_test(A, G, M.sss_stat, 500, 0.05, rng, False)
print(f"dKS(Adelie,Gentoo) = {v_obs:.3f}   permutation p = {p:.4g}   reject={rej}")

# (2) unit-invariance under REALISTIC unit choices a real dataset would actually use:
# body mass in grams or kilograms, flipper length in mm or cm. dKS depends only on
# per-axis order, so it is IDENTICAL for all four; MMD blends the two axes through a
# Euclidean metric, so it shifts when you switch grams <-> kilograms.
COMBOS = [  # (tick label, flipper scale from mm, body-mass scale from g)
    ("g\nmm",  1.0,   1.0),
    ("kg\nmm", 1.0,   1e-3),
    ("kg\ncm", 1e-1,  1e-3),
]
dks_u, mmd_u = [], []
for _, fs, ms in COMBOS:
    s = np.array([fs, ms])           # col 0 = flipper (mm), col 1 = body mass (g)
    dks_u.append(dks.approx(A * s, G * s))
    mmd_u.append(mmd(A * s, G * s))
dks_u, mmd_u = np.array(dks_u), np.array(mmd_u)
print(f"dKS-Sketch over unit choices: {dks_u.min():.4f}..{dks_u.max():.4f} "
      f"(spread {dks_u.max()-dks_u.min():.1e} -> invariant)")
print(f"MMD over unit choices: {mmd_u.min():.3f}..{mmd_u.max():.3f} "
      f"({mmd_u.max()/mmd_u.min()-1:+.0%} swing from grams <-> kilograms)")

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
ax1.scatter(A[:, 1], A[:, 0], s=20, c="#FF6D00", alpha=0.7, label=f"Adelie penguin ($n={len(A)}$)")
ax1.scatter(G[:, 1], G[:, 0], s=20, c="#00C853", alpha=0.7, label=f"Gentoo penguin ($n={len(G)}$)")
ax1.set_xlabel("body mass (g)"); ax1.set_ylabel("flipper length (mm)")
ax1.set_title(rf"Two penguin species — dKS test $p={p:.2g}$")
ax1.legend(frameon=True); ax1.grid(alpha=0.3)

# right panel: the four realistic unit choices as grouped bars, each method
# normalized to its value at the natural (g, mm) choice -> dKS stays at 1.00,
# MMD jumps when body mass switches to kilograms.
labels = [c[0] for c in COMBOS]
x = np.arange(len(COMBOS)); w = 0.38
ax2.axhline(1.0, color="0.6", ls=":", lw=1, zorder=0)
b1 = ax2.bar(x - w/2, dks_u / dks_u[0], w, color="#2979FF", label="dKS-Sketch")
b2 = ax2.bar(x + w/2, mmd_u / mmd_u[0], w, color="#D50000", label="MMD")
ax2.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
ax2.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_xlabel("unit choice   (top = body mass,  bottom = flipper length)")
ax2.set_ylabel("distance  /  value at the natural (g, mm) choice")
ax2.set_ylim(0, 1.35)
ax2.set_title("Unit-invariance")
ax2.legend(frameon=True); ax2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_penguins.png"), dpi=150)
fig.savefig(os.path.join(FIGS, "fig_penguins.pdf"))
print("wrote figures/fig_penguins.png /.pdf")
