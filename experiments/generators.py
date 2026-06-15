"""Data generators for the dKS two-sample experiments.

The power study compares a clean reference P against a Huber-contaminated Q:

    P ~ Uniform([-1, 1]^2)
    Q ~ (1 - alpha) * Uniform([-1, 1]^2)  +  alpha * N(bump_mean, bump_sd^2 I)

alpha = 0 reproduces the null (P and Q identically distributed); alpha > 0 is the
alternative. Each generator takes a numpy Generator so runs are reproducible.
"""
import numpy as np


def uniform(n, rng, low=-1.0, high=1.0):
    """n points from Uniform([low, high]^2)."""
    return rng.uniform(low, high, size=(int(n), 2))


def gaussian(n, rng, mean=(0.0, 0.0), sd=1.0):
    """n points from N(mean, sd^2 I)."""
    return rng.normal(loc=mean, scale=sd, size=(int(n), 2))


def huber_contaminated(n, rng, alpha, low=-1.0, high=1.0,
                       bump_mean=(0.0, 0.0), bump_sd=0.2):
    """Huber mixture: each point is Uniform([low, high]^2) with prob (1 - alpha),
    otherwise drawn from a Gaussian bump N(bump_mean, bump_sd^2 I).
    """
    n = int(n)
    pts = rng.uniform(low, high, size=(n, 2))
    is_bump = rng.random(n) < alpha
    k = int(is_bump.sum())
    if k:
        pts[is_bump] = rng.normal(loc=bump_mean, scale=bump_sd, size=(k, 2))
    return pts


# ---------------------------------------------------------------------------
# 2D Huber power-experiment pair.  P is clean uniform; Q_alpha mixes in a central
# Gaussian bump with probability alpha.  The bump is TRUNCATED to [-1, 1]^2 by
# rejection sampling (out-of-box points are redrawn, never coordinate-clamped),
# so Q carries no boundary atoms -- clamping would pile mass on the edges and
# manufacture a spurious, trivially-detectable difference.  alpha = 0 reduces
# exactly to uniform_square (the null).
# ---------------------------------------------------------------------------
def uniform_square(n, rng):
    """n x 2 points from Uniform([-1, 1]^2).  This is the reference P."""
    return rng.uniform(-1.0, 1.0, size=(int(n), 2))


def _truncated_normal_box(k, sigma, rng, lo=-1.0, hi=1.0):
    """k draws from N(0, sigma^2 I) restricted to [lo, hi]^2 by rejection.

    Out-of-box candidates are redrawn (no clamping), giving a genuine truncated
    Gaussian with no boundary atoms.  At sigma=0.15 the box is ~6.7 sigma wide,
    so rejection essentially never fires; the loop just guarantees correctness.
    """
    out = np.empty((k, 2))
    filled = 0
    while filled < k:
        cand = rng.normal(0.0, sigma, size=(k - filled, 2))
        inside = cand[np.all((cand >= lo) & (cand <= hi), axis=1)]
        take = min(len(inside), k - filled)
        out[filled:filled + take] = inside[:take]
        filled += take
    return out


def huber_mixture(n, alpha, rng, sigma=0.15):
    """Huber-contaminated Q_alpha on [-1, 1]^2 (2D).

    Each of the n points is Uniform([-1, 1]^2) with probability (1 - alpha),
    otherwise from a TRUNCATED N(0, sigma^2 I) at the origin (rejection sampling
    via _truncated_normal_box -- no coordinate clamping, hence no edge atoms).
    alpha = 0 returns a plain Uniform([-1, 1]^2) sample, identical in
    distribution to uniform_square (the null).
    """
    n = int(n)
    pts = rng.uniform(-1.0, 1.0, size=(n, 2))
    if alpha <= 0.0:
        return pts                       # exactly uniform_square: the null
    is_bump = rng.random(n) < alpha
    k = int(is_bump.sum())
    if k:
        pts[is_bump] = _truncated_normal_box(k, sigma, rng)
    return pts
