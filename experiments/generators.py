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
