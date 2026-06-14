"""Minimal usage of the dks Python binding.

    pip install .
    python bindings/python/example.py
"""
import numpy as np
import dks

rng = np.random.default_rng(0)
P = rng.random((5000, 2))                 # P ~ Uniform([0,1]^2)
Q = rng.normal(0.5, 0.15, size=(5000, 2))  # Q ~ Gaussian bump

print("exact  dKS =", dks.exact(P, Q))
print("approx dKS =", dks.approx(P, Q))            # default 2*sqrt(n) grid
print("approx dKS =", dks.approx(P, Q, eps=0.02))  # within 0.02 of exact
