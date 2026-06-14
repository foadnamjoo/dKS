# dKS experiments

Two-sample experiments built on the `dks` Python module (the pybind11 binding —
build it first from the repo root with `pip install .`).

```
generators.py        uniform / gaussian / Huber-contaminated point sets
methods.py           the two-sample tests (permutation + direct threshold)
run_power.py         empirical rejection rate vs (alpha, n)   -> power.pdf
run_calibration.py   null tail of dKS vs theoretical bound    -> calibration.pdf
```

## Setup

```sh
# from the repo root, with the binding installed:
pip install .                      # provides `import dks`
pip install -r experiments/requirements.txt
```

## Null calibration (Jeff's plot)

Draws many uniform samples under H0 and compares the empirical tail
`P(D_n >= eps)` to the bound `exp(-n eps^2 / 4)`. The empirical curve sitting
below the bound is the evidence that the calibrated test controls its size
(addresses the reviewer concern about conservativeness).

```sh
python experiments/run_calibration.py --n 200 --reps 1500
```

## Power / empirical rejection

P is clean `Uniform([-1,1]^2)`; Q is Huber-contaminated at level `alpha`
(`alpha = 0` is the null, a size check; `alpha > 0` is power). Compares **Our
Method** (direct analytic threshold, no bootstrap) against the **B-Bootstrap
Exact** permutation test, one panel per `alpha`.

```sh
python experiments/run_power.py            # quick demo
python experiments/run_power.py --full     # full plan: alpha in {0,.05,.1,.2}, n in {20..500}
```

Defaults to the **fast O(n log n) approximation** for the statistic — the exact
O(n^2) statistic inside a permutation loop is intractable at large n. Pass
`--exact` only for small-n checks.

## ⚠ Open item — the direct-threshold constant

`methods.threshold_d2()` currently uses a **provisional** constant
(`C2 = 4 ln(2n)`) and a placeholder functional form. With it, "Our Method" is so
conservative it has near-zero power — which is itself a demonstration of the
reviewer's concern. **Confirm the exact d=2 threshold / C_2 from the paper's
Section 6.2 with Peter**, then set it via `threshold_d2(..., C2=...)` or replace
the function. The permutation test needs no such constant and is correct as-is.
