# dKS experiments

Two-sample **power** and null-**calibration** experiments built on the `dks`
Python module (the pybind11 binding — build it first from the repo root with
`pip install .`, then `import dks` works inside the venv).

```
generators.py        point-set generators (uniform_square, huber_mixture, ...)
methods.py           statistics, permutation test, and the direct test
run_power.py         Huber power study           -> results/power.csv
run_calibration.py   null tail of dKS vs bound   -> results/calibration.csv
plots.py             reads results/*.csv         -> figures/*.pdf (+ .png)
```

`results/` and `figures/` are git-ignored (regenerated on every run).

## Setup

```sh
# from the repo root, with the binding installed into the venv:
pip install .                                # provides `import dks`
pip install -r experiments/requirements.txt  # numpy, matplotlib
```

## Naming (one statistic, two ways to compute it)

These names are constants in `methods.py` and are used everywhere, including
plot legends:

| call         | name                          | legend tag        | cost        |
|--------------|-------------------------------|-------------------|-------------|
| `dks.exact`  | **exact-sample dKS**          | `exact-sample dKS`| `O(n^2)`    |
| `dks.approx` | **Sample-Sketch-Solve dKS**   | `SSS-dKS`         | `O(n log n)`|

"exact-sample" means exact for the *sample*, not the population — never
"baseline" or plain "exact dKS". **Sample-Sketch-Solve** is Jeff's framework
name for the grid approximation.

## Methods compared

We compare **only** exact-sample dKS vs Sample-Sketch-Solve dKS (plus the direct
variant). This is an *algorithmic* comparison — the same dKS statistic computed
exactly vs sketched — not a statistic bake-off, so **MMD / energy distance /
Bickel are intentionally excluded**: they would answer a different question
(which divergence?) than the one here (exact vs near-linear computation of dKS).

1. **exact-sample dKS** — `permutation_test(stat_fn=exact_stat)`. The rigorous
   reference; exactly valid for any `B`, needs no analytic constant.
2. **SSS-dKS** — `permutation_test(stat_fn=sss_stat)`, same `B`. The main method.
3. **SSS-dKS direct (PENDING Peter's C2)** — `sss_direct_test`: B-free, rejects
   when `dks.approx(P,Q) > tau`, with the **clean** threshold from inverting
   `delta = exp(-n*eps^2/4)`:

   ```
   tau = 2 * sqrt( ln(1/delta) / n )
   ```

   > ⚠ **Open constant.** This clean form has no `ln(2n)` factor. The paper's
   > Section 6.2 form (kept as `methods.threshold_d2`, `C2 = 4 ln(2n)`) does.
   > Peter may also use a different direct variant. `run_calibration.py` decides
   > empirically which constant is right: if the empirical null tail hugs
   > `exp(-n eps^2/4)` from below, the clean inversion is the correct one.

The permutation test supports randomized tie-breaking (`--randomized`,
Hemerik & Goeman 2018) for *exactly* level `delta`; the conservative p-value
`(1 + #{T_b >= T_obs})/B` is the default.

## Power experiment

`P = Uniform([-1,1]^2)`; `Q_alpha` is a Huber mixture — each point is uniform
w.p. `(1-alpha)`, else a central `N(0, sigma^2 I)` bump clipped into the square.
`alpha = 0` is the null (rejection rate ≈ `delta`, a size check); `alpha > 0` is
power. Per `(method, n, alpha)` it records `rejection_rate` and `avg_runtime`.

```sh
python experiments/run_power.py            # small verification grid (~20 s)
# scale up for the paper:
python experiments/run_power.py --n 50 100 200 400 800 --B 300 --Z 300
```

Flags: `--n` (sizes), `--alpha` (levels), `--B` (permutations), `--Z` (trials),
`--delta`, `--eps` (SSS grid; `<=0` = `2*sqrt(n)`), `--seed`, `--randomized`.

## Null calibration

`P = Q = Uniform([-1,1]^2)`, `n` large; `Z` fresh redraws of the SSS-dKS
statistic. `plots.py` overlays the empirical tail `P_hat(dKS >= eps)` on
`exp(-n eps^2/4)`.

```sh
python experiments/run_calibration.py --n 10000 --Z 1000     # ~2 s
```

## Plots

```sh
python experiments/plots.py        # writes all four to figures/
```

| figure                      | x                  | y                  |
|-----------------------------|--------------------|--------------------|
| `fig_runtime_vs_n`          | `n`                | avg runtime (log)  |
| `fig_power_vs_n`            | `n`                | rejection rate     |
| `fig_power_vs_runtime` ★    | avg runtime (log)  | rejection rate     |
| `fig_calibration`           | `eps`              | tail prob (log)    |

★ headline (per Jeff): rejection probability vs runtime. Styling — exact-sample
= dashed + triangle, SSS-dKS = solid + circle, SSS-dKS direct = dotted; one
color per `alpha`.

## Suggested final config

The defaults are a fast verification grid. For the paper:

```sh
python experiments/run_power.py --n 50 100 200 400 800 --B 300 --Z 300
python experiments/run_calibration.py --n 10000 --Z 2000
python experiments/plots.py
```

i.e. `n` up to **400–800**, `B` ≈ **200–500**, `Z` ≥ **300** (tighter rejection
estimates, and a wider runtime axis where the `O(n^2)` vs `O(n log n)` gap is
dramatic). Runtime scales roughly with `n_cells * Z * B * cost(stat)`.

## Typical findings (small verification grid)

- Power rises with `alpha` and with `n`; `alpha = 0` sits near `delta = 0.05`.
- SSS-dKS reaches essentially the **same power** as exact-sample dKS at **far
  lower runtime** (e.g. at `n = 200`, ~3 ms vs ~43 ms — the gap widens with `n`).
- The direct method is **~1000x cheaper** (one approx evaluation, no `B`) with
  non-trivial, emerging power; its conservativeness is exactly the threshold-
  constant question above.
- The empirical null tail sits **below** `exp(-n eps^2/4)` — the calibrated test
  controls its size.
