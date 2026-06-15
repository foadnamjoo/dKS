# dKS experiments (2D)

Two-sample **power** and null-**calibration** experiments built on the `dks`
Python module (the pybind11 binding — build it first from the repo root with
`pip install .`, then `import dks` works inside the venv). **2D only.**

```
generators.py        uniform_square (P), huber_mixture (Q, truncated bump), ...
methods.py           statistics, permutation test, two direct thresholds + bounds
run_power.py         Huber power study (with CIs)   -> results/power.csv
run_calibration.py   null tail, exact AND approx    -> results/calibration.csv
plots.py             reads results/*.csv            -> figures/*.pdf (+ .png)
```

`results/` and `figures/` are git-ignored (regenerated on every run).

## Setup

```sh
pip install .                                 # provides `import dks`
pip install -r experiments/requirements.txt   # numpy, matplotlib
```

## Naming (one statistic, computed two ways)

Constants in `methods.py`, used everywhere including plot legends:

| call         | name                        | legend tag | what it is                                            |
|--------------|-----------------------------|------------|-------------------------------------------------------|
| `dks.exact`  | **exact-sample dKS**        | exact-sample dKS | the **DIRECT, brute-force `O(n^2)`** dKS on the sample |
| `dks.approx` | **Sample-Sketch-Solve dKS** | SSS-dKS    | deterministic `O(n log n)` grid approximation         |

"exact-sample" = exact for the *sample*, **not** a theoretically-optimal exact
2D algorithm and not a population quantity. We deliberately keep the `O(n^2)`
brute force as the baseline; the experiments show how close and how much faster
the approximation is than computing dKS directly. **SSS-dKS is deterministic**
(fixed `2*sqrt(n)` grid, no random sub-sampling — verified by repeated calls on
fixed input), so it is a valid permutation-test statistic with no extra seeding.

## Methods compared

Only **exact-sample dKS** vs **SSS-dKS** (+ two direct variants). This is an
*algorithmic* comparison — the same dKS statistic computed exactly vs sketched
vs via an analytic threshold — so **MMD / energy distance / Bickel are
intentionally excluded**: they answer a different question (which divergence?)
than the one here (exact vs near-linear computation of dKS).

1. **exact-sample dKS** — `permutation_test(stat_fn=exact_stat)`. The validity
   anchor; valid / **conservative** at level `delta` for any `B`. dKS is discrete
   so ties occur — report "valid/conservative level delta", not "exact".
   `--randomized` adds Hemerik & Goeman tie-breaking for *exactly* level delta.
2. **SSS-dKS** — `permutation_test(stat_fn=sss_stat)`, same `B`. The main method.
3. **SSS-dKS direct (clean)** and **(union/Sec-6.2)** — B-free: reject when
   `dks.approx > tau`. Two candidate thresholds, **neither claimed valid without
   proof (PENDING Peter's C2)**:

   ```
   tau_clean(n, delta) = 2*sqrt( ln(1/delta) / n )                 # from delta = exp(-n eps^2/4)
   tau_union(n, delta) = sqrt( (4*ln(2n)/n) * log2(1/delta) )      # paper Sec 6.2 union/VC factor
   ```

   Which threshold is honest is open for Peter. The permutation test is the
   validity anchor; the **calibration plot is the empirical check** — it overlays
   the exact null tail on both matching bounds (`clean = exp(-n eps^2/4)`,
   `union = 2^(-n eps^2/(4 ln 2n))`).

## Power experiment

`P = uniform_square(n)`; `Q = huber_mixture(n, alpha)` — Huber mixture whose
central Gaussian bump is **truncated to `[-1,1]^2` by rejection sampling** (no
coordinate clamping, so no boundary atoms); `alpha = 0` reduces exactly to `P`.
The `alpha = 0` rows **are the empirical size / Type-I check** (should bracket
`delta` within the 95% Wilson CI for the permutation tests). Per
`(method, n, alpha)`: `rejection_rate`, a 95% **Wilson CI**, and `avg_runtime`.

```sh
python experiments/run_power.py            # verification grid (~75 s)
python experiments/run_power.py --n 50 100 200 400 800 --B 300 --Z 300   # paper
```
CSV columns: `method, n, alpha, B, delta, Z, rejection_rate, ci_low, ci_high, avg_runtime`.
Flags: `--n --alpha --B --Z --delta --eps --seed --randomized`.

## Null calibration

`P = Q = Uniform([-1,1]^2)`, `Z` fresh redraws, recording the null dKS with
**both** `dks.exact` (the bounded quantity) and `dks.approx` (so its downward
bias is visible).

```sh
python experiments/run_calibration.py --n 2000 --Z 500      # verification (~26 s)
python experiments/run_calibration.py --n 10000 --Z 1000    # FINAL (one-time ~10-15 min, exact O(n^2))
```
CSV columns: `statistic {exact|approx}, value, n, Z`.

## Plots

```sh
python experiments/plots.py        # all four to figures/
```

| figure                   | x                 | y                                   |
|--------------------------|-------------------|-------------------------------------|
| `fig_runtime_vs_n`       | `n`               | avg runtime (log) — exact vs SSS    |
| `fig_power_vs_n`         | `n`               | rejection rate + 95% CI band        |
| `fig_power_vs_runtime` ★ | avg runtime (log) | rejection rate + 95% CI band        |
| `fig_calibration`        | `eps`             | tail prob (log) — both empiricals + both bounds |

★ headline (per Jeff): rejection probability vs runtime. Styling — exact-sample
= dashed+triangle, SSS-dKS = solid+circle, direct clean = dotted (faint), direct
union = dash-dot (faint); one color per `alpha`; CI bands shaded.

## Suggested FINAL config

```sh
python experiments/run_power.py --n 50 100 200 400 800 --B 300 --Z 300
python experiments/run_calibration.py --n 10000 --Z 1000
python experiments/plots.py
```
Power: `n` up to **400–800**, `B` ≈ **200–500**, `Z` ≥ **300** (tighter CIs, a
wider runtime axis where the `O(n^2)` vs `O(n log n)` gap is dramatic).
Calibration: `n = 10000`, `Z = 1000` (a one-time ~10–15 min exact run).

## Findings (verification grid: n∈{50,100,200,400}, α∈{0,.15,.3}, B=Z=100)

- Power rises with `alpha` and `n`; `alpha = 0` brackets `delta = 0.05` within
  the 95% Wilson CI for both permutation tests (empirical size OK).
- SSS-dKS reaches essentially the **same power** as exact-sample dKS at **far
  lower runtime** (`n = 400`: 1.00 @ ~7 ms vs ~180 ms — ~26x; the gap widens
  with `n`, matching `O(n log n)` vs `O(n^2)`).
- Direct/clean is **~1000x cheaper** (one approx eval) with emerging power
  (~0.60 at `n = 400`, `alpha = 0.3`); direct/union is far more conservative
  (≈0 power here) — consistent with the calibration picture below.
- Calibration: the **exact** empirical null tail sits **below the clean bound**
  in the operative tail (so `tau_clean` is empirically supported; the `union`
  bound is very loose). The **approx** tail sits slightly left of exact — a
  ~7% downward bias (mean exact − mean approx ≈ +0.003 at `n = 2000`).
