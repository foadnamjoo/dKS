# dKS — Multi-Dimensional Kolmogorov–Smirnov Distance

A small, fast C++ library for the **dominating-rectangle Kolmogorov–Smirnov
distance** between two-dimensional point sets, plus a command-line tool. It is
the reference implementation for the paper *"Efficient and Stable
Multi-Dimensional Kolmogorov–Smirnov Distance"* (Jacobs, Namjoo, Phillips).

The distance between point sets `P` and `Q` is

```
dKS(P, Q) = max over z of | #{p in P : p <= z} / |P|  -  #{q in Q : q <= z} / |Q| |
```

where `p <= z` means `p.x <= z.x` and `p.y <= z.y`. Unlike Euclidean-based
distances (MMD, Wasserstein), it is invariant to the choice of units on each
axis, so it behaves sensibly when coordinates have different units (e.g. height
vs. weight, temperature vs. pressure).

## Two algorithms (matching the paper)

The library computes the same distance two ways; the names follow the paper's
Algorithm 1 and Algorithm 2:

| Function | Paper name | Cost | What it does |
|---|---|---|---|
| `dks::exact(P, Q)`       | **dKS-Baseline** (Alg. 1) | `O(n^2)`     | computes `dKS(P, Q)` exactly *for the given point sets*, sweeping every distinct dominating rectangle (thresholds in `P ∪ Q`) |
| `dks::approx(P, Q, eps)` | **dKS-Sketch** (Alg. 2)   | `O(n log n)` | within `eps` of dKS-Baseline, via a grid; `eps <= 0` uses a `2*sqrt(n)` grid — the resolution past which finer computation stops being statistically meaningful (sampling error `~ 1/sqrt(n)`) |

`dks::exact` is **exact for the point sets `P, Q`** (no grid approximation) — it
is *not* a population-exact distance: like any sample-based estimate it still
carries `~ 1/sqrt(n)` error relative to the underlying distributions. dKS-Sketch
matches dKS-Baseline to within that same order, so at realistic sample sizes the
two agree. Counts are normalized by each set's own size, so `|P| != |Q|` is
handled correctly.

## Quickstart

The core is **header-only** — copy `include/dks/dks.hpp` into your project and:

```cpp
#include <dks/dks.hpp>

std::vector<dks::Point> P = {{0.1, 0.2}, {0.5, 0.7}};
std::vector<dks::Point> Q = {{0.2, 0.2}, {0.6, 0.8}};

double d  = dks::approx(P, Q);        // dKS-Sketch: fast, default grid
double de = dks::exact(P, Q);         // dKS-Baseline: brute force, exact for P,Q
double da = dks::approx(P, Q, 0.05);  // dKS-Sketch within 0.05 of dKS-Baseline
```

### Build the CLI and tests

With Make:

```sh
make            # builds ./dks
make test       # builds and runs the correctness tests
```

or with CMake:

```sh
cmake -B build && cmake --build build
ctest --test-dir build
```

### Command-line tool

```sh
./dks P.txt Q.txt            # dKS-Sketch (fast, default)
./dks P.txt Q.txt --exact    # dKS-Baseline (brute force)
```

Each file holds one point per line, `x y` or `x,y` (blank lines and `#`
comments ignored). A run prints one value (pass `--approx --exact` for both):

```
|P| = 4, |Q| = 4
dKS (approx) = 0.250000
```

## Python

Python bindings (via pybind11) expose the same two functions on NumPy arrays. From the repo root:

```sh
pip install .
```

Then:

```python
import numpy as np
import dks

P = np.array([[0.1, 0.2], [0.5, 0.7]])
Q = np.array([[0.2, 0.2], [0.6, 0.8]])

dks.approx(P, Q)        # dKS-Sketch: fast O(n log n), default 2*sqrt(n) grid
dks.exact(P, Q)         # dKS-Baseline: brute force, O(n^2)
dks.approx(P, Q, 0.05)  # dKS-Sketch within 0.05 of dKS-Baseline
```

Inputs are (N, 2) float arrays; the two sets may differ in size. Requires NumPy and a C++ compiler.

## Layout

```
include/dks/dks.hpp   header-only core (dKS-Baseline + dKS-Sketch)
cli/dks_cli.cpp       command-line tool
examples/example.cpp  minimal library usage
tests/test_dks.cpp    correctness tests (vs. brute-force reference)
CMakeLists.txt        CMake build
Makefile              plain-make build
```

## Correctness

`tests/test_dks.cpp` checks `dks::exact` (dKS-Baseline) against an independent
`O(n^3)` brute-force reference on random instances (including unequal sizes and
tied / duplicate points), verifies that `dks::approx` (dKS-Sketch) stays within
`eps` of dKS-Baseline, and covers identity and fully-separated edge cases.

## Roadmap

- Higher dimensions (`d = 3, 4`) following the paper's Klee's-measure reduction.
- Two-sample testing utilities (calibrated thresholds; permutation baseline).

## License

TBD.
