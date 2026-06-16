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

## Why two algorithms

| Function | Cost | Use |
|---|---|---|
| `dks::exact(P, Q)`        | `O(n^2)`     | exact value; the maximizing corner is found over all thresholds in `P ∪ Q` |
| `dks::approx(P, Q, eps)`  | `O(n log n)` | within `eps` of exact; with `eps <= 0` uses a `2*sqrt(n)` grid, the resolution at which exact computation stops being statistically meaningful (sampling error `~ 1/sqrt(n)`) |

Counts are normalized by each set's own size, so `|P| != |Q|` is handled
correctly.

## Quickstart

The core is **header-only** — copy `include/dks/dks.hpp` into your project and:

```cpp
#include <dks/dks.hpp>

std::vector<dks::Point> P = {{0.1, 0.2}, {0.5, 0.7}};
std::vector<dks::Point> Q = {{0.2, 0.2}, {0.6, 0.8}};

double d  = dks::approx(P, Q);        // fast, default resolution
double de = dks::exact(P, Q);         // exact
double da = dks::approx(P, Q, 0.05);  // within 0.05 of exact
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
./dks P.txt Q.txt --exact
```

Each file holds one point per line, `x y` or `x,y` (blank lines and `#`
comments ignored):

```
|P| = 4, |Q| = 4
dKS (approx) = 0.250000
dKS (exact)  = 0.250000
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

dks.approx(P, Q)        # fast O(n log n), default 2*sqrt(n) grid
dks.exact(P, Q)         # exact O(n^2)
dks.approx(P, Q, 0.05)  # within 0.05 of exact
```

Inputs are (N, 2) float arrays; the two sets may differ in size. Requires NumPy and a C++ compiler.

## Layout

```
include/dks/dks.hpp   header-only core (exact + approx)
cli/dks_cli.cpp       command-line tool
examples/example.cpp  minimal library usage
tests/test_dks.cpp    correctness tests (vs. brute-force reference)
CMakeLists.txt        CMake build
Makefile              plain-make build
```

## Correctness

`tests/test_dks.cpp` checks `exact()` against an independent `O(n^3)`
brute-force reference on random instances (including unequal sizes and tied /
duplicate points), verifies that `approx()` stays within `eps` of `exact()`,
and covers identity and fully-separated edge cases.

## Roadmap

- Higher dimensions (`d = 3, 4`) following the paper's Klee's-measure reduction.
- Two-sample testing utilities (calibrated thresholds; permutation baseline).

## License

TBD.
