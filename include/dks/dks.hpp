// dks.hpp — Multi-dimensional Kolmogorov–Smirnov distance (d = 2)
//
// Header-only. Drop it in and `#include <dks/dks.hpp>`.
//
// Implements the dominating-rectangle ("lower-orthant") Kolmogorov–Smirnov
// distance between two 2-D point sets P and Q:
//
//     dKS(P, Q) = max_z | |{p in P : p <= z}| / |P|
//                        - |{q in Q : q <= z}| / |Q| |
//
// where p <= z means p.x <= z.x AND p.y <= z.y.
//
//   - dks::exact(P, Q)        O(n^2)        the maximizing z need not be a data
//                                           point, but each of its coordinates is
//                                           realized by some point's coordinate,
//                                           so thresholds are drawn from P u Q.
//   - dks::approx(P, Q, eps)  O(n log n)    grid approximation; |result - exact|
//                                           <= eps. With eps <= 0 it uses a grid
//                                           of side 2*sqrt(n), matching the
//                                           1/sqrt(n) sampling floor.
//
// Counts are normalized by each set's own size, so |P| != |Q| is handled
// correctly (it reduces to dividing by n when the sizes are equal).
//
// Reference: Jacobs, Namjoo, Phillips, "Efficient and Stable Multi-Dimensional
// Kolmogorov-Smirnov Distance."

#ifndef DKS_DKS_HPP
#define DKS_DKS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace dks {

struct Point {
    double x;
    double y;
};

namespace detail {

// 2-D inclusive prefix sum (summed-area table) over a grid of integer counts.
inline void prefix_sum_2d(std::vector<std::vector<int>>& C) {
    const std::size_t rows = C.size();
    if (rows == 0) return;
    const std::size_t cols = C[0].size();
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const int up   = (i > 0)            ? C[i - 1][j]     : 0;
            const int left = (j > 0)            ? C[i][j - 1]     : 0;
            const int diag = (i > 0 && j > 0)   ? C[i - 1][j - 1] : 0;
            C[i][j] += up + left - diag;
        }
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Exact dKS in O(n^2).
//
// Fix each distinct x-threshold from P u Q; sweep points in y-sorted order,
// counting those whose x is within the threshold, and track the largest
// normalized count difference. This is the d=2 "Baseline" of the paper.
// ---------------------------------------------------------------------------
inline double exact(const std::vector<Point>& P, const std::vector<Point>& Q) {
    const std::size_t np = P.size();
    const std::size_t nq = Q.size();
    if (np == 0 || nq == 0) return 0.0;

    // Combined points tagged by membership (true = P), sorted by y ascending.
    struct Tagged { double x, y; bool in_p; };
    std::vector<Tagged> pts;
    pts.reserve(np + nq);
    for (const auto& p : P) pts.push_back({p.x, p.y, true});
    for (const auto& q : Q) pts.push_back({q.x, q.y, false});
    std::sort(pts.begin(), pts.end(),
              [](const Tagged& a, const Tagged& b) { return a.y < b.y; });

    // Distinct x-thresholds from P u Q.
    std::vector<double> xthr;
    xthr.reserve(pts.size());
    for (const auto& t : pts) xthr.push_back(t.x);
    std::sort(xthr.begin(), xthr.end());
    xthr.erase(std::unique(xthr.begin(), xthr.end()), xthr.end());

    const double inv_np = 1.0 / static_cast<double>(np);
    const double inv_nq = 1.0 / static_cast<double>(nq);

    const std::size_t m = pts.size();
    double best = 0.0;
    for (const double xt : xthr) {
        long cp = 0, cq = 0;
        std::size_t k = 0;
        while (k < m) {
            // Add every point sharing this y-level (within the x-threshold)
            // before evaluating, so tied coordinates don't create a transient
            // imbalance that doesn't correspond to any real threshold z.
            const double yt = pts[k].y;
            std::size_t k2 = k;
            while (k2 < m && pts[k2].y == yt) {
                if (pts[k2].x <= xt) { if (pts[k2].in_p) ++cp; else ++cq; }
                ++k2;
            }
            const double diff =
                std::fabs(static_cast<double>(cp) * inv_np -
                          static_cast<double>(cq) * inv_nq);
            if (diff > best) best = diff;
            k = k2;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Approximate dKS in O(n log n) via a stratified grid.
//
// `eps` controls resolution: eps > 0 uses a grid of side ceil(2/eps) so the
// answer is within eps of exact; eps <= 0 uses side 2*sqrt(n), the resolution
// at which exact computation stops being meaningful (sampling error ~ 1/sqrt(n)).
// ---------------------------------------------------------------------------
inline double approx(const std::vector<Point>& P, const std::vector<Point>& Q,
                     double eps = -1.0) {
    const std::size_t np = P.size();
    const std::size_t nq = Q.size();
    if (np == 0 || nq == 0) return 0.0;

    const std::size_t n = std::max(np, nq);
    int grid = (eps > 0.0)
                   ? static_cast<int>(std::ceil(2.0 / eps))
                   : static_cast<int>(2.0 * std::sqrt(static_cast<double>(n)));
    if (grid < 1) grid = 1;

    // Grid boundaries: evenly spaced by rank in the sorted combined coords.
    std::vector<double> xs, ys;
    xs.reserve(np + nq);
    ys.reserve(np + nq);
    for (const auto& p : P) { xs.push_back(p.x); ys.push_back(p.y); }
    for (const auto& q : Q) { xs.push_back(q.x); ys.push_back(q.y); }
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());

    const std::size_t total = xs.size();
    // Grid lines at evenly-spaced RANKS across the FULL range [0, total-1], so the last
    // line is always the maximum coordinate (x_k = max, per Algorithm 2). The previous
    // construction stepped from the smallest value and stopped short of the maximum,
    // leaving the top slice with no grid line and biasing the estimate low.
    const int lines = std::max(1, std::min(grid, static_cast<int>(total)));

    std::vector<double> bx, by;
    bx.reserve(lines);
    by.reserve(lines);
    for (int k = 0; k < lines; ++k) {
        const std::size_t idx = (lines <= 1)
            ? total - 1
            : static_cast<std::size_t>(
                  static_cast<double>(k) * static_cast<double>(total - 1) / (lines - 1) + 0.5);
        bx.push_back(xs[idx]);
        by.push_back(ys[idx]);
    }

    const int gx = static_cast<int>(bx.size());
    const int gy = static_cast<int>(by.size());

    std::vector<std::vector<int>> Cp(gx, std::vector<int>(gy, 0));
    std::vector<std::vector<int>> Cq(gx, std::vector<int>(gy, 0));

    auto bin = [&](const std::vector<Point>& pts, std::vector<std::vector<int>>& C) {
        for (const auto& p : pts) {
            int i = static_cast<int>(
                        std::upper_bound(bx.begin(), bx.end(), p.x) - bx.begin()) - 1;
            int j = static_cast<int>(
                        std::upper_bound(by.begin(), by.end(), p.y) - by.begin()) - 1;
            i = std::min(std::max(i, 0), gx - 1);
            j = std::min(std::max(j, 0), gy - 1);
            ++C[i][j];
        }
    };
    bin(P, Cp);
    bin(Q, Cq);

    detail::prefix_sum_2d(Cp);
    detail::prefix_sum_2d(Cq);

    const double inv_np = 1.0 / static_cast<double>(np);
    const double inv_nq = 1.0 / static_cast<double>(nq);

    double best = 0.0;
    for (int i = 0; i < gx; ++i) {
        for (int j = 0; j < gy; ++j) {
            const double diff = std::fabs(static_cast<double>(Cp[i][j]) * inv_np -
                                          static_cast<double>(Cq[i][j]) * inv_nq);
            if (diff > best) best = diff;
        }
    }
    return best;
}

}  // namespace dks

#endif  // DKS_DKS_HPP
