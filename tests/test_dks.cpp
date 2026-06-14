// test_dks.cpp — correctness checks for the dKS core.
#include "dks/dks.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using dks::Point;

// O(n^3) brute-force reference: try every (x_i, y_j) corner from P u Q and
// recount from scratch. Deliberately naive; used only to validate exact().
static double brute(const std::vector<Point>& P, const std::vector<Point>& Q) {
    const std::size_t np = P.size(), nq = Q.size();
    if (np == 0 || nq == 0) return 0.0;
    std::vector<double> xs, ys;
    for (const auto& p : P) { xs.push_back(p.x); ys.push_back(p.y); }
    for (const auto& q : Q) { xs.push_back(q.x); ys.push_back(q.y); }
    double best = 0.0;
    for (double xt : xs) {
        for (double yt : ys) {
            long cp = 0, cq = 0;
            for (const auto& p : P) if (p.x <= xt && p.y <= yt) ++cp;
            for (const auto& q : Q) if (q.x <= xt && q.y <= yt) ++cq;
            double d = std::fabs(double(cp) / np - double(cq) / nq);
            if (d > best) best = d;
        }
    }
    return best;
}

static bool close(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol;
}

int main() {
    int failures = 0;

    // 1) Identical sets -> distance 0.
    {
        std::vector<Point> A = {{0.1, 0.2}, {0.5, 0.7}, {0.9, 0.3}};
        if (!close(dks::exact(A, A), 0.0)) { std::puts("FAIL identity exact"); ++failures; }
        if (!close(dks::approx(A, A), 0.0)) { std::puts("FAIL identity approx"); ++failures; }
    }

    // 2) Hand case: P at origin-ish, Q shifted past it -> max difference 1.
    {
        std::vector<Point> P = {{0.0, 0.0}, {0.1, 0.1}};
        std::vector<Point> Q = {{1.0, 1.0}, {1.1, 1.1}};
        // At z=(0.5,0.5): all of P dominated (1.0), none of Q (0.0) -> diff 1.
        if (!close(dks::exact(P, Q), 1.0)) { std::puts("FAIL shifted exact"); ++failures; }
    }

    // 3) exact() must match the brute-force reference on random instances,
    //    including unequal |P| and |Q|.
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int trial = 0; trial < 200; ++trial) {
            std::size_t np = 1 + rng() % 25;
            std::size_t nq = 1 + rng() % 25;
            std::vector<Point> P(np), Q(nq);
            for (auto& p : P) p = {U(rng), U(rng)};
            for (auto& q : Q) q = {U(rng), U(rng)};
            double e = dks::exact(P, Q);
            double b = brute(P, Q);
            if (!close(e, b)) {
                std::printf("FAIL exact vs brute: exact=%.15g brute=%.15g (np=%zu nq=%zu)\n",
                            e, b, np, nq);
                ++failures;
                break;
            }
        }
    }

    // 4) approx() should track exact() within eps on larger random instances.
    {
        std::mt19937 rng(99);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const double eps = 0.05;
        double worst = 0.0;
        for (int trial = 0; trial < 30; ++trial) {
            std::size_t n = 2000;
            std::vector<Point> P(n), Q(n);
            for (auto& p : P) p = {U(rng), U(rng)};
            for (auto& q : Q) q = {U(rng), U(rng)};
            double e = dks::exact(P, Q);
            double a = dks::approx(P, Q, eps);
            worst = std::max(worst, std::fabs(e - a));
        }
        std::printf("approx vs exact: worst |diff| over trials = %.4f (eps=%.2f)\n", worst, eps);
        if (worst > eps + 1e-9) { std::puts("FAIL approx exceeded eps"); ++failures; }
    }


    // 5) Ties: shared/duplicate points must still match the brute reference.
    {
        std::vector<Point> P = {{0.2,0.2},{0.2,0.2},{0.5,0.5},{0.8,0.1}};
        std::vector<Point> Q = {{0.2,0.2},{0.5,0.5},{0.5,0.5},{0.9,0.9}};
        if (!close(dks::exact(P,Q), brute(P,Q))) { std::puts("FAIL ties exact vs brute"); ++failures; }
    }

    if (failures == 0) std::puts("ALL TESTS PASSED");
    else std::printf("%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
