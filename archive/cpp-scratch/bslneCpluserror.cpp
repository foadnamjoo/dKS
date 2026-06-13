#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;

struct Point {
    double x, y;
    int label;   // +1 for P, -1 for Q
};

/* ---------------- Baseline KS ---------------- */

double baselineKS(const vector<Point>& pts) {
    int n = pts.size();

    vector<Point> byY = pts;
    sort(byY.begin(), byY.end(),
         [](const Point& a, const Point& b){ return a.y < b.y; });

    vector<double> xs;
    xs.reserve(n);
    for (auto &p : pts) xs.push_back(p.x);

    double M = 0.0;

    for (double x0 : xs) {
        int s = 0;
        for (auto &p : byY) {
            if (p.x <= x0) {
                s += p.label;
                M = max(M, (double)abs(s));
            }
        }
    }
    return M / (n / 2.0);
}

/* ---------------- Data generation (FIXED) ---------------- */

vector<Point> makeUniform(int n, uint32_t seedP, uint32_t seedQ) {
    mt19937 genP(seedP), genQ(seedQ);
    uniform_real_distribution<> dist(0.0, 1.0);

    vector<Point> pts;
    pts.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        pts.push_back({dist(genP), dist(genP), +1});
        pts.push_back({dist(genQ), dist(genQ), -1});
    }
    return pts;
}

vector<Point> makeGaussian(int n, uint32_t seedP, uint32_t seedQ) {
    mt19937 genP(seedP), genQ(seedQ);
    normal_distribution<> dist(0.0, 1.0);

    vector<Point> pts;
    pts.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        pts.push_back({dist(genP), dist(genP), +1});
        pts.push_back({dist(genQ), dist(genQ), -1});
    }
    return pts;
}

/* ---------------- Main ---------------- */

int main() {
    vector<int> ns = {256, 512, 1024, 2048, 4096, 8192,
                      16384, 32768};

    for (int n : ns) {

        // independent P and Q (but reproducible)
        auto U = makeUniform(n, 123u, 456u);
        auto G = makeGaussian(n, 123u, 456u);

        auto t1 = chrono::high_resolution_clock::now();
        double errU = baselineKS(U);
        auto t2 = chrono::high_resolution_clock::now();
        double tu = chrono::duration<double>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        double errG = baselineKS(G);
        t2 = chrono::high_resolution_clock::now();
        double tg = chrono::duration<double>(t2 - t1).count();

        cout << "n=" << n << "\n";
        cout << "  Uniform:  time=" << tu << "  error=" << errU << "\n";
        cout << "  Gaussian: time=" << tg << "  error=" << errG << "\n";
    }
    return 0;
}

