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

/* ===================== BASELINE KS ===================== */
double baselineKS(const vector<Point>& pts) {
    int n = pts.size();

    vector<Point> byY = pts;
    sort(byY.begin(), byY.end(),
         [](const Point& a, const Point& b) {
             return a.y < b.y;
         });

    vector<double> xs;
    xs.reserve(n);
    for (const auto& p : pts) xs.push_back(p.x);

    double M = 0.0;

    for (double x0 : xs) {
        int s = 0;
        for (const auto& p : byY) {
            if (p.x <= x0) {
                s += p.label;
                M = max(M, static_cast<double>(abs(s)));
            }
        }
    }

    return M / (n / 2.0);  // normalize
}

/* ===================== DATA GENERATORS ===================== */
vector<Point> makeUniform(int n, mt19937& gen) {
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<Point> pts;
    pts.reserve(2 * n);

    for (int i = 0; i < n; i++) {
        pts.push_back({dist(gen), dist(gen), +1});
        pts.push_back({dist(gen), dist(gen), -1});
    }
    return pts;
}

vector<Point> makeGaussian(int n, mt19937& gen) {
    normal_distribution<double> dist(0.0, 1.0);
    vector<Point> pts;
    pts.reserve(2 * n);

    for (int i = 0; i < n; i++) {
        pts.push_back({dist(gen), dist(gen), +1});
        pts.push_back({dist(gen), dist(gen), -1});
    }
    return pts;
}

/* ===================== MAIN ===================== */
int main() {
    vector<int> ns = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    // TRUE randomness: different every run
    random_device rd;
    mt19937 gen(rd());

    for (int n : ns) {
        auto U = makeUniform(n, gen);
        auto G = makeGaussian(n, gen);

        // Uniform
        auto t1 = chrono::high_resolution_clock::now();
        double errU = baselineKS(U);
        auto t2 = chrono::high_resolution_clock::now();
        double timeU = chrono::duration<double>(t2 - t1).count();

        // Gaussian
        t1 = chrono::high_resolution_clock::now();
        double errG = baselineKS(G);
        t2 = chrono::high_resolution_clock::now();
        double timeG = chrono::duration<double>(t2 - t1).count();

        cout << "n=" << n << "\n";
        cout << "  Uniform:  time=" << timeU << "  error=" << errU << "\n";
        cout << "  Gaussian: time=" << timeG << "  error=" << errG << "\n";
    }

    return 0;
}

