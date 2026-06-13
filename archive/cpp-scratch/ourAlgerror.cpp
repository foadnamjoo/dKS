#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;

struct Point {
    double x, y;
};

/* ---------------- Data generation (FIXED: seed passed in) ---------------- */

vector<Point> makeUniform(int n, uint32_t seed) {
    mt19937 gen(seed);
    uniform_real_distribution<> dist(0.0, 1.0);
    vector<Point> pts;
    pts.reserve(n);
    for (int i = 0; i < n; i++) pts.push_back({dist(gen), dist(gen)});
    return pts;
}

vector<Point> makeGaussian(int n, uint32_t seed) {
    mt19937 gen(seed);
    normal_distribution<> dist(0.0, 1.0);
    vector<Point> pts;
    pts.reserve(n);
    for (int i = 0; i < n; i++) pts.push_back({dist(gen), dist(gen)});
    return pts;
}

/* ---------------- Our Algo (matching your Python structure) ---------------- */

double our_algo_error(const vector<Point>& P, const vector<Point>& Q) {
    int n = (int)Q.size();
    int grid_size = 2 * (int)std::sqrt((double)n);
    if (grid_size < 1) grid_size = 1;

    vector<vector<int>> counts_p(grid_size, vector<int>(grid_size, 0));
    vector<vector<int>> counts_q(grid_size, vector<int>(grid_size, 0));

    // Collect combined coords for quantile grid boundaries
    vector<double> xs, ys;
    xs.reserve(2 * n);
    ys.reserve(2 * n);
    for (auto &p : P) { xs.push_back(p.x); ys.push_back(p.y); }
    for (auto &q : Q) { xs.push_back(q.x); ys.push_back(q.y); }

    sort(xs.begin(), xs.end());
    sort(ys.begin(), ys.end());

    int num_points = (int)xs.size();
    int step = num_points / (grid_size + 1);
    if (step < 1) step = 1;

    vector<double> selected_x, selected_y;
    selected_x.reserve(grid_size + 2);
    selected_y.reserve(grid_size + 2);

    for (int i = 0; i < num_points && (int)selected_x.size() < grid_size; i += step)
        selected_x.push_back(xs[i]);
    for (int i = 0; i < num_points && (int)selected_y.size() < grid_size; i += step)
        selected_y.push_back(ys[i]);

    // Safety: if something weird happens, ensure non-empty
    if (selected_x.empty()) selected_x.push_back(xs[0]);
    if (selected_y.empty()) selected_y.push_back(ys[0]);

    auto increment_counts = [&](const vector<Point>& pts, vector<vector<int>>& counts) {
        for (auto &p : pts) {
            int i = (int)(upper_bound(selected_x.begin(), selected_x.end(), p.x) - selected_x.begin()) - 1;
            int j = (int)(upper_bound(selected_y.begin(), selected_y.end(), p.y) - selected_y.begin()) - 1;
            i = min(max(i, 0), grid_size - 1);
            j = min(max(j, 0), grid_size - 1);
            counts[i][j] += 1;
        }
    };

    increment_counts(P, counts_p);
    increment_counts(Q, counts_q);

    auto cumulative_2d = [&](const vector<vector<int>>& counts) {
        vector<vector<int>> C(grid_size, vector<int>(grid_size, 0));
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int a = counts[i][j];
                int up = (i > 0) ? C[i-1][j] : 0;
                int left = (j > 0) ? C[i][j-1] : 0;
                int diag = (i > 0 && j > 0) ? C[i-1][j-1] : 0;
                C[i][j] = up + left - diag + a;
            }
        }
        return C;
    };

    auto Cp = cumulative_2d(counts_p);
    auto Cq = cumulative_2d(counts_q);

    int max_diff = 0;
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int d = abs(Cp[i][j] - Cq[i][j]);
            if (d > max_diff) max_diff = d;
        }
    }

    return (double)max_diff / (double)n;
}

/* ---------------- Main: runtime + error for Uniform and Gaussian ---------------- */

int main() {
    vector<int> ns = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};

    for (int n : ns) {
        // Different seeds => P and Q are independent but reproducible
        auto Pu = makeUniform(n, 123u);
        auto Qu = makeUniform(n, 456u);

        auto Pg = makeGaussian(n, 123u);
        auto Qg = makeGaussian(n, 456u);

        auto t1 = chrono::high_resolution_clock::now();
        double err_u = our_algo_error(Pu, Qu);
        auto t2 = chrono::high_resolution_clock::now();
        double tu = chrono::duration<double>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        double err_g = our_algo_error(Pg, Qg);
        t2 = chrono::high_resolution_clock::now();
        double tg = chrono::duration<double>(t2 - t1).count();

        cout << "n=" << n << "\n";
        cout << "  Uniform:  time=" << tu << "  error=" << err_u << "\n";
        cout << "  Gaussian: time=" << tg << "  error=" << err_g << "\n";
    }
    return 0;
}

