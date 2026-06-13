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

/* ---------- Data generation ---------- */

vector<Point> makeUniform(int n) {
    mt19937 gen(123);
    uniform_real_distribution<> dist(0.0, 1.0);
    vector<Point> pts;
    pts.reserve(n);
    for (int i = 0; i < n; i++)
        pts.push_back({dist(gen), dist(gen)});
    return pts;
}

vector<Point> makeGaussian(int n) {
    mt19937 gen(123);
    normal_distribution<> dist(0.0, 1.0);
    vector<Point> pts;
    pts.reserve(n);
    for (int i = 0; i < n; i++)
        pts.push_back({dist(gen), dist(gen)});
    return pts;
}

/* ---------- Our Algorithm ---------- */

double our_algo(const vector<Point>& P, const vector<Point>& Q) {
    int n = Q.size();
    int grid_size = 2 * (int)std::sqrt(n);

    vector<vector<int>> Cp(grid_size, vector<int>(grid_size, 0));
    vector<vector<int>> Cq(grid_size, vector<int>(grid_size, 0));

    vector<double> xs, ys;
    xs.reserve(2*n);
    ys.reserve(2*n);

    for (auto& p : P) { xs.push_back(p.x); ys.push_back(p.y); }
    for (auto& p : Q) { xs.push_back(p.x); ys.push_back(p.y); }

    sort(xs.begin(), xs.end());
    sort(ys.begin(), ys.end());

    int N = xs.size();
    int step = max(1, N / (grid_size + 1));

    vector<double> sel_x, sel_y;
    for (int i = 0; i < N && (int)sel_x.size() < grid_size; i += step)
        sel_x.push_back(xs[i]);
    for (int i = 0; i < N && (int)sel_y.size() < grid_size; i += step)
        sel_y.push_back(ys[i]);

    auto add_points = [&](const vector<Point>& pts,
                          vector<vector<int>>& C) {
        for (auto& p : pts) {
            int i = int(upper_bound(sel_x.begin(), sel_x.end(), p.x) - sel_x.begin()) - 1;
            int j = int(upper_bound(sel_y.begin(), sel_y.end(), p.y) - sel_y.begin()) - 1;
            i = min(max(i, 0), grid_size - 1);
            j = min(max(j, 0), grid_size - 1);
            C[i][j]++;
        }
    };

    add_points(P, Cp);
    add_points(Q, Cq);

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (i > 0) Cp[i][j] += Cp[i-1][j];
            if (j > 0) Cp[i][j] += Cp[i][j-1];
            if (i > 0 && j > 0) Cp[i][j] -= Cp[i-1][j-1];
        }
    }

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (i > 0) Cq[i][j] += Cq[i-1][j];
            if (j > 0) Cq[i][j] += Cq[i][j-1];
            if (i > 0 && j > 0) Cq[i][j] -= Cq[i-1][j-1];
        }
    }

    int max_diff = 0;
    for (int i = 0; i < grid_size; i++)
        for (int j = 0; j < grid_size; j++)
            max_diff = max(max_diff, abs(Cp[i][j] - Cq[i][j]));

    return double(max_diff) / double(n);
}

/* ---------- Main ---------- */

int main() {
    vector<int> ns = {
        256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536, 131072
    };

    for (int n : ns) {

        auto Pu = makeUniform(n);
        auto Qu = makeUniform(n);

        auto Pg = makeGaussian(n);
        auto Qg = makeGaussian(n);

        auto t1 = chrono::high_resolution_clock::now();
        our_algo(Pu, Qu);
        auto t2 = chrono::high_resolution_clock::now();
        double tu = chrono::duration<double>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        our_algo(Pg, Qg);
        t2 = chrono::high_resolution_clock::now();
        double tg = chrono::duration<double>(t2 - t1).count();

        cout << "n=" << n
             << "  Uniform=" << tu
             << "  Gaussian=" << tg << endl;
    }
}

