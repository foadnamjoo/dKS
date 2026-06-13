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

/* ===================== OUR ALGO ===================== */
double optimizedKS(const vector<Point>& P, const vector<Point>& Q) {
    int n = Q.size();               // P and Q have same size
    int grid_size = 2 * static_cast<int>(sqrt(n));

    // Grid counts
    vector<vector<int>> Cp(grid_size, vector<int>(grid_size, 0));
    vector<vector<int>> Cq(grid_size, vector<int>(grid_size, 0));

    // Combine coordinates
    vector<double> all_x, all_y;
    all_x.reserve(2 * n);
    all_y.reserve(2 * n);

    for (const auto& p : P) {
        all_x.push_back(p.x);
        all_y.push_back(p.y);
    }
    for (const auto& q : Q) {
        all_x.push_back(q.x);
        all_y.push_back(q.y);
    }

    sort(all_x.begin(), all_x.end());
    sort(all_y.begin(), all_y.end());

    int total = all_x.size();
    int step = total / (grid_size + 1);

    vector<double> selected_x, selected_y;
    for (int i = step; i < total && selected_x.size() < grid_size; i += step)
        selected_x.push_back(all_x[i]);
    for (int i = step; i < total && selected_y.size() < grid_size; i += step)
        selected_y.push_back(all_y[i]);

    auto increment = [&](const vector<Point>& pts,
                         vector<vector<int>>& C) {
        for (const auto& p : pts) {
            int i = upper_bound(selected_x.begin(), selected_x.end(), p.x)
                    - selected_x.begin() - 1;
            int j = upper_bound(selected_y.begin(), selected_y.end(), p.y)
                    - selected_y.begin() - 1;

            i = max(0, min(i, grid_size - 1));
            j = max(0, min(j, grid_size - 1));
            C[i][j]++;
        }
    };

    increment(P, Cp);
    increment(Q, Cq);

    // Prefix sums
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (i > 0) Cp[i][j] += Cp[i - 1][j];
            if (j > 0) Cp[i][j] += Cp[i][j - 1];
            if (i > 0 && j > 0) Cp[i][j] -= Cp[i - 1][j - 1];

            if (i > 0) Cq[i][j] += Cq[i - 1][j];
            if (j > 0) Cq[i][j] += Cq[i][j - 1];
            if (i > 0 && j > 0) Cq[i][j] -= Cq[i - 1][j - 1];
        }
    }

    int max_diff = 0;
    for (int i = 0; i < grid_size; i++)
        for (int j = 0; j < grid_size; j++)
            max_diff = max(max_diff, abs(Cp[i][j] - Cq[i][j]));

    return static_cast<double>(max_diff) / n;
}

/* ===================== DATA GENERATORS ===================== */
vector<Point> makeUniform(int n, mt19937& gen) {
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<Point> pts(n);
    for (int i = 0; i < n; i++)
        pts[i] = {dist(gen), dist(gen)};
    return pts;
}

vector<Point> makeGaussian(int n, mt19937& gen) {
    normal_distribution<double> dist(0.0, 1.0);
    vector<Point> pts(n);
    for (int i = 0; i < n; i++)
        pts[i] = {dist(gen), dist(gen)};
    return pts;
}

/* ===================== MAIN ===================== */
int main() {
    vector<int> ns = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

    random_device rd;
    mt19937 gen(rd());  // fresh randomness each run

    for (int n : ns) {
        auto Pu = makeUniform(n, gen);
        auto Qu = makeUniform(n, gen);

        auto Pg = makeGaussian(n, gen);
        auto Qg = makeGaussian(n, gen);

        auto t1 = chrono::high_resolution_clock::now();
        double errU = optimizedKS(Pu, Qu);
        auto t2 = chrono::high_resolution_clock::now();
        double timeU = chrono::duration<double>(t2 - t1).count();

        t1 = chrono::high_resolution_clock::now();
        double errG = optimizedKS(Pg, Qg);
        t2 = chrono::high_resolution_clock::now();
        double timeG = chrono::duration<double>(t2 - t1).count();

        cout << "n=" << n << "\n";
        cout << "  Uniform:  time=" << timeU << "  error=" << errU << "\n";
        cout << "  Gaussian: time=" << timeG << "  error=" << errG << "\n";
    }

    return 0;
}

