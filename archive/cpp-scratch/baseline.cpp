#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std;
using clk = chrono::high_resolution_clock;

/* --------------------------------------------------
   O(n^2) 2D KS-style baseline
   (same structure as Matheny-style baseline)
-------------------------------------------------- */
double baseline_2d(const vector<pair<double,double>>& A,
                   const vector<pair<double,double>>& B)
{
    int n = (int)A.size();
    double M = 0.0;

    for (int i = 0; i < n; i++) {
        double x = A[i].first;
        for (int j = 0; j < n; j++) {
            double y = A[j].second;

            int ca = 0, cb = 0;
            for (int k = 0; k < n; k++) {
                ca += (A[k].first <= x && A[k].second <= y);
                cb += (B[k].first <= x && B[k].second <= y);
            }

            double s = double(ca - cb) / n;
            M = std::max(M, std::abs(s));
        }
    }
    return M;
}

int main()
{
    vector<int> n_values = {
        256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536, 131072
    };

    mt19937_64 rng(0);
    uniform_real_distribution<double> U(0.0, 1.0);
    normal_distribution<double> G(0.0, 1.0);

    volatile double sink = 0.0; // prevents optimization

    for (int n : n_values) {

        vector<pair<double,double>> U1(n), U2(n);
        vector<pair<double,double>> G1(n), G2(n);

        for (int i = 0; i < n; i++) {
            U1[i] = {U(rng), U(rng)};
            U2[i] = {U(rng), U(rng)};
            G1[i] = {G(rng), G(rng)};
            G2[i] = {G(rng), G(rng)};
        }

        // --- Uniform first ---
        auto t0 = clk::now();
        sink += baseline_2d(U1, U2);
        auto t1 = clk::now();

        // --- Gaussian second ---
        auto t2 = clk::now();
        sink += baseline_2d(G1, G2);
        auto t3 = clk::now();

        double tU = chrono::duration<double>(t1 - t0).count();
        double tG = chrono::duration<double>(t3 - t2).count();

        cout << "n=" << n
             << "  Uniform=" << tU
             << "  Gaussian=" << tG
             << endl;
    }

    // make sure sink is used
    if (sink == 123456.789) cerr << sink << endl;

    return 0;
}

