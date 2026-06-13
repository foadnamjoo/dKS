#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;

static inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

double ks2d(const vector<pair<double,double>>& A,
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
                if (A[k].first <= x && A[k].second <= y) ca++;
                if (B[k].first <= x && B[k].second <= y) cb++;
            }

            double s = (double)ca/n - (double)cb/n;
            M = std::max(M, std::abs(s));
        }
    }
    return M;
}

int main() {
    vector<int> n_values = {128,256,512,1024,2048,4096}; 
    // DO NOT go bigger with this cubic baseline. 8192 will be brutal.

    std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> U(0.0,1.0);
    std::normal_distribution<double> G(0.0,1.0);

    volatile double sink = 0.0; // prevents optimization

    for (int n : n_values) {
        vector<pair<double,double>> U1(n), U2(n), G1(n), G2(n);

        for (int i=0;i<n;i++){
            U1[i]={U(rng),U(rng)};
            U2[i]={U(rng),U(rng)};
            G1[i]={G(rng),G(rng)};
            G2[i]={G(rng),G(rng)};
        }

        double t0 = now_sec();
        double uval = ks2d(U1, U2);
        double t1 = now_sec();

        double t2 = now_sec();
        double gval = ks2d(G1, G2);
        double t3 = now_sec();

        sink += uval + gval;

        cout << "n=" << n
             << "  Uniform(time=" << (t1-t0) << ", val=" << uval << ")"
             << "  Gaussian(time=" << (t3-t2) << ", val=" << gval << ")"
             << "\n";
    }

    // print sink so compiler can't ignore it
    cerr << "sink=" << sink << "\n";
}

