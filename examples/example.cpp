// example.cpp — minimal use of the dKS library.
//   c++ -O2 -std=c++17 -I../include example.cpp -o example && ./example
#include "dks/dks.hpp"
#include <cstdio>
#include <vector>

int main() {
    std::vector<dks::Point> P = {{0.1, 0.2}, {0.5, 0.7}, {0.9, 0.3}, {0.4, 0.4}};
    std::vector<dks::Point> Q = {{0.2, 0.2}, {0.6, 0.8}, {0.8, 0.2}, {0.5, 0.5}};

    std::printf("exact  dKS = %.6f\n", dks::exact(P, Q));
    std::printf("approx dKS = %.6f\n", dks::approx(P, Q));        // 2*sqrt(n) grid
    std::printf("approx dKS = %.6f  (eps = 0.05)\n", dks::approx(P, Q, 0.05));
    return 0;
}
