// dks_cli.cpp — command-line front end for the dKS core.
//
// Usage:
//   dks <fileP> <fileQ> [--eps E] [--exact] [--approx]
//
// Each input file holds one point per line as "x y" or "x,y".
// Blank lines and lines beginning with '#' are ignored.
//
// Default output is the O(n log n) approximation. Pass --exact to also
// compute the O(n^2) exact value, or --eps E to set the approximation
// resolution (answer is within E of exact).

#include "dks/dks.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<dks::Point> read_points(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "dks: cannot open '%s'\n", path.c_str());
        std::exit(2);
    }
    std::vector<dks::Point> pts;
    std::string line;
    while (std::getline(in, line)) {
        // strip a trailing comment and surrounding blanks
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream ss(line);
        double x, y;
        if (ss >> x >> y) pts.push_back({x, y});
    }
    return pts;
}

void usage() {
    std::fprintf(stderr,
        "usage: dks <fileP> <fileQ> [--eps E] [--exact] [--approx]\n"
        "  files: one point per line, \"x y\" or \"x,y\"\n"
        "  --eps E   approximation resolution (default: 2*sqrt(n) grid)\n"
        "  --exact   also compute the O(n^2) exact distance\n"
        "  --approx  compute the approximation (default if neither given)\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> files;
    double eps = -1.0;
    bool want_exact = false, want_approx = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--exact") want_exact = true;
        else if (a == "--approx") want_approx = true;
        else if (a == "--eps") {
            if (i + 1 >= argc) { usage(); return 2; }
            eps = std::atof(argv[++i]);
        } else if (a == "-h" || a == "--help") { usage(); return 0; }
        else files.push_back(a);
    }
    if (files.size() != 2) { usage(); return 2; }
    if (!want_exact && !want_approx) want_approx = true;

    const auto P = read_points(files[0]);
    const auto Q = read_points(files[1]);
    std::printf("|P| = %zu, |Q| = %zu\n", P.size(), Q.size());

    if (want_approx) std::printf("dKS (approx) = %.6f\n", dks::approx(P, Q, eps));
    if (want_exact)  std::printf("dKS (exact)  = %.6f\n", dks::exact(P, Q));
    return 0;
}
