// dks_py.cpp — pybind11 binding for the dKS core.
//
// Exposes two functions to Python, both taking (N, 2) NumPy arrays:
//     dks.exact(P, Q)            -> float   (O(n^2))
//     dks.approx(P, Q, eps=-1.0) -> float   (O(n log n), within eps of exact)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdexcept>
#include <vector>

#include "dks/dks.hpp"

namespace py = pybind11;

namespace {

std::vector<dks::Point> to_points(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
    const auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 2)
        throw std::invalid_argument("expected an (N, 2) array of points");
    const double* data = static_cast<const double*>(buf.ptr);
    const std::size_t n = static_cast<std::size_t>(buf.shape[0]);
    std::vector<dks::Point> pts(n);
    for (std::size_t i = 0; i < n; ++i)
        pts[i] = {data[2 * i], data[2 * i + 1]};
    return pts;
}

double exact_py(const py::array_t<double>& P, const py::array_t<double>& Q) {
    const auto p = to_points(P);
    const auto q = to_points(Q);
    py::gil_scoped_release release;  // pure C++; let other threads run
    return dks::exact(p, q);
}

double approx_py(const py::array_t<double>& P, const py::array_t<double>& Q,
                 double eps) {
    const auto p = to_points(P);
    const auto q = to_points(Q);
    py::gil_scoped_release release;
    return dks::approx(p, q, eps);
}

}  // namespace

PYBIND11_MODULE(dks, m) {
    m.doc() =
        "Multi-dimensional Kolmogorov-Smirnov distance (d = 2).\n"
        "P and Q are (N, 2) NumPy arrays of points.";
    m.def("exact", &exact_py, py::arg("P"), py::arg("Q"),
          "Exact dKS in O(n^2).");
    m.def("approx", &approx_py, py::arg("P"), py::arg("Q"), py::arg("eps") = -1.0,
          "Approximate dKS in O(n log n); within eps of exact. "
          "eps <= 0 uses a 2*sqrt(n) grid.");
    m.attr("__version__") = "0.1.0";
}
