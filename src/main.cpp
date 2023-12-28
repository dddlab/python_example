#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/SparseCore>

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

Eigen::SparseMatrix<double> accord(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diag,
    Eigen::Ref<Eigen::VectorXd> hist_norm_diff,
    Eigen::Ref<Eigen::VectorXd> hist_hn
    );

PYBIND11_MODULE(_gaccord, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: gaccord

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def(
        "accord",
        &accord,
        R"pbdoc(
            ACCORD, cov variant

            Column-major sparse matrix
        )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
