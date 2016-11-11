//
// Simple example of using pybind based on http://pybind11.readthedocs.io/en/latest/basics.html 
// but extended to use the Numpy-Eigen binding.
//

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <pybind11/eigen.h>

Eigen::MatrixXd add(const Eigen::MatrixXd &i, const Eigen::MatrixXd &j) 
{
    return i + j;
}

namespace py = pybind11;

PYBIND11_PLUGIN(bind_test) {
    py::module m("bind_test", "pybind11 example plugin");

    m.def("add", &add, "A function which adds two numbers");

    return m.ptr();
}
