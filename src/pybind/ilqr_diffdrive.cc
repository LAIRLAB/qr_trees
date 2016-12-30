//
// Simple example of using pybind based on http://pybind11.readthedocs.io/en/latest/basics.html 
// but extended to use the Numpy-Eigen binding.
//

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <experiments/single_obs_diffdrive.hh>

#include <array>
#include <tuple>

Eigen::MatrixXd add(const Eigen::MatrixXd &i, const Eigen::MatrixXd &j) 
{
    return i + j;
}

namespace py = pybind11;

PYBIND11_PLUGIN(ilqr_diffdrive) 
{
    py::module m("ilqr_diffdrive", "iLQR on DiffDrive dynamics.");

    py::enum_<PolicyTypes>(m, "PolicyTypes")
    .value("HINDSIGHT", PolicyTypes::HINDSIGHT)
    .value("TRUE_ILQR", PolicyTypes::TRUE_ILQR)
    .value("ARGMAX_ILQR", PolicyTypes::ARGMAX_ILQR)
    .value("PROB_WEIGHTED_CONTROL", PolicyTypes::PROB_WEIGHTED_CONTROL)
    .export_values() 
    .def("__str__", &to_string);

    m.def("control", [](PolicyTypes policy, bool has_obs, 
                std::array<double, 2> prior) 
            {
                std::string state_fname, obs_fname;
                const double cost_to_go
                    = control_diffdrive(policy, has_obs, prior,
                        state_fname, obs_fname);
                return std::make_tuple(cost_to_go, state_fname, obs_fname); 
            }, 
            "A function which adds two numbers");

    return m.ptr();
}
