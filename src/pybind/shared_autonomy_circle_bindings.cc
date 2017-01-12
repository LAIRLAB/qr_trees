//
// Simple example of using pybind based on http://pybind11.readthedocs.io/en/latest/basics.html 
// but extended to use the Numpy-Eigen binding.
//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// December 2016
//

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <experiments/shared_autonomy_circle.hh>

#include <experiments/simulators/circle_world.hh>
#include <pybind/circle_world_bindings.hh>

#include <array>
#include <sstream>
#include <tuple>


namespace py = pybind11;

PYBIND11_PLUGIN(shared_autonomy_circle_bindings) 
{
    //using Circle = circle_world::Circle;
    //using CircleWorld = circle_world::CircleWorld;

    py::module m("shared_autonomy_circle_bindings", "iLQR on shared autonomy circle world.");


    py::enum_<PolicyTypes>(m, "PolicyTypes")
    .value("HINDSIGHT", PolicyTypes::HINDSIGHT)
    .value("TRUE_ILQR", PolicyTypes::TRUE_ILQR)
    .value("ARGMAX_ILQR", PolicyTypes::ARGMAX_ILQR)
    .value("PROB_WEIGHTED_CONTROL", PolicyTypes::PROB_WEIGHTED_CONTROL)
    .export_values() 
    .def("__str__", &to_string);

    m.def("control_shared_autonomy", [](PolicyTypes policy, 
                CircleWorld &true_world, 
                CircleWorld &other_world, 
                std::array<double, 2> prior,
                std::string state_fname = "",
                std::string obs_fname = ""
                ) 
            {
                const double cost_to_go
                    = control_shared_autonomy(policy, true_world, 
                        other_world, prior,
                        state_fname, obs_fname);
                return std::make_tuple(cost_to_go, state_fname, obs_fname); 
            }, 
            "Function for running iLQR on shared autonomy circle world.");
    
    add_circle_class(m);
    add_circle_world_class(m);


    return m.ptr();
}
