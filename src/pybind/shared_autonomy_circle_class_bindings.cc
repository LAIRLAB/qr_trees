//
// Simple example of using pybind based on http://pybind11.readthedocs.io/en/latest/basics.html 
// but extended to use the Numpy-Eigen binding.
//
// Shervin Javdani (sjavdani@cs.cmu.edu)
// January 2017
//

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <experiments/shared_autonomy_circle_class.hh>

#include <experiments/simulators/circle_world.hh>
#include <pybind/circle_world_bindings.hh>

#include <array>
#include <sstream>
#include <tuple>


namespace py = pybind11;

//template< typename T >
//inline
//std::vector< T > to_std_vector( const py::object& iterable )
//{
//    return std::vector< T >( py::[](const simulators::directdrive::StateVector &s)
//            py::stl_input_iterator< T >ve( iterable ),
//                             py::stl_input_iterator< T >( ) );
//}

using namespace experiments;

PYBIND11_PLUGIN(shared_autonomy_circle_class_bindings) 
{

    py::module m("shared_autonomy_circle_class_bindings", "iLQR on shared autonomy circle world.");

    add_circle_class(m);
    add_circle_world_class(m);

    py::enum_<PolicyTypes>(m, "PolicyTypes")
    .value("HINDSIGHT", PolicyTypes::HINDSIGHT)
    .value("TRUE_ILQR", PolicyTypes::TRUE_ILQR)
    .value("ARGMAX_ILQR", PolicyTypes::ARGMAX_ILQR)
    .value("PROB_WEIGHTED_CONTROL", PolicyTypes::PROB_WEIGHTED_CONTROL)
    .export_values() 
    .def("__str__", &to_string);


    py::class_<SharedAutonomyCircle>(m, "SharedAutonomyCircle")
        .def(py::init<const PolicyTypes, const CircleWorld&, const std::vector<StateVector>&, const std::vector<double>&, const int, const int>())
        .def("get_last_state", &SharedAutonomyCircle::get_last_state)
        .def("get_last_control", &SharedAutonomyCircle::get_last_control)
        .def("get_state_at_ind", &SharedAutonomyCircle::get_state_at_ind)
        .def("get_control_at_ind", &SharedAutonomyCircle::get_control_at_ind)
        .def("get_num_timesteps_remaining", &SharedAutonomyCircle::get_num_timesteps_remaining)
        .def("is_done", &SharedAutonomyCircle::is_done)
        .def("get_rollout_cost", &SharedAutonomyCircle::get_rollout_cost)
        .def("run_control", &SharedAutonomyCircle::run_control)
        .def("get_num_states_computed", &SharedAutonomyCircle::get_num_states_computed)
        .def("get_states", &SharedAutonomyCircle::get_states)
    ;;


    


    return m.ptr();
}
