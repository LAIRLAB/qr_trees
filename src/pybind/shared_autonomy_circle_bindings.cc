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

//template< typename T >
//inline
//std::vector< T > to_std_vector( const py::object& iterable )
//{
//    return std::vector< T >( py::[](const simulators::directdrive::StateVector &s)
//            py::stl_input_iterator< T >ve( iterable ),
//                             py::stl_input_iterator< T >( ) );
//}

PYBIND11_PLUGIN(shared_autonomy_circle_bindings) 
{

    py::module m("shared_autonomy_circle_bindings", "iLQR on shared autonomy circle world.");


    py::enum_<PolicyTypes>(m, "PolicyTypes")
    .value("HINDSIGHT", PolicyTypes::HINDSIGHT)
    .value("TRUE_ILQR", PolicyTypes::TRUE_ILQR)
    .value("ARGMAX_ILQR", PolicyTypes::ARGMAX_ILQR)
    .value("PROB_WEIGHTED_CONTROL", PolicyTypes::PROB_WEIGHTED_CONTROL)
    .export_values() 
    .def("__str__", &to_string);


//    py::class_<simulators::directdrive::StateVector>(m, "StateVector")
//        .def(py::init<const double, const double>())
//        .def(py::init<const Eigen::Vector2d&>())
//        .def("__repr__", [](const simulators::directdrive::StateVector &s) 
//                {
//                    std::ostringstream oss;
//                    oss << "StateVector(" << s(0) << " " << s(1) << ")";
//                    return oss.str();
//                });
//        ;
//
//
//    py::class_< std::vector<simulators::directdrive::StateVector> >(m, "GoalList")
//        .def(py::init<>())
//        .def("add_goal", [](const simulators::directdrive::StateVector &s) {
//        .def("__repr__", [](const simulators::directdrive::StateVector &s) 
//                {
//                    std::ostringstream oss;
//                    oss << "StateVector(" << s(0) << " " << s(1) << ")";
//                    return oss.str();
//                });
//        ;

    m.def("control_shared_autonomy", [](PolicyTypes policy, 
                    CircleWorld &world,
                    //py::list goal_states_list,//simulators::directdrive::StateVector goal_states,
                    std::vector<simulators::directdrive::StateVector> goal_states,
                    std::vector<double> goal_priors,
                    int true_goal_ind,
                    std::string state_fname = "",
                    std::string obs_fname = ""
                )
            {
                //std::vector<simulators::directdrive::StateVector> goal_states;

                //for (auto item: goal_states_list)
                    //goal_states.push_back( (simulators::directdrive::StateVector)item);

                //std::vector<double> goal_priors(goal_priors_array);

                const double cost_to_go
                    = control_shared_autonomy(policy, world, goal_states, goal_priors, true_goal_ind, state_fname, obs_fname);
                return std::make_tuple(cost_to_go, state_fname, obs_fname); 
            }, 
            "Function for running iLQR on shared autonomy circle world.");
    
    add_circle_class(m);
    add_circle_world_class(m);


    return m.ptr();
}
