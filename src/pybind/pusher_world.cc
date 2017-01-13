//
// Python bindings for the simple objects defined in objects.hh.
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// January 2017
//

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <experiments/circle_pusher.hh>
#include <experiments/simulators/objects.hh>
#include <experiments/simulators/pusher.hh>

#include <sstream>

namespace py = pybind11;

PYBIND11_PLUGIN(pusher_world) 
{
    using Circle = objects::Circle;
    using PusherWorld = pusher::PusherWorld;

    py::module m("pusher_world", "Simple objects (circle, etc.).");

    py::enum_<pusher::State>(m, "State")
    .value("POS_X", pusher::State::POS_X)
    .value("POS_Y", pusher::State::POS_Y)
    .value("V_X", pusher::State::V_X)
    .value("V_Y", pusher::State::V_Y)
    .value("OBJ_X", pusher::State::OBJ_X)
    .value("OBJ_Y", pusher::State::OBJ_Y)
    .value("OBJ_STUCK", pusher::State::OBJ_STUCK)
    .export_values();

    py::enum_<pusher::Control>(m, "Control")
    .value("dV_X", pusher::Control::dV_X)
    .value("dV_Y", pusher::Control::dV_Y)
    .export_values();

    py::class_<PusherWorld>(m, "PusherWorld")
        .def(py::init<const Circle &, const Circle &, const Eigen::Vector2d &, 
                const double>())
        .def("step", &PusherWorld::step)
        .def("state", &PusherWorld::state_vector)
        .def("reset", &PusherWorld::reset)
        .def("pusher", (const Circle& (PusherWorld::*)() const) &PusherWorld::pusher)
        .def("object", (const Circle& (PusherWorld::*)() const) &PusherWorld::object)
        //.def_property("pusher", (const Circle& (PusherWorld::*)() const) &PusherWorld::pusher, [](PusherWorld &c, const Circle &pusher) { c.pusher() = pusher; })
        //.def_property("object", (const Circle& (PusherWorld::*)() const) &PusherWorld::object, [](PusherWorld&c, const Circle &object) { c.object() = object; })
        .def("__repr__", [](const PusherWorld &w) 
                {
                    std::ostringstream oss;
                    oss << "PusherWorld(" << w.pusher().repr()
                        << ", " << w.object().repr() 
                        << ", [" << w.pusher_vel()[0] << "," << w.pusher_vel()[1] << "]"
                        << ", " << w.dt()
                        << ")";
                    return oss.str();
                });
        ;


    py::enum_<PolicyTypes>(m, "PolicyTypes")
    .value("HINDSIGHT", PolicyTypes::HINDSIGHT)
    .value("TRUE_ILQR", PolicyTypes::TRUE_ILQR)
    .value("ARGMAX_ILQR", PolicyTypes::ARGMAX_ILQR)
    .value("PROB_WEIGHTED_CONTROL", PolicyTypes::PROB_WEIGHTED_CONTROL)
    .export_values() 
    .def("__str__", &to_string);

    m.def("control_pusher", 
            [](const PolicyTypes policy, 
                const int true_obj_index
                ) 
            { 
                const std::vector<double> &obj_probability = {0.5, 0.5};
                const std::vector<Circle> &possible_objects = {Circle(3,-5,0), Circle(3,5,0)};
                std::vector<pusher::Vector<pusher::STATE_DIM>> states;
                const double rollout_cost = control_pusher(policy, possible_objects, obj_probability, 
                    true_obj_index, states);
                return std::make_tuple(rollout_cost, states);
            });

    return m.ptr();
}
