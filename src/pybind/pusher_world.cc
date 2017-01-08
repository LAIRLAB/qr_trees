//
// Python bindings for the simple objects defined in objects.hh.
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// January 2017
//

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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
    .export_values();

    py::enum_<pusher::Control>(m, "Control")
    .value("dV_X", pusher::Control::dV_X)
    .value("dV_Y", pusher::Control::dV_Y)
    .export_values();

    py::class_<PusherWorld>(m, "PusherWorld")
        .def(py::init<const Circle &, const Circle &, const Eigen::Vector2d &, 
                const double>())
        .def("step", &PusherWorld::step)
        .def("state", &PusherWorld::state)
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

    return m.ptr();
}
