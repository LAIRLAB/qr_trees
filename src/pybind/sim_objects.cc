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


namespace py = pybind11;

PYBIND11_PLUGIN(sim_objects) 
{
    using Circle = objects::Circle;

    py::module m("sim_objects", "Simple objects (circle, etc.).");

    py::class_<Circle>(m, "Circle")
        .def(py::init<const double, const double, const double>())
        .def(py::init<const double, const Eigen::Vector2d &>())
        .def_property("radius", (double (Circle::*)() const) &Circle::radius, [](Circle &c, double r) { c.radius() = r; })
        .def_property("position", (const Eigen::Vector2d& (Circle::*)() const) &Circle::position, [](Circle &c, const Eigen::Vector2d &p) { c.position() = p; })
        .def_property("x", (double (Circle::*)() const) &Circle::x, [](Circle &c, const double &x) { c.x() = x; })
        .def_property("y", (double (Circle::*)() const) &Circle::y, [](Circle &c, const double &y) { c.y() = y; })
        .def("__repr__", [](const Circle &c) 
                {
                    return c.repr();
                });
        ;

    return m.ptr();
}
