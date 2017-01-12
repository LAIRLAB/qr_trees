#include <pybind11/pybind11.h>
#include <experiments/simulators/circle_world.hh>

namespace py = pybind11;
using Circle = circle_world::Circle;
using CircleWorld = circle_world::CircleWorld;

void add_circle_class(py::module& m)
{
    py::class_<Circle>(m, "Circle")
        .def(py::init<const double, const double, const double>())
        .def(py::init<const double, const Eigen::Vector2d &>())
        .def_property("radius", (double (Circle::*)() const) &Circle::radius, [](Circle &c, double r) { c.radius() = r; })
        .def_property("position", (const Eigen::Vector2d& (Circle::*)() const) &Circle::position, [](Circle &c, const Eigen::Vector2d &p) { c.position() = p; })
        .def("__repr__", [](const Circle &c) 
                {
                    std::ostringstream oss;
                    oss << "Circle(" << c.radius() 
                    << ", [" << c.position().transpose() << "])"; 
                    return oss.str();
                });
        ;
}

void add_circle_world_class(py::module& m)
{
    py::class_<CircleWorld>(m, "CircleWorld")
        .def(py::init<>())
        .def(py::init<const std::array<double, 4> &>())
        .def("add_obstacle", (void (CircleWorld::*)(const double, double, double)) &CircleWorld::add_obstacle)
        .def("add_obstacle", (void (CircleWorld::*)(const double, const Eigen::Vector2d&)) &CircleWorld::add_obstacle)
        .def("add_obstacle", (void (CircleWorld::*)(const Circle &)) &CircleWorld::add_obstacle)
        .def("dimensions", &CircleWorld::dimensions)
        .def_property("obstacles", (std::vector<Circle>& (CircleWorld::*)()) &CircleWorld::obstacles, [](CircleWorld &c, std::vector<Circle>& obs)
                {
                    c.obstacles() = obs;
                })
        .def("__repr__", [](const CircleWorld &w)
                {
                    auto dims = w.dimensions();
                    std::ostringstream oss;
                    oss << "CircleWorld(" << w.obstacles().size()
                    << " obstacles, world_size=[" << dims[0] 
                    << " " << dims[1] << " " << dims[2] 
                    << " " << dims[3] << "])"; 
                    return oss.str();
                });
        ;

}
