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

#include <experiments/single_obs_diffdrive.hh>

#include <experiments/simulators/circle_world.hh>

#include <array>
#include <sstream>
#include <tuple>

Eigen::MatrixXd add(const Eigen::MatrixXd &i, const Eigen::MatrixXd &j) 
{
    return i + j;
}

namespace py = pybind11;

PYBIND11_PLUGIN(ilqr_diffdrive) 
{
    using Circle = circle_world::Circle;
    using CircleWorld = circle_world::CircleWorld;

    py::module m("ilqr_diffdrive", "iLQR on DiffDrive dynamics.");

    py::enum_<PolicyTypes>(m, "PolicyTypes")
    .value("HINDSIGHT", PolicyTypes::HINDSIGHT)
    .value("TRUE_ILQR", PolicyTypes::TRUE_ILQR)
    .value("ARGMAX_ILQR", PolicyTypes::ARGMAX_ILQR)
    .value("PROB_WEIGHTED_CONTROL", PolicyTypes::PROB_WEIGHTED_CONTROL)
    .export_values() 
    .def("__str__", &to_string);

    m.def("single_obs_control_diffdrive", [](PolicyTypes policy, bool has_obs, std::array<double, 2> prior) 
            {
                std::string state_fname, obs_fname;
                const double cost_to_go
                    = single_obs_control_diffdrive(policy, has_obs, prior,
                        state_fname, obs_fname);
                return std::make_tuple(cost_to_go, state_fname, obs_fname); 
            }, 
            "Function for controlling diffdrive.");

    m.def("control_diffdrive", [](PolicyTypes policy, 
                CircleWorld &true_world, 
                CircleWorld &other_world, 
                std::array<double, 2> prior,
                std::string state_fname = "",
                std::string obs_fname = ""
                ) 
            {
                const double cost_to_go
                    = control_diffdrive(policy, true_world, 
                        other_world, prior,
                        state_fname, obs_fname);
                return std::make_tuple(cost_to_go, state_fname, obs_fname); 
            }, 
            "Function for controlling diffdrive.");

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

    return m.ptr();
}
