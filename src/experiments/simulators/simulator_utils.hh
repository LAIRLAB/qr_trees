//
// Helper utilities for simulators.
//

#pragma once

#include <ilqr/ilqr_taylor_expansions.hh> // Definition of dynamics function.

#include <Eigen/Dense>

#include <functional>

namespace simulators
{

template<int dim>
using Vector = Eigen::Matrix<double, dim, 1>;

// Wrapper struct so we can use template paramters.
template<int xdim, int udim>
struct RK4 
{
using Dynamics = std::function<Vector<xdim>(const Vector<xdim> &x, const Vector<udim> &u)>;

// 4th order Runge-Kutta integration of an ODE given by dynamics.
static Vector<xdim> rk4(const double dt,
        const Vector<xdim> state, 
        const Vector<udim> control,
        const Dynamics &dynamics) 
{

    // This cannot be changed
    constexpr double RK4_INTEGRATION_CONSTANT = 1.0/6.0;

    // Formula from: http://mathworld.wolfram.com/Runge-KuttaMethod.html 
    const Vector<xdim> k1 = dynamics(state, control);
    const Vector<xdim> k2 = dynamics(state + 0.5*dt*k1, control);
    const Vector<xdim> k3 = dynamics(state + 0.5*dt*k2, control);
    const Vector<xdim> k4 = dynamics(state + dt*k3, control);

    const Vector<xdim> result = state + RK4_INTEGRATION_CONSTANT*dt*(k1 + 2.0*(k2 + k3) + k4);
    return result;
}

};

constexpr double INTEGRATION_FREQUENCY = 5;
constexpr double MIN_INTEGRATION_DT = 1e-2;


Eigen::VectorXd step(const Eigen::VectorXd &state, const Eigen::VectorXd &control, 
                     const double dt, const ilqr::DynamicsFunc &dynamics, 
                     const double min_integration_dt = MIN_INTEGRATION_DT,
                     const double integration_frequency  = INTEGRATION_FREQUENCY
                     );

} // namespace simulators
