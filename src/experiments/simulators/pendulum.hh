//
// Dynamics of a single pendulum.
// State space is 2-d [\theta, \dot\theta]: angle, angular velocity
// If \theta=\pi is straight up, \theta=0 is straight down
// 

# pragma once

#include <Eigen/Dense>

#include <experiments/simulators/simulator_utils.hh>

namespace simulators
{
namespace pendulum
{

ilqr::DynamicsFunc make_discrete_dynamics_func(const double dt, const double length, const double damping_coeff);

Eigen::VectorXd continuous_dynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control, 
                                    const double length, const double damping_coeff);
} // namespace pendulum
} // namespace simulators
