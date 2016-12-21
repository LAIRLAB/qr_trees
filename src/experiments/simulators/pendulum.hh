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

template<int dim>
using Vector = simulators::Vector<dim>;

enum State
{
    THETA = 0,
    THETADOT,
    STATE_DIM
};

enum Control 
{
    TORQUE = 0,
    CONTROL_DIM
};

// Useful for holding the parameters of the pendulum, including integration timestep.
class Pendulum
{
public:
    Pendulum(const double dt, const double damping_coeff, const double length);
    // Discrete time dynamics.
    Vector<STATE_DIM> operator()(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u);
private:
    double dt_;
    double damping_coeff_;
    double length_;

};


// Returns a function that can return x_{t+1} = f(x_t, u_t) for a chosen dt. 
// Underlying uses RK4 integration.
ilqr::DynamicsFunc make_discrete_dynamics_func(const double dt, const double length, const double damping_coeff);

// Defines pendulum dynamics as xdot = f(x,u).
//Eigen::VectorXd continuous_dynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control, 
//                                    const double length, const double damping_coeff);

Vector<STATE_DIM> continuous_dynamics(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u, 
                                    const double length, const double damping_coeff);
} // namespace pendulum
} // namespace simulators
