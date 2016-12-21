
#include <experiments/simulators/pendulum.hh>

#include <utils/debug_utils.hh>

namespace simulators
{
namespace pendulum
{

using namespace std::placeholders;

Pendulum::Pendulum(const double dt, const double damping_coeff, const double length)
    : dt_(dt), damping_coeff_(damping_coeff), length_(length)
{
}

Vector<STATE_DIM> Pendulum::operator()(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u)
{
    const auto dyn = std::bind(continuous_dynamics, _1, _2, length_, damping_coeff_);
    return simulators::RK4<STATE_DIM,CONTROL_DIM>::rk4(dt_, x, u, dyn);
}

ilqr::DynamicsFunc make_discrete_dynamics_func(const double dt, const double length, const double damping_coeff)
{
    const auto params_bound_dyn 
        = [length, damping_coeff](const Eigen::VectorXd& state, const Eigen::VectorXd& control)
        {
            return static_cast<Eigen::VectorXd>(continuous_dynamics(state, control, length, damping_coeff));
        };

    return [dt, params_bound_dyn](const Eigen::VectorXd& state, const Eigen::VectorXd& control)
        {
            return static_cast<Eigen::VectorXd>(simulators::step(state, control, dt, params_bound_dyn, 0.1, 2));
        };
}

Vector<STATE_DIM> continuous_dynamics(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u, 
                                    const double length, const double damping_coeff) 
{
  Vector<STATE_DIM> x_dot;

  x_dot[THETA] = x[THETADOT];
  x_dot[THETADOT] = -9.81*std::sin(x[THETA])/length - damping_coeff*x[THETADOT] + u[TORQUE];

  return x_dot;
}

} // namespace pendulum
} // namespace simulators

