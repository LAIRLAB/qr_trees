
#include <experiments/simulators/pendulum.hh>

#include <utils/debug_utils.hh>

namespace simulators
{
namespace pendulum
{

ilqr::DynamicsFunc make_discrete_dynamics_func(const double dt, const double length, const double damping_coeff)
{
    const auto params_bound_dyn 
        = [length, damping_coeff](const Eigen::VectorXd& state, const Eigen::VectorXd& control)
        {
            return static_cast<Eigen::VectorXd>(continuous_dynamics(state, control, length, damping_coeff));
        };

    return [dt, params_bound_dyn](const Eigen::VectorXd& state, const Eigen::VectorXd& control)
        {
            return static_cast<Eigen::VectorXd>(simulators::step(state, control, dt, params_bound_dyn));
        };
}

Eigen::VectorXd continuous_dynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control, 
                                    const double length, const double damping_coeff) 
{
  IS_EQUAL(state.size(), 2); 
  IS_EQUAL(control.size(), 1); 
  Eigen::VectorXd velocities(state.size());

  velocities[0] = state[1];
  velocities[1] = -9.81*std::sin(state[0])/length - damping_coeff*state[1] + control[0];

  return velocities;
}

} // namespace pendulum
} // namespace simulators

