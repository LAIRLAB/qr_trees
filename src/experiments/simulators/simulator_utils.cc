
#include <experiments/simulators/simulator_utils.hh>

namespace 
{

  // 4th order Runge-Kutta integration of an ODE given by dynamics.
  Eigen::VectorXd rk4(const double dt,
                      const Eigen::VectorXd& state, 
                      const Eigen::VectorXd& control,
                      const ilqr::DynamicsFunc &dynamics) 
  {

  // This cannot be changed
  constexpr double RK4_INTEGRATION_CONSTANT = 1.0/6.0;

  // Formula from: http://mathworld.wolfram.com/Runge-KuttaMethod.html 
  const Eigen::VectorXd& k1 = dynamics(state, control);
  const Eigen::VectorXd& k2 = dynamics(state + 0.5*dt*k1, control);
  const Eigen::VectorXd& k3 = dynamics(state + 0.5*dt*k2, control);
  const Eigen::VectorXd& k4 = dynamics(state + dt*k3, control);

  const Eigen::VectorXd result = state + RK4_INTEGRATION_CONSTANT*dt*(k1 + 2.0*(k2 + k3) + k4);

  return result;
}


}

namespace simulators
{

Eigen::VectorXd step(const Eigen::VectorXd &state, const Eigen::VectorXd &control, 
                     const double dt, const ilqr::DynamicsFunc &dynamics, 
                     const double min_integration_dt,
                     const double integration_frequency)
{
  // integration dt to be considered based on passed in dt
  double integration_dt = dt/integration_frequency; 
  int num_steps = static_cast<int>(std::ceil(dt/integration_dt));

  if (integration_dt > min_integration_dt) {
      // Since we are above the min integration point, compute how many integration steps we should 
      // take in order to be at least at the min dt or lower dt
      num_steps = static_cast<int>(std::ceil(dt/MIN_INTEGRATION_DT));
      integration_dt = dt/static_cast<double>(num_steps); 
  }

  Eigen::VectorXd result = state;

  for (int i = 0; i < num_steps; ++i) 
  {
    result.noalias() = rk4(integration_dt, result, control, dynamics);
  }

  return result;
}

} // namespace simulators
