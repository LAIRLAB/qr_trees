
#include <experiments/simulators/simplecar.hh>

#include <utils/debug_utils.hh>

namespace simulators
{
namespace simplecar 
{

ilqr::DynamicsFunc make_discrete_dynamics_func(const double dt)
{
    return [dt](const Eigen::VectorXd& state, const Eigen::VectorXd& control)
    {
        return static_cast<Eigen::VectorXd>(simplecar::discrete_dynamics(state, control, dt));
    };
}

Eigen::VectorXd discrete_dynamics(const Eigen::VectorXd& xt, 
                                  const Eigen::VectorXd& ut, 
                                  const double dt) 
{
    IS_EQUAL(xt.size(), simplecar::STATE_DIM);
    IS_EQUAL(ut.size(), simplecar::CONTROL_DIM);

    // State at t+1
    Eigen::VectorXd xt1(xt.size());

    const double theta = xt[ANG];
    xt1[POS_X] = xt[POS_X] + std::cos(theta)*xt[VEL]*dt;
    xt1[POS_Y] = xt[POS_Y] + std::sin(theta)*xt[VEL]*dt;
    xt1[VEL] = xt[VEL] + ut[CNTRL_ACC]*dt;
    xt1[ANG] = xt[ANG] + ut[CNTRL_DELTA_ANG]*dt;
    return xt1;
}

} // namespace pendulum
} // namespace simulators

