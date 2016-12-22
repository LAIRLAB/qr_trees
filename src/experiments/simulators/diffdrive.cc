
#include <experiments/simulators/diffdrive.hh>

#include <utils/debug_utils.hh>

namespace simulators
{
namespace diffdrive
{

using namespace std::placeholders;

DiffDrive::DiffDrive(const double dt, 
                    const std::array<double, 2> &control_lims,
                    const std::array<double,4> &world_lims, 
                    const double wheel_dist)
    : dt_(dt), control_lims_(control_lims), world_lims_(world_lims), wheel_dist_(wheel_dist)
{
}

Vector<STATE_DIM> DiffDrive::operator()(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u)
{
    const auto dyn = std::bind(continuous_dynamics, _1, _2, wheel_dist_);
    Vector<CONTROL_DIM> u_lim = u;
    u_lim[V_LEFT] = std::min(u[V_LEFT], control_lims_[1]);
    u_lim[V_RIGHT] = std::min(u[V_RIGHT], control_lims_[1]);
    u_lim[V_LEFT] = std::max(u[V_LEFT], control_lims_[0]);
    u_lim[V_RIGHT] = std::max(u[V_RIGHT], control_lims_[0]);

    Vector<STATE_DIM> xt1 = simulators::RK4<STATE_DIM,CONTROL_DIM>::rk4(dt_, x, u_lim, dyn);

    //if (world_lims_[0]) {}
    xt1[POS_X] = std::min(xt1[POS_X], world_lims_[1]);
    xt1[POS_X] = std::max(xt1[POS_X], world_lims_[0]);
    xt1[POS_Y] = std::min(xt1[POS_Y], world_lims_[3]);
    xt1[POS_Y] = std::max(xt1[POS_Y], world_lims_[2]);
    return xt1;
}


Vector<STATE_DIM> continuous_dynamics(const Vector<STATE_DIM>& x, 
        const Vector<CONTROL_DIM>& u, const double wheel_dist) 
{
    Vector<STATE_DIM> x_dot;

    // Differential-drive. Set the derivative wrt to the state.
    const double coupled_vel = 0.5*(u[V_LEFT] + u[V_RIGHT]);
    x_dot[POS_X] = coupled_vel * std::cos(x[THETA]); // xdot
    x_dot[POS_Y] = coupled_vel * std::sin(x[THETA]); // ydot
    x_dot[THETA] = (u[V_LEFT] - u[V_RIGHT])/wheel_dist; // thetadot

    return x_dot;
}

} // namespace diffdrive
} // namespace simulators

