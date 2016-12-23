
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
    IS_GREATER(control_lims_[1], control_lims_[0]);
    IS_GREATER(world_lims_[1], world_lims_[0]);
    IS_GREATER(world_lims_[3], world_lims_[2]);
}

Vector<STATE_DIM> DiffDrive::operator()(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u)
{
    const auto dyn = std::bind(continuous_dynamics, _1, _2, wheel_dist_);

    Vector<CONTROL_DIM> u_lim = u;
    const double u_range = (control_lims_[1] - control_lims_[0]);
    for (int c_dim = 0; c_dim < CONTROL_DIM; ++c_dim)
    {
        u_lim[c_dim] = u_range * 1.0/(1.0 + std::exp(-u[c_dim])) + control_lims_[0];
    }
    //IS_GREATER_EQUAL(u_lim[V_LEFT], control_lims_[0]);
    //IS_GREATER_EQUAL(u_lim[V_LEFT], control_lims_[0]);
    //IS_LESS_EQUAL(u_lim[V_RIGHT], control_lims_[1]);
    //IS_LESS_EQUAL(u_lim[V_RIGHT], control_lims_[1]);

    Vector<STATE_DIM> xt1 = simulators::RK4<STATE_DIM,CONTROL_DIM>::rk4(dt_, x, u_lim, dyn);

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
    //const double coupled_vel = 0.5*(u[V_LEFT] + u[V_RIGHT]);
    const double coupled_vel = 0.5*(x[dV_LEFT] + x[dV_RIGHT]);
    x_dot[POS_X] = coupled_vel * std::cos(x[THETA]); // xdot
    x_dot[POS_Y] = coupled_vel * std::sin(x[THETA]); // ydot
    x_dot[THETA] = x[dTHETA];
    x_dot[dTHETA] = (u[V_LEFT] - u[V_RIGHT])/wheel_dist; // theta double-dot
    x_dot[dV_LEFT]= u[V_LEFT]; 
    x_dot[dV_RIGHT] = u[V_RIGHT];

    return x_dot;
}

} // namespace diffdrive
} // namespace simulators

