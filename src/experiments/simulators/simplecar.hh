//
// Dynamics of a simple car.
// State space is 4-d [pos_x, pos_y, vel, angle]
// Control is 2-d [acceleration, delta-angle]
// 

# pragma once


#include <ilqr/ilqr_taylor_expansions.hh> // Definition of dynamics function.

#include <Eigen/Dense>

#include <cstdint>

namespace simulators
{
namespace simplecar
{

enum State 
{
    POS_X = 0,
    POS_Y,
    VEL,
    ANG
};

enum Control 
{
    CNTRL_ACC = 0,
    CNTRL_DELTA_ANG,
};

// State is [position, velocity, Steering Angle]
constexpr int STATE_DIM = 5;

// Control is [Acceleration, Delta-Steering Angle]
constexpr int CONTROL_DIM = 2;

ilqr::DynamicsFunc make_discrete_dynamics_func(const double dt);

Eigen::VectorXd discrete_dynamics(const Eigen::VectorXd& xt,
                                  const Eigen::VectorXd& ut,
                                  const double dt);
} // namespace pendulum
} // namespace simulators
