// 
// Simulator for directly controlling the state through velocity commands
//
// Shervin Javdani (sjavdani@cs.cmu.edu)
// January 2017
//

# pragma once

//#include <experiments/simulators/simulator_utils.hh>

#include <Eigen/Dense>

#include <array>

namespace simulators
{
namespace directdrive
{

//template<int dim>
//using Vector = Eigen::Matrix<double, dim, 1>;

enum State
{
    POS_X = 0,
    POS_Y,
    STATE_DIM
};

enum Control 
{
    V_X = 0,
    V_Y,
    CONTROL_DIM
};

using StateVector = Eigen::Matrix<double, STATE_DIM, 1>;
using ControlVector = Eigen::Matrix<double, CONTROL_DIM, 1>;

// Useful for holding the parameters of the Differential Drive robot, 
// including integration timestep.
// Wheel distance defaults to 0.258 m, the width for the iRobot Create. 
class DirectDrive
{
public:
    DirectDrive(const double dt, 
            const std::array<double, CONTROL_DIM> &control_lims, // {-min_u, max_u}
            const std::array<double, STATE_DIM*2> &world_lims // {-min_x, max_x, -min_y, max_y}
            );
    // Discrete time dynamics.
    StateVector operator()(const StateVector& x, const ControlVector& u);

    double dt() { return dt_; }
private:
    double dt_;
    std::array<double, 2> control_lims_;
    std::array<double, 4> world_lims_;
};

StateVector continuous_dynamics(const StateVector& x, const ControlVector& u) { return u;}
} // namespace directdrive
} // namespace simulators
