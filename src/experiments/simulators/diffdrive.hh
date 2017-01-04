//
// Dynamics of Differential Drive circle robot. 
// Based on the dynamics in: http://arl.cs.utah.edu/pubs/ACC2014.pdf
//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// December 2016
//

# pragma once

#include <experiments/simulators/simulator_utils.hh>

#include <Eigen/Dense>

#include <array>

namespace simulators
{
namespace diffdrive
{

template<int dim>
using Vector = Eigen::Matrix<double, dim, 1>;

enum State
{
    POS_X = 0,
    POS_Y,
    THETA,
    dTHETA,
    dV_LEFT,
    dV_RIGHT,
    STATE_DIM
};

enum Control 
{
    V_LEFT = 0,
    V_RIGHT,
    CONTROL_DIM
};

// Useful for holding the parameters of the Differential Drive robot, 
// including integration timestep.
// Wheel distance defaults to 0.258 m, the width for the iRobot Create. 
class DiffDrive
{
public:
    DiffDrive(const double dt, 
            const std::array<double, 2> &control_lims, // {-min_u, max_u}
            const std::array<double, 4> &world_lims, // {-min_x, max_x, -min_y, max_y}
            const double wheel_dist=0.258);
    // Discrete time dynamics.
    Vector<STATE_DIM> operator()(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u);

    double wheel_dist() { return wheel_dist_; }
    double dt() { return dt_; }
private:
    double dt_;
    std::array<double, 2> control_lims_;
    std::array<double, 4> world_lims_;
    double wheel_dist_;
};

Vector<STATE_DIM> continuous_dynamics(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u, 
                                    const double wheel_dist);
} // namespace diffdrive
} // namespace simulators
