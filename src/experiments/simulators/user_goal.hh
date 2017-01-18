// 
// Cost function and helpers to simulate how user models a goal
//
// Shervin Javdani (sjavdani@cs.cmu.edu)
// January 2017
//

# pragma once

#include <experiments/simulators/directdrive.hh>
#include <experiments/simulators/circle_world.hh>

namespace user_goal
{

using CircleWorld = circle_world::CircleWorld;
using Circle = circle_world::Circle;

using State = simulators::directdrive::State;
constexpr int STATE_DIM = simulators::directdrive::STATE_DIM;
constexpr int CONTROL_DIM = simulators::directdrive::CONTROL_DIM;
using StateVector = simulators::directdrive::StateVector;
using ControlVector = simulators::directdrive::ControlVector;

template <int rows>
using Vector = Eigen::Matrix<double, rows, 1>;

template <int rows, int cols>
using Matrix = Eigen::Matrix<double, rows, cols>;


double obstacle_cost(const CircleWorld &world, const double robot_radius, const StateVector &xt);
double ct(const StateVector &x, const ControlVector &u, const int t, const CircleWorld &world, const StateVector& goal_state);
// Final timestep cost function
double cT(const StateVector &x, const StateVector& goal_state);



struct User_Goal {
    public: 
        User_Goal(const StateVector& goal_state, double prob, const CircleWorld& world);

        StateVector goal_state_;
        double prob_;
        std::function<double(const StateVector&)> cT_;
        std::function<double(const StateVector&, const ControlVector&, const int)> ct_;
};




} //namespace user_goal
