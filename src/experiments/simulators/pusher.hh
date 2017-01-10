//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// January 2017
//

#pragma once

#include <experiments/simulators/objects.hh>

namespace pusher
{

using Circle = objects::Circle;

template<int dim>
using Vector = Eigen::Matrix<double, dim, 1>;

// State of the pusher that is controlled.
enum State
{
    POS_X = 0,
    POS_Y,
    V_X,
    V_Y,
    OBJ_X,
    OBJ_Y,
    OBJ_STUCK,
    STATE_DIM
};

enum Control 
{
    dV_X = 0,
    dV_Y,
    CONTROL_DIM
};

class PusherWorld
{
public:
    PusherWorld(const Circle &pusher, const Circle &object, 
            const Eigen::Vector2d &pusher_vel, const double dt);

    Vector<STATE_DIM> state_vector();

    Vector<STATE_DIM> operator()(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u);
    Vector<STATE_DIM> step(const Vector<STATE_DIM>& xt, const Vector<CONTROL_DIM>& ut, const double dt);

    void reset(const Vector<STATE_DIM>& x);

    const Circle& pusher() const { return pusher_; };
    const Circle& object() const { return object_; };

    Circle& pusher() { return pusher_; };
    Circle& object() { return object_; };

    const Eigen::Vector2d& pusher_vel() const { return pusher_vel_; };

    double dt() const { return dt_; }

private:
    double dt_;

    Circle pusher_;
    Eigen::Vector2d pusher_vel_;

    Circle object_;

    // Is the object in contact and stuck to the pusher.
    bool object_stuck_ = false;

    // Returns true if object is stuck to pusher.
    bool intersect_and_project_obj(); 
};

}
