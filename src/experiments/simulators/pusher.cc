
#include <experiments/simulators/pusher.hh>
#include <utils/debug_utils.hh>


namespace pusher
{
    
PusherWorld::PusherWorld(const Circle &pusher, const Circle &object, 
        const Eigen::Vector2d &pusher_vel, const double dt) 
    : dt_(dt), pusher_(pusher), pusher_vel_(pusher_vel), object_(object), 
    initial_object_(object) 
{
    reset(state());
};

Vector<STATE_DIM> PusherWorld::state()
{
    Vector<STATE_DIM> x; 
    x.setZero();
    x[POS_X] = pusher_.x();
    x[POS_Y] = pusher_.y();
    x[V_X] = pusher_vel_[0];
    x[V_Y] = pusher_vel_[1];

    return x;
}

void PusherWorld::reset(const Vector<STATE_DIM>& x)
{
    
    object_stuck_ = false;

    pusher_.x() = x[POS_X];
    pusher_.y() = x[POS_Y];
    pusher_vel_[0] = x[V_X];
    pusher_vel_[1] = x[V_Y];

    object_ = initial_object_; 

    intersect_and_project_obj();
}

Vector<STATE_DIM> PusherWorld::operator()(const Vector<STATE_DIM>& x, const Vector<CONTROL_DIM>& u)
{
    return step(x, u);
}

Vector<STATE_DIM> PusherWorld::step(const Vector<STATE_DIM>& xt, const Vector<CONTROL_DIM>& ut)
{
    // Make sure the pusher matchesthe incoming state.
    IS_EQUAL(pusher_.x(), xt[POS_X]);
    IS_EQUAL(pusher_.y(), xt[POS_Y]);
    IS_EQUAL(pusher_vel_[0], xt[V_X]);
    IS_EQUAL(pusher_vel_[1], xt[V_Y]);

    const Vector<2> vt(xt[V_X]*dt_, xt[V_Y]*dt_); 

    Vector<STATE_DIM> xt1;
    xt1[POS_X] = xt[POS_X] + vt[0];
    xt1[POS_Y] = xt[POS_Y] + vt[1];
    xt1[V_X] = xt[V_X] + ut[dV_X]*dt_;
    xt1[V_Y] = xt[V_Y] + ut[dV_Y]*dt_;

    // Update the pusher state
    pusher_.x() = xt1[POS_X];
    pusher_.y() = xt1[POS_Y];
    pusher_vel_[0] = xt1[V_X];
    pusher_vel_[1] = xt1[V_Y];

    // If the object is stuck to the pusher, we need to push it along
    // with the pusher itself.
    if (object_stuck_)
    {
        object_.x() += vt[0];
        object_.y() += vt[1];
    }

    intersect_and_project_obj();

    return xt1;
}

void PusherWorld::intersect_and_project_obj() 
{
    // Update the object by first checking collision.
    double dist = 9999;
    bool intersects = pusher_.distance(object_, dist);

    // Object is now 'stuck' to the pusher.
    if (intersects)
    {
        object_stuck_ = true;
    }

    // If the object is 'inside' then we need to project it back out.
    if (dist < 0)
    {
        Vector<2> proj_vec = object_.position() - pusher_.position(); 
        double proj_norm = proj_vec.norm(); 
        // If we get exactly 0, then try using the velocity vector. 
        // if that is also 0, then perturb by a little and try again.
        if (proj_norm == 0 && pusher_vel_.norm() != 0)
        {
            proj_vec = pusher_vel_;
            proj_norm = proj_vec.norm(); 
        } 
        else 
        {
            proj_vec = object_.position() + 1e-5*Eigen::Vector2d::Random()
                - pusher_.position(); 
            proj_norm = proj_vec.norm(); 
        }

        object_.position() += std::abs(dist) * proj_vec / proj_vec.norm();
    }
}


} // pusher
