#include <utils/debug_utils.hh>
#include <utils/print_helpers.hh>

#include <algorithm>

#include <experiments/simulators/user_goal.hh>

namespace user_goal
{

//Matrix<STATE_DIM, STATE_DIM> QT = 25*Matrix<STATE_DIM,STATE_DIM>::Identity(); // Quadratic state cost for final timestep.
#ifdef USE_VEL
Matrix<STATE_DIM, STATE_DIM> Q = 1e-5*Matrix<STATE_DIM,STATE_DIM>::Identity();
Matrix<STATE_DIM, STATE_DIM> QT = Eigen::DiagonalMatrix<double,STATE_DIM,STATE_DIM>(StateVector(200., 200., 25., 25.));
Matrix<CONTROL_DIM, CONTROL_DIM> R = 0.2*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();
#else
Matrix<STATE_DIM, STATE_DIM> Q = 0.01*Matrix<STATE_DIM,STATE_DIM>::Identity();
Matrix<STATE_DIM, STATE_DIM> QT = 25*Matrix<STATE_DIM,STATE_DIM>::Identity(); // Quadratic state cost for final timestep.
Matrix<CONTROL_DIM, CONTROL_DIM> R = 2*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();
#endif



StateVector xT; // Goal state for final timestep.
StateVector x0; // Start state for 0th timestep.

//Q = 1*Matrix<STATE_DIM,STATE_DIM>::Identity();
//QT = 25*Matrix<STATE_DIM,STATE_DIM>::Identity();
//R = 2*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();

double robot_radius_ = 3.35/2.0; // iRobot create;
double obstacle_factor_ = 300.0;
double scale_factor_ = 1.0e0;

double obstacle_cost(const CircleWorld &world, const double robot_radius, const StateVector &xt)
{
    Eigen::Vector2d robot_pos;
    robot_pos << xt[State::POS_X], xt[State::POS_Y];

    // Compute minimum distance to the edges of the world.
    double cost = 0;

    for (auto obstacle : world.obstacles()){
        Eigen::Vector2d d = robot_pos - obstacle.position();
        double distr = d.norm(); 
        double dist = distr - robot_radius - obstacle.radius();
        cost += obstacle_factor_ * exp(-scale_factor_*dist);
    }
    return cost;
}

double ct(const StateVector &x, const ControlVector &u, const int t, const CircleWorld &world, const StateVector& goal_state)
{
    double cost = 0;

    // position
//    if (t == 0)
//    {
//        StateVector dx = x - x0;
//        cost += 0.5*(dx.transpose()*Q*dx)[0];
//    }

    // Control cost
    //const ControlVector du = u - u_nominal;
    //cost += 0.5*(du.transpose()*R*du)[0];
    cost += 0.5*(u.transpose()*R*u)[0];

    //add cost for not being at goal
    const StateVector dx = x - goal_state;
    cost += 0.5*(dx.transpose()*Q*dx)[0];

    cost += obstacle_cost(world, robot_radius_, x);

    return cost;
}

// Final timestep cost function
double cT(const StateVector &x, const StateVector& goal_state)
{
    const StateVector dx = x - goal_state;
    return 0.5*(dx.transpose()*QT*dx)[0];
}


User_Goal::User_Goal(const StateVector& goal_state, double prob, const CircleWorld& world)
        : goal_state_(goal_state), prob_(prob), cT_(std::bind(cT, std::placeholders::_1, goal_state)), ct_(std::bind(ct, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, world, goal_state))
    {}

} //namespace user_goal
