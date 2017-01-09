//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// January 2017
//

#include <experiments/circle_pusher.hh>

#include <experiments/simulators/pusher.hh>
#include <experiments/simulators/objects.hh>
#include <templated/iLQR_hindsight.hh>
#include <utils/math_utils_temp.hh>
#include <utils/debug_utils.hh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace 
{

using Circle = objects::Circle;

using State = pusher::State;
constexpr int STATE_DIM = pusher::STATE_DIM;
constexpr int CONTROL_DIM = pusher::CONTROL_DIM;

template <int rows>
using Vector = Eigen::Matrix<double, rows, 1>;

template <int rows, int cols>
using Matrix = Eigen::Matrix<double, rows, cols>;

double robot_radius = 4;
double object_radius = 3;
Matrix<STATE_DIM, STATE_DIM> Q;
Matrix<STATE_DIM, STATE_DIM> QT; // Quadratic state cost for final timestep.
Matrix<CONTROL_DIM, CONTROL_DIM> R;

Vector<STATE_DIM> xT; // Goal state for final timestep.
Vector<STATE_DIM> x0; // Start state for 0th timestep.
Vector<CONTROL_DIM> u_nominal; 

int T;

double ct(const Vector<STATE_DIM> &x, const Vector<CONTROL_DIM> &u, const int t)
{
    double cost = 0;

    // position
    if (t == 0)
    {
        Vector<STATE_DIM> dx = x - x0;
        cost += 0.5*(dx.transpose()*Q*dx)[0];
    }

    // Control cost
    //const Vector<CONTROL_DIM> du = u - u_nominal;
    const Vector<CONTROL_DIM> du = u;
    cost += 0.5*(du.transpose()*R*du)[0];


    const Eigen::Vector2d robot_pos(x[State::POS_X], x[State::POS_Y]);
    const Eigen::Vector2d obj_pos(x[State::OBJ_X], x[State::OBJ_Y]);
    const double diff = (robot_pos - obj_pos).norm();

    // This moves the robot towards the object until it touches.
    cost += 5.0*std::max(0.0, diff - robot_radius - object_radius);

    return cost;
}

// Final timestep cost function
double cT(const Vector<STATE_DIM> &x)
{
    const Vector<STATE_DIM> dx = x - xT;
    return 0.5*(dx.transpose()*QT*dx)[0];
}

int get_argmax(const std::array<double, 2> &prob)
{
    if (prob[0] > prob[1])
    {
        return 0;
    }
    return 1;
}


} // namespace

double control_pusher(const PolicyTypes policy, 
        std::vector<pusher::Vector<pusher::STATE_DIM>> &states
        )
{
    using namespace std::placeholders;

    T = 50;
	const double dt = 1.0/6.0;
    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    DEBUG("Running with policy \"" << to_string(policy))

	xT = Vector<STATE_DIM>::Zero();
	xT[State::POS_X] = 0;
	xT[State::POS_Y] = 25;
	xT[State::V_X] = 0; 
	xT[State::V_Y] = 0;
	xT[State::OBJ_X] = 10;//xT[State::POS_X]+robot_radius+object_radius; 
	xT[State::OBJ_Y] = 15;//xT[State::POS_Y]+robot_radius+object_radius; 
	xT[State::OBJ_STUCK] = 1; 

    PRINT("Target: " << xT.transpose());

	x0 = Vector<STATE_DIM>::Zero();
	x0[State::POS_X] = 0;
	x0[State::POS_Y] = 0;
	x0[State::V_X] = 0;
	x0[State::V_Y] = 0;
	x0[State::OBJ_X] = 0; 
	x0[State::OBJ_Y] = 10; 
	x0[State::OBJ_STUCK] = 0; 

	Q = 1*Matrix<STATE_DIM,STATE_DIM>::Identity();

    QT = 1e-3*Matrix<STATE_DIM,STATE_DIM>::Identity();
    QT(State::V_X, State::V_X) = 5.0;
    QT(State::V_Y, State::V_Y) = 5.0;
    QT(State::OBJ_X, State::OBJ_X) = 50.0;
    QT(State::OBJ_Y, State::OBJ_Y) = 50.0;
    //QT(State::OBJ_STUCK, State::OBJ_STUCK) = 1.0;

	R = 5*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();

    // Initial linearization points are linearly interpolated states and zero
    // control.
    u_nominal[0] = 0.0;
    u_nominal[1] = 0.5;

    //const std::array<double, 2> CONTROL_LIMS = {{-5, 5}};
    
    Circle pusher(robot_radius, x0[State::POS_X], x0[State::POS_Y]);
    Circle object(object_radius, x0[State::OBJ_X], x0[State::OBJ_Y]);
    pusher::PusherWorld true_world(pusher, object, 
            Eigen::Vector2d(x0[State::V_X], x0[State::V_Y]), 
            dt);

    constexpr bool verbose = false;
    constexpr int max_iters_begin = 300;
    constexpr int max_iters = 100;
    constexpr double mu = 0.05;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 10;

    // Setup the cost function with environments that have the obstacle and one
    // that does not.
    auto ct_true_world = ct; 
    auto cT_true_world = cT; 
    //auto ct_other_world = std::bind(ct, _1, _2, _3, other_world);
    
    //std::array<double, 2> obs_probability = OBS_PRIOR;
    
    // Make a copy
    pusher::PusherWorld true_dynamics = true_world;

    // Setup the true system solver.
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> true_branch(true_dynamics, cT_true_world, ct_true_world, 1.0);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> true_chain_solver({true_branch});

    clock_t ilqr_begin_time = clock();
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> *solver 
        = &true_chain_solver; 

    solver->solve(T, x0, u_nominal, mu, max_iters_begin, 
            verbose, convg_thresh, start_alpha);
    PRINT("start: Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

    std::vector<Vector<pusher::Control::CONTROL_DIM>> controls;
    //return solver->forward_pass(0, x0, states, controls, 1.0); 

    double rollout_cost = 0;
    Vector<STATE_DIM> xt = x0;
    // store initial state
    states.push_back(xt);
    //TODO: How to run this for full T?
    for (int t = 0; t < T-1; ++t)
    {
        const bool t_offset = t >  0 ? 1 : 0;
        const int plan_horizon = T-t;
        //const int plan_horizon = std::min(T-t, MPC_HORIZON);
        
        ilqr_begin_time = clock();

        Vector<CONTROL_DIM> ut;
        solver->solve(plan_horizon, xt, u_nominal, mu, 
                max_iters, verbose, convg_thresh, start_alpha, 
                true, t_offset);
        ut = solver->compute_first_control(xt); 

        PRINT("t=" << t << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

        rollout_cost += ct_true_world(xt, ut, t);
        const Vector<STATE_DIM> xt1 = true_world(xt, ut);

        xt = xt1;
        states.push_back(xt);

        //const Eigen::Vector2d robot_position(xt[State::POS_X], xt[State::POS_Y]);
    }
    rollout_cost += cT_true_world(xt);
    DEBUG(" x_rollout(" << T-1 << ")= " << xt.transpose());
    DEBUG(" Total cost rollout: " << rollout_cost);

    return rollout_cost;
}

