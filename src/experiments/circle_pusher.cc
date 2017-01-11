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
#include <numeric>

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

// Useful typedef shorthands.
using DynamicsFunc = std::function<Vector<STATE_DIM>(const Vector<STATE_DIM> &,const Vector<CONTROL_DIM> &)>;
using Branch = ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM>;

// Used for the first time step when the object pose is uknown 
constexpr double UNKOWN_OBJ_POS = -9999; 

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

    // If the object X or object Y is nan, then the object state is 
    // unknown in this timestep. 
    if (x[State::OBJ_X] != UNKOWN_OBJ_POS)
    {
        const Eigen::Vector2d robot_pos(x[State::POS_X], x[State::POS_Y]);
        const Eigen::Vector2d obj_pos(x[State::OBJ_X], x[State::OBJ_Y]);
        const double diff = (robot_pos - obj_pos).norm();

        // This moves the robot towards the object until it touches.
        cost += 5.0*std::max(0.0, diff - robot_radius - object_radius);
    }

    return cost;
}

// Final timestep cost function
double cT(const Vector<STATE_DIM> &x)
{
    const Vector<STATE_DIM> dx = x - xT;
    return 0.5*(dx.transpose()*QT*dx)[0];
}

Vector<STATE_DIM> pusher_dynamics(const Vector<STATE_DIM> &x, const Vector<CONTROL_DIM> &u, 
        const Eigen::Vector2d &obj_start_pos, pusher::PusherWorld &world)
{
    Vector<STATE_DIM> x_fixed = x;
    // If the object X or object Y is nan, then the object state is unknown in
    // the previous timestep. We set it in the dynamics to move forward.
    if (x[State::OBJ_X] == UNKOWN_OBJ_POS || x[State::OBJ_Y] == UNKOWN_OBJ_POS)
    {
        x_fixed[State::OBJ_X] = obj_start_pos[0];
        x_fixed[State::OBJ_Y] = obj_start_pos[1];
    }
    return world(x_fixed, u);
}

int get_argmax(const std::vector<double> &probs)
{
    const int arg_max_prob = std::distance(probs.begin(),
                std::max_element(probs.begin(), probs.end()));
    return arg_max_prob;
}


} // namespace

double control_pusher(const PolicyTypes policy, 
        std::vector<pusher::Vector<pusher::STATE_DIM>> &states
        )
{
    using namespace std::placeholders;

    const Circle pusher(robot_radius, x0[State::POS_X], x0[State::POS_Y]);

    const Circle true_object(object_radius, -5, 10);
    const Circle other_object(object_radius, 5, 10);

    const std::vector<Circle> possible_objects = {true_object, other_object};
    const int true_obj_index = 0;
    std::vector<double> obs_probability = {0.5, 0.5};
    IS_EQUAL(possible_objects.size(), obs_probability.size());
    IS_BETWEEN_LOWER_INCLUSIVE(true_obj_index, 0, possible_objects.size());

    T = 25;
	const double dt = 0.5;
    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    DEBUG("Running with policy \"" << to_string(policy))

	xT = Vector<STATE_DIM>::Zero();
	xT[State::POS_X] = 0;
	xT[State::POS_Y] = 25;
	xT[State::V_X] = 0; 
	xT[State::V_Y] = 0;
	xT[State::OBJ_X] = 0;//xT[State::POS_X]+robot_radius+object_radius; 
	xT[State::OBJ_Y] = 25;//xT[State::POS_Y]+robot_radius+object_radius; 
	xT[State::OBJ_STUCK] = 1; 

    PRINT("Target: " << xT.transpose());

	x0 = Vector<STATE_DIM>::Zero();
	x0[State::POS_X] = 0;
	x0[State::POS_Y] = 0;
	x0[State::V_X] = 0;
	x0[State::V_Y] = 0;
	x0[State::OBJ_X] = UNKOWN_OBJ_POS; 
	x0[State::OBJ_Y] = UNKOWN_OBJ_POS; 
	x0[State::OBJ_STUCK] = 0; 

	Q = 1*Matrix<STATE_DIM,STATE_DIM>::Identity();
    Q(State::OBJ_X, State::OBJ_X) = 0.0;
    Q(State::OBJ_Y, State::OBJ_Y) = 0.0;

    QT = 1e-3*Matrix<STATE_DIM,STATE_DIM>::Identity();
    QT(State::V_X, State::V_X) = 5.0;
    QT(State::V_Y, State::V_Y) = 5.0;
    QT(State::OBJ_X, State::OBJ_X) = 80.0;
    QT(State::OBJ_Y, State::OBJ_Y) = 80.0;
    //QT(State::OBJ_STUCK, State::OBJ_STUCK) = 1.0;

	R = 10*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();

    // Initial linearization points are linearly interpolated states and zero
    // control.
    u_nominal[0] = 0.0;
    u_nominal[1] = 0.5;

    //const std::array<double, 2> CONTROL_LIMS = {{-5, 5}};
    
    constexpr bool verbose = false;
    constexpr int max_iters_begin = 500;
    constexpr int max_iters = 300;
    constexpr double mu = 0.05;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 10;
    
    // We use a version that can take UNKOWN_OBJ_POS and convert it on the next
    // time step to resolve the ambiguity.

    std::vector<DynamicsFunc> dynamics_funcs; 
    std::vector<pusher::PusherWorld> worlds; 
    dynamics_funcs.reserve(possible_objects.size());
    worlds.reserve(possible_objects.size());
    for(const auto &object : possible_objects)
    {
        worlds.emplace_back(pusher, object, Eigen::Vector2d(x0[State::V_X], x0[State::V_Y]), dt);
        dynamics_funcs.push_back(std::bind(pusher_dynamics, _1, _2, 
                    std::cref(object.position()), std::ref(worlds.back())));
    }
    auto true_dynamics = dynamics_funcs[true_obj_index];
    auto true_world = worlds[true_obj_index];

    // Setup the true system solver.
    const Branch true_branch(true_dynamics, cT, ct, 1.0);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> true_chain_solver({true_branch});

    std::vector<Branch> hindsight_branches; 
    hindsight_branches.reserve(possible_objects.size());
    for (size_t i = 0; i < possible_objects.size(); ++i)
    {
        hindsight_branches.emplace_back(dynamics_funcs[i], cT, ct, obs_probability[i]);
    }
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> hindsight_solver({hindsight_branches});

    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> *solver = nullptr; 
    switch(policy)
    {
    case PolicyTypes::TRUE_ILQR:
        solver = &true_chain_solver;
        break;
    case PolicyTypes::HINDSIGHT:
        solver = &hindsight_solver;
        break;
    case PolicyTypes::ARGMAX_ILQR:
        IS_TRUE(false); // unsupported for now
        //solver = &argmax_solver;
        break;
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
        IS_TRUE(false); // unsupported for now
        // do nothing as this requires two separate solvers
        break;
    };

    clock_t ilqr_begin_time = clock();
    solver->solve(T, x0, u_nominal, mu, max_iters_begin, 
            verbose, convg_thresh, start_alpha);
    PRINT("start: Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

    std::vector<Vector<pusher::Control::CONTROL_DIM>> controls;
    //return solver->forward_pass(0, x0, states, controls, 1.0); 

    pusher::PusherWorld test_world = true_world;
    bool uncertainty_resolved = false;

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

        rollout_cost += ct(xt, ut, t);
        const Vector<STATE_DIM> xt1 = true_dynamics(xt, ut);

        // We found the true object!
        if (xt1[State::OBJ_STUCK])
        {
            std::for_each(obs_probability.begin(), obs_probability.end(), [](double &v) { v = 0; } );
            obs_probability[true_obj_index] = 1;
            uncertainty_resolved = true;
        }
        if (!uncertainty_resolved)
        {
            // If we would have touched any of the obstacles, then we would observe
            // them.
            for (size_t i = 0; i < possible_objects.size(); ++i)
            {
                // If we have already elimated this option, then continue.
                if (obs_probability[i] == 0)
                {
                    continue;
                }

                const Eigen::Vector2d &obj_pos = possible_objects[i].position();
                auto test_dynamics = std::bind(pusher_dynamics, _1, _2, std::cref(obj_pos), std::ref(test_world));

                Vector<STATE_DIM> xt_test = xt;
                xt_test[State::OBJ_X] = obj_pos[0];
                xt_test[State::OBJ_Y] = obj_pos[1];
                Vector<STATE_DIM> xt1_test = test_dynamics(xt_test, ut);
                obs_probability[i] = 1;
                if (xt1_test[State::OBJ_STUCK] != xt1[State::OBJ_STUCK])
                {
                    obs_probability[i] = 0;
                }
            }
            // normalize the probabilities.
            const double Z = std::accumulate(obs_probability.begin(), obs_probability.end(), 0.0);
            std::for_each(obs_probability.begin(), obs_probability.end(), [Z](double &v) { v /= Z; } );
            const double Z_after = std::accumulate(obs_probability.begin(), obs_probability.end(), 0.0);
            IS_ALMOST_EQUAL(Z_after, 1.0, 1e-3);
        }
        IS_GREATER(obs_probability[true_obj_index], 0.0);
        

        xt = xt1;
        states.push_back(xt);
        
        switch(policy)
        {
        case PolicyTypes::TRUE_ILQR:
            break;
        case PolicyTypes::HINDSIGHT:
            for (size_t i = 0; i < possible_objects.size(); ++i)
            {
                solver->set_branch_probability(i, obs_probability[i]);
            }
            break;
        case PolicyTypes::ARGMAX_ILQR:
        {
            const int argmax_branch = get_argmax(obs_probability);
            const int other_branch = (argmax_branch == 0) ? 1 : 0;
            solver->set_branch_probability(argmax_branch, 1.0);
            solver->set_branch_probability(other_branch, 0.0);
            break;
        }
        case PolicyTypes::PROB_WEIGHTED_CONTROL:
            // do nothing as this requires two separate solvers
            break;
        };


        //const Eigen::Vector2d robot_position(xt[State::POS_X], xt[State::POS_Y]);
    }
    rollout_cost += cT(xt);
    DEBUG(" x_rollout(" << T-1 << ")= " << xt.transpose());
    DEBUG(" Total cost rollout: " << rollout_cost);

    return rollout_cost;
}

