#include <experiments/shared_autonomy_circle.hh>

#include <experiments/simulators/directdrive.hh>
#include <experiments/simulators/circle_world.hh>
#include <templated/iLQR_hindsight_value.hh>
#include <utils/math_utils_temp.hh>
#include <utils/debug_utils.hh>

#include <filters/goal_predictor.hh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace 
{

using CircleWorld = circle_world::CircleWorld;
using Circle = circle_world::Circle;

using State = simulators::directdrive::State;
constexpr int STATE_DIM = simulators::directdrive::STATE_DIM;
constexpr int CONTROL_DIM = simulators::directdrive::CONTROL_DIM;
//constexpr int OBS_DIM = circle_world::OBSTACLE_DIM;

template <int rows>
using Vector = Eigen::Matrix<double, rows, 1>;

template <int rows, int cols>
using Matrix = Eigen::Matrix<double, rows, cols>;

using StateVector = simulators::directdrive::StateVector;
using ControlVector = simulators::directdrive::ControlVector;

double robot_radius = 3.35/2.0; // iRobot create;
double obstacle_factor = 300.0;
double scale_factor = 1.0e0;

Matrix<STATE_DIM, STATE_DIM> Q;
Matrix<STATE_DIM, STATE_DIM> QT; // Quadratic state cost for final timestep.
Matrix<CONTROL_DIM, CONTROL_DIM> R;
StateVector xT; // Goal state for final timestep.
StateVector x0; // Start state for 0th timestep.
ControlVector u_nominal; 

int T;

double obstacle_cost(const CircleWorld &world, const double robot_radius, const StateVector &xt)
{
    Eigen::Vector2d robot_pos;
    robot_pos << xt[State::POS_X], xt[State::POS_Y];

    // Compute minimum distance to the edges of the world.
    double cost = 0;

    auto obstacles = world.obstacles();
    for (size_t i = 0; i < obstacles.size(); ++i) {
        Eigen::Vector2d d = robot_pos - obstacles[i].position();
        double distr = d.norm(); 
        double dist = distr - robot_radius - obstacles[i].radius();
        cost += obstacle_factor * exp(-scale_factor*dist);
    }
    return cost;
}

double ct(const StateVector &x, const ControlVector &u, const int t, const CircleWorld &world)
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

    cost += obstacle_cost(world, robot_radius, x);

    return cost;
}

// Final timestep cost function
double cT(const StateVector &x)
{
    const StateVector dx = x - xT;
    return 0.5*(dx.transpose()*QT*dx)[0];
}

void states_to_file(const StateVector& x0, const StateVector& xT, 
        const std::vector<StateVector> &states, 
        const std::string &fname)
{
    std::ofstream file(fname, std::ofstream::trunc | std::ofstream::out);
    auto print_vector = [&file](const StateVector &x)
    {
        constexpr int PRINT_WIDTH = 13;
        constexpr char DELIMITER[] = " ";

        for (int i = 0; i < STATE_DIM; ++i)
        {
            file << std::left << std::setw(PRINT_WIDTH) << x[i] << DELIMITER;
        }
        file << std::endl;
    };
    print_vector(x0); 
    print_vector(xT);
    for (const auto& state : states)
    {
        print_vector(state);
    }
    file.close();
}

void obstacles_to_file(const CircleWorld &world, const std::string &fname)
{
    std::ofstream file(fname, std::ofstream::trunc | std::ofstream::out);
    IS_TRUE(file.is_open());
    file << world;
    file.close();
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

double control_shared_autonomy(const PolicyTypes policy,
        const CircleWorld &true_world,
        const CircleWorld &other_world,
        const std::array<double, 2> &OBS_PRIOR,
        std::string &state_output_fname,
        std::string &obstacle_output_fname
        )
{
    using namespace std::placeholders;

    const std::string states_fname = "states.csv";
    const std::string obstacles_fname = "obstacles.csv";

    const std::array<double, 4> world_dims = true_world.dimensions();
    IS_TRUE(std::equal(world_dims.begin(), world_dims.begin(), true_world.dimensions().begin()));

    // Currently each can only have 1 obstacle.
    IS_LESS_EQUAL(true_world.obstacles().size(), 1);
    IS_LESS_EQUAL(other_world.obstacles().size(), 1);
    

    T = 50;
	const double dt = 1.0/6.0;
    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    DEBUG("Running with policy \"" << to_string(policy)
            << "\" with num obs in true= \"" 
            << true_world.obstacles().size() << "\"");

	xT = StateVector::Zero();
	xT[State::POS_X] = 0;
	xT[State::POS_Y] = 25;

	x0 = StateVector::Zero();
	x0[State::POS_X] = 0;
	x0[State::POS_Y] = -25;

	Q = 1*Matrix<STATE_DIM,STATE_DIM>::Identity();
//	const double rot_cost = 0.5;
//    Q(State::THETA, State::THETA) = rot_cost;
//    Q(State::dV_LEFT, State::dV_LEFT) = 0.1;
//
  QT = 25*Matrix<STATE_DIM,STATE_DIM>::Identity();
//    QT(State::THETA, State::THETA) = 50.0;
//    QT(State::dTHETA, State::dTHETA) = 5.0;
//    QT(State::dV_LEFT, State::dV_LEFT) = 5.0;
//    QT(State::dV_RIGHT, State::dV_RIGHT) = 5.0;

	R = 2*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();

    // Initial linearization points are linearly interpolated states and zero
    // control.
    u_nominal[0] = 0.0;
    u_nominal[1] = 0.0;

    const std::array<double, 2> CONTROL_LIMS = {{-5, 5}};
    simulators::directdrive::DirectDrive system(dt, CONTROL_LIMS, world_dims);

    auto dynamics = system;

    constexpr bool verbose = false;
    constexpr int max_iters = 300;
    constexpr double mu = 0.25;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 1;

    // Setup the cost function with different environments
    auto ct_true_world = std::bind(ct, _1, _2, _3, true_world);
    auto ct_other_world = std::bind(ct, _1, _2, _3, other_world);
    
    std::array<double, 2> obs_probability = OBS_PRIOR;

    // Setup the true system solver.
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> true_branch(dynamics, cT, ct_true_world, 1.0);
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> true_chain_solver({true_branch});

    // Setup the "our method" hindsight optimization approach.
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> hindsight_world_1(dynamics, cT, ct_true_world, obs_probability[0]);
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> hindsight_world_2(dynamics, cT, ct_other_world, obs_probability[1]);
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> hindsight_solver({hindsight_world_1, hindsight_world_2});

    // The argmax approach.
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> argmax_world_1(dynamics, cT, ct_true_world, obs_probability[0]);
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> argmax_world_2(dynamics, cT, ct_other_world, obs_probability[1]);
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> argmax_solver({argmax_world_1, argmax_world_2});
    const int argmax_branch = get_argmax(obs_probability);
    const int other_branch = (argmax_branch == 0) ? 1 : 0;
    argmax_solver.set_branch_probability(argmax_branch, 1.0);
    argmax_solver.set_branch_probability(other_branch, 0.0);

    // Weighted control approach.
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> branch_world_1(dynamics, cT, ct_true_world, 1.0);
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> branch_world_2(dynamics, cT, ct_other_world, 1.0);
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> weighted_cntrl_world_1({branch_world_1});
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> weighted_cntrl_world_2({branch_world_2});

    // TODO need default constructor or something to set it to.
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> *solver = nullptr; 
    switch(policy)
    {
    case PolicyTypes::TRUE_ILQR:
        solver = &true_chain_solver;
        break;
    case PolicyTypes::HINDSIGHT:
        solver = &hindsight_solver;
        break;
    case PolicyTypes::ARGMAX_ILQR:
        solver = &argmax_solver;
        break;
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
        // do nothing as this requires two separate solvers
        break;
    };


    std::vector<StateVector> states;

    clock_t ilqr_begin_time = clock();
    if (policy == PolicyTypes::PROB_WEIGHTED_CONTROL)
    {
        weighted_cntrl_world_1.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
        weighted_cntrl_world_2.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    }
    else
    {
        solver->solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    }
    //PRINT("Pre-solve" << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

    double rollout_cost = 0;
    StateVector xt = x0;
    // store initial state
    states.push_back(xt);
    //TODO: How to run this for full T?
    for (int t = 0; t < T-1; ++t)
    {
        const bool t_offset = t >  0 ? 1 : 0;
        const int plan_horizon = T-t;
        //const int plan_horizon = std::min(T-t, MPC_HORIZON);
        
        ControlVector ut;
        ilqr_begin_time = clock();
        if (policy == PolicyTypes::PROB_WEIGHTED_CONTROL)
        {
            weighted_cntrl_world_1.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            weighted_cntrl_world_2.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            const ControlVector ut_with_obs = weighted_cntrl_world_1.compute_first_control(xt); 
            const ControlVector ut_without_obs = weighted_cntrl_world_2.compute_first_control(xt); 
            ut = obs_probability[0] * ut_with_obs + obs_probability[1] * ut_without_obs ;
        }
        else
        {
            solver->solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            ut = solver->compute_first_control(xt); 
        }
        //PRINT("t=" << t << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

        rollout_cost += ct_true_world(xt, ut, t);
        const StateVector xt1 = dynamics(xt, ut);

        xt = xt1;
        states.push_back(xt);

        const Eigen::Vector2d robot_position(xt[State::POS_X], xt[State::POS_Y]);
        //int num_true_obs_seen = 0;
        for (const auto obstacle : true_world.obstacles())
        {
            const double net_distance = (robot_position - obstacle.position()).norm() 
                                        - robot_radius - obstacle.radius();
            // If we see the obstacle, then we know this world is probably true.
            if (net_distance < 2)
            {
                //++num_true_obs_seen;
                obs_probability[0] = 1.0;
                obs_probability[1] = 0.0;
            }
        }
        //int num_other_obs_seen = 0; // TODO can maybe use this with the discrete filter?
        for (const auto obstacle : other_world.obstacles())
        {
            const double net_distance = (robot_position - obstacle.position()).norm() 
                                        - robot_radius - obstacle.radius();
            // If we get close to the the obstacle in the other_world, then we know the other world is probably 
            // false since the "sensor" didn't pick it up.
            if (net_distance < 2)
            {
                //++num_other_obs_seen;
                obs_probability[0] = 1.0;
                obs_probability[1] = 0.0;
            }
        }

        // Update parts required based on policy.
        switch(policy)
        {
        case PolicyTypes::TRUE_ILQR:
            break;
        case PolicyTypes::HINDSIGHT:
            solver->set_branch_probability(0, obs_probability[0]);
            solver->set_branch_probability(1, obs_probability[1]);
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

    }
    rollout_cost += cT(xt);
    DEBUG(" x_rollout(" << T-1 << ")= " << xt.transpose());
    DEBUG(" Total cost rollout: " << rollout_cost);

    std::string policy_fname = states_fname;
    switch(policy)
    {
    case PolicyTypes::TRUE_ILQR:
        policy_fname = "ilqr_true";
        break;
    case PolicyTypes::HINDSIGHT:
        policy_fname = "hindsight_" + std::to_string(static_cast<int>(OBS_PRIOR[0]*100)) + "-" + std::to_string(static_cast<int>(OBS_PRIOR[1]*100));
        break;
    case PolicyTypes::ARGMAX_ILQR:
    {
        policy_fname = "argmax_" + std::to_string(static_cast<int>(OBS_PRIOR[0]*100)) + "-" + std::to_string(static_cast<int>(OBS_PRIOR[1]*100));
        break;
    }
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
        // do nothing as this requires two separate solvers
        policy_fname = "weighted_" + std::to_string(static_cast<int>(OBS_PRIOR[0]*100)) + "-" + std::to_string(static_cast<int>(OBS_PRIOR[1]*100));
        break;
    };

    state_output_fname = state_output_fname + "_" +  policy_fname + "_" + states_fname;

    states_to_file(x0, xT, states, state_output_fname);
    SUCCESS("Wrote states to: " << state_output_fname);

    obstacle_output_fname = obstacle_output_fname + "_" + obstacles_fname;
    obstacles_to_file(true_world, obstacle_output_fname);
    SUCCESS("Wrote obstacles to: " << obstacle_output_fname);
    
    return rollout_cost;
}

