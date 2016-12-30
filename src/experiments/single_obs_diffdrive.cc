#include <experiments/single_obs_diffdrive.hh>

#include <experiments/simulators/diffdrive.hh>
#include <experiments/simulators/circle_world.hh>
#include <templated/iLQR_hindsight.hh>
#include <utils/math_utils_temp.hh>
#include <utils/debug_utils.hh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace 
{

using CircleWorld = circle_world::CircleWorld;
using Circle = circle_world::Circle;

using State = simulators::diffdrive::State;
constexpr int STATE_DIM = simulators::diffdrive::STATE_DIM;
constexpr int CONTROL_DIM = simulators::diffdrive::CONTROL_DIM;
//constexpr int OBS_DIM = circle_world::OBSTACLE_DIM;

template <int rows>
using Vector = Eigen::Matrix<double, rows, 1>;

template <int rows, int cols>
using Matrix = Eigen::Matrix<double, rows, cols>;

double robot_radius = 3.35/2.0; // iRobot create;
double obstacle_factor = 300.0;
double scale_factor = 1.0e0;

Matrix<STATE_DIM, STATE_DIM> Q;
Matrix<STATE_DIM, STATE_DIM> QT; // Quadratic state cost for final timestep.
Matrix<CONTROL_DIM, CONTROL_DIM> R;
Vector<STATE_DIM> xT; // Goal state for final timestep.
Vector<STATE_DIM> x0; // Start state for 0th timestep.
Vector<CONTROL_DIM> u_nominal; 

int T;

double obstacle_cost(const CircleWorld &world, const double robot_radius, const Vector<STATE_DIM> &xt)
{
    Eigen::Vector2d robot_pos;
    robot_pos << xt[State::POS_X], xt[State::POS_Y];

    // Compute minimum distance to the edges of the world.
    double cost = 0;

    auto obstacles = world.obstacles();
	for (size_t i = 0; i < obstacles.size(); ++i) {
		Vector<2> d = robot_pos - obstacles[i].position();
		double distr = d.norm(); 
		double dist = distr - robot_radius - obstacles[i].radius();
		cost += obstacle_factor * exp(-scale_factor*dist);
	}
    return cost;
}

double ct(const Vector<STATE_DIM> &x, const Vector<CONTROL_DIM> &u, const int t, const CircleWorld &world)
{
    double cost = 0;

    // position
    if (t == 0)
    {
        Vector<STATE_DIM> dx = x - x0;
        cost += 0.5*(dx.transpose()*Q*dx)[0];
    }

    // Control cost
    const Vector<CONTROL_DIM> du = u - u_nominal;
    cost += 0.5*(du.transpose()*R*du)[0];

    cost += 10*x[State::dTHETA]*x[State::dTHETA];
    cost += 0.05*x[State::dV_LEFT]*x[State::dV_LEFT];
    cost += 0.05*x[State::dV_RIGHT]*x[State::dV_RIGHT];

    cost += obstacle_cost(world, robot_radius, x);

    return cost;
}

// Final timestep cost function
double cT(const Vector<STATE_DIM> &x)
{
    const Vector<STATE_DIM> dx = x - xT;
    return 0.5*(dx.transpose()*QT*dx)[0];
}

void states_to_file(const Vector<STATE_DIM>& x0, const Vector<STATE_DIM>& xT, 
        const std::vector<Vector<STATE_DIM>> &states, 
        const std::string &fname)
{
    std::ofstream file(fname, std::ofstream::trunc | std::ofstream::out);
    auto print_vector = [&file](const Vector<STATE_DIM> &x)
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

double control_diffdrive(const PolicyTypes policy,
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

	xT = Vector<STATE_DIM>::Zero();
	xT[State::POS_X] = 0;
	xT[State::POS_Y] = 25;
	xT[State::THETA] = M_PI/2; 
	xT[State::dTHETA] = 0;

	x0 = Vector<STATE_DIM>::Zero();
	x0[State::POS_X] = 0;
	x0[State::POS_Y] = -25;
	x0[State::THETA] = M_PI/2;
	x0[State::dTHETA] = 0;

	Q = 1*Matrix<STATE_DIM,STATE_DIM>::Identity();
	const double rot_cost = 0.5;
    Q(State::THETA, State::THETA) = rot_cost;
    Q(State::dV_LEFT, State::dV_LEFT) = 0.1;

    QT = 25*Matrix<STATE_DIM,STATE_DIM>::Identity();
    QT(State::THETA, State::THETA) = 50.0;
    QT(State::dTHETA, State::dTHETA) = 5.0;
    QT(State::dV_LEFT, State::dV_LEFT) = 5.0;
    QT(State::dV_RIGHT, State::dV_RIGHT) = 5.0;

	R = 2*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();

    // Initial linearization points are linearly interpolated states and zero
    // control.
    u_nominal[0] = 2.5;
    u_nominal[1] = 2.5;

    const std::array<double, 2> CONTROL_LIMS = {{-5, 5}};
    simulators::diffdrive::DiffDrive system(dt, CONTROL_LIMS, world_dims);

    auto dynamics = system;

    constexpr bool verbose = false;
    constexpr int max_iters = 300;
    constexpr double mu = 0.80;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 1;

    // Setup the cost function with environments that have the obstacle and one
    // that does not.
    auto ct_true_world = std::bind(ct, _1, _2, _3, true_world);
    auto ct_other_world = std::bind(ct, _1, _2, _3, other_world);
    
    std::array<double, 2> obs_probability = OBS_PRIOR;

    // Setup the true system solver.
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> true_branch(dynamics, cT, ct_true_world, 1.0);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> true_chain_solver({true_branch});

    // Setup the "our method" hindsight optimization approach.
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> hindsight_world_1(dynamics, cT, ct_true_world, obs_probability[0]);
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> hindsight_world_2(dynamics, cT, ct_other_world, obs_probability[1]);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> hindsight_solver({hindsight_world_1, hindsight_world_2});

    // The argmax approach.
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> argmax_world_1(dynamics, cT, ct_true_world, obs_probability[0]);
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> argmax_world_2(dynamics, cT, ct_other_world, obs_probability[1]);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> argmax_solver({argmax_world_1, argmax_world_2});
    const int argmax_branch = get_argmax(obs_probability);
    const int other_branch = (argmax_branch == 0) ? 1 : 0;
    argmax_solver.set_branch_probability(argmax_branch, 1.0);
    argmax_solver.set_branch_probability(other_branch, 0.0);

    // Weighted control approach.
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> branch_world_1(dynamics, cT, ct_true_world, 1.0);
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> branch_world_2(dynamics, cT, ct_other_world, 1.0);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> weighted_cntrl_world_1({branch_world_1});
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> weighted_cntrl_world_2({branch_world_2});

    // TODO need default constructor or something to set it to.
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
        solver = &argmax_solver;
        break;
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
        // do nothing as this requires two separate solvers
        break;
    };


    std::vector<Vector<STATE_DIM>> states;

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
    Vector<STATE_DIM> xt = x0;
    // store initial state
    states.push_back(xt);
    //TODO: How to run this for full T?
    for (int t = 0; t < T-1; ++t)
    {
        const bool t_offset = t >  0 ? 1 : 0;
        const int plan_horizon = T-t;
        //const int plan_horizon = std::min(T-t, MPC_HORIZON);
        
        Vector<CONTROL_DIM> ut;
        ilqr_begin_time = clock();
        if (policy == PolicyTypes::PROB_WEIGHTED_CONTROL)
        {
            weighted_cntrl_world_1.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            weighted_cntrl_world_2.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            const Vector<CONTROL_DIM> ut_with_obs = weighted_cntrl_world_1.compute_first_control(xt); 
            const Vector<CONTROL_DIM> ut_without_obs = weighted_cntrl_world_2.compute_first_control(xt); 
            ut = obs_probability[0] * ut_with_obs + obs_probability[1] * ut_without_obs ;
        }
        else
        {
            solver->solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            ut = solver->compute_first_control(xt); 
        }
        //PRINT("t=" << t << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

        rollout_cost += ct_true_world(xt, ut, t);
        const Vector<STATE_DIM> xt1 = dynamics(xt, ut);

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

double single_obs_control_diffdrive(const PolicyTypes policy,
        bool true_world_with_obs,
        const std::array<double, 2> &OBS_PRIOR,
        std::string &state_output_fname,
        std::string &obstacle_output_fname
        )
{
    std::array<double, 4> world_dims = {{-30, 30, -30, 30}};
    CircleWorld true_world(world_dims);
    CircleWorld other_world(world_dims);

    const Eigen::Vector2d obstacle_pos(0, 0.0);
	constexpr double obs_radius = 5.0;
    if (true_world_with_obs)
    {
        true_world.add_obstacle(obs_radius, obstacle_pos);
    }
    else
    {
        other_world.add_obstacle(obs_radius, obstacle_pos);
    }
        
    const double cost_to_go = control_diffdrive(policy, true_world, other_world, 
            OBS_PRIOR, state_output_fname, obstacle_output_fname); 
    return cost_to_go;
}
