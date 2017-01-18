#include <experiments/shared_autonomy_circle.hh>

#include <experiments/simulators/directdrive.hh>
#include <experiments/simulators/circle_world.hh>
#include <templated/iLQR_hindsight_value.hh>
#include <utils/math_utils_temp.hh>
#include <utils/debug_utils.hh>
#include <utils/print_helpers.hh>

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

double ct(const StateVector &x, const ControlVector &u, const int t, const CircleWorld &world, const StateVector& goal_state);
double cT(const StateVector &x, const StateVector& goal_state);

struct Goal {
    
    Goal(const StateVector& goal_state, double prob, const CircleWorld& world)
        : goal_state_(goal_state), prob_(prob), cT_(std::bind(cT, std::placeholders::_1, goal_state)), ct_(std::bind(ct, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, world, goal_state))
    {}

    StateVector goal_state_;
    double prob_;
    std::function<double(const StateVector&)> cT_;
    std::function<double(const StateVector&, const ControlVector&, const int)> ct_;
};

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

    cost += obstacle_cost(world, robot_radius, x);

    return cost;
}

// Final timestep cost function
double cT(const StateVector &x, const StateVector& goal_state)
{
    const StateVector dx = x - goal_state;
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


} // namespace

double control_shared_autonomy(const PolicyTypes policy,
        const CircleWorld &world,
        const std::vector<StateVector>& goal_states,
        const std::vector<double>& goal_priors,
        const int true_goal_ind,
        std::string &state_output_fname,
        std::string &obstacle_output_fname
        )
{
    using namespace std::placeholders;

    const std::string states_fname = "states.csv";
    const std::string obstacles_fname = "obstacles.csv";

    const std::array<double, 4> world_dims = world.dimensions();
    IS_TRUE(std::equal(world_dims.begin(), world_dims.begin(), world.dimensions().begin()));

    const size_t NUM_GOALS = goal_states.size();

    // Currently each can only have 1 obstacle.
    IS_LESS_EQUAL(world.obstacles().size(), 1);
    

    T = 50;
    const double dt = 1.0/6.0;
    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    DEBUG("Running with policy \"" << to_string(policy)
            << "\" with num obs in true= \"" 
            << world.obstacles().size() << "\"");

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

    //construct goal objects, which includes their cost functions
    std::vector<Goal> goals;
    for (size_t i=0; i < goal_states.size(); ++i)
    {
        goals.emplace_back(goal_states[i], goal_priors[i], world);
    }

    // set the true cost function based on user goal ind
    auto ct_true_world = goals[true_goal_ind].ct_;
    auto cT_true_world = goals[true_goal_ind].cT_;

//    std::vector<std::function<decltype(std::bind(cT, _1, goal_states[0]))> > goal_cT;
//    for (size_t i=0; i < goal_states.size(); i++)
//    {
//        goal_cT.emplace_back(std::bind(cT, _1, goal_states[i]));
//    }

   
    // Setup the true system solver.
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> true_branch(dynamics, goals[true_goal_ind].cT_, goals[true_goal_ind].ct_, 1.0);
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> true_chain_solver({true_branch});

    // Setup the "our method" hindsight optimization approach.
    std::vector<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM>> hindsight_worlds;
    for (auto &goal: goals)
    {
        hindsight_worlds.emplace_back(dynamics, goal.cT_, goal.ct_, goal.prob_);
    }
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> hindsight_solver(hindsight_worlds);

    // The argmax approach.
    std::vector<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM>> argmax_worlds;
    size_t argmax_branch = std::distance(goal_priors.begin(), std::max_element(goal_priors.begin(), goal_priors.end()));
    for (size_t i=0; i < goals.size(); i++)
    {
        auto &goal = goals[i];
        if (i == argmax_branch)
            argmax_worlds.emplace_back(dynamics, goal.cT_, goal.ct_, 1.);
        else
            argmax_worlds.emplace_back(dynamics, goal.cT_, goal.ct_, 0.);
    }
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> argmax_solver(argmax_worlds);

    // Weighted control approach.
//    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> branch_world_1(dynamics, cT, ct_true_world, 1.0);
//    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> branch_world_2(dynamics, cT, ct_other_world, 1.0);
//    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> weighted_cntrl_world_1({branch_world_1});
//    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> weighted_cntrl_world_2({branch_world_2});


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

    constexpr bool verbose = false;
    constexpr int max_iters = 300;
    constexpr double mu = 0.25;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 1;

    
    std::vector<StateVector> states;

    filters::GoalPredictor goal_predictor(goal_priors);  

    clock_t ilqr_begin_time = clock();
    if (policy == PolicyTypes::PROB_WEIGHTED_CONTROL)
    {
//        weighted_cntrl_world_1.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
//        weighted_cntrl_world_2.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    }
    else
    {
        solver->solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    }
    true_chain_solver.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
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
        ControlVector user_ut;
        ilqr_begin_time = clock();
        if (policy == PolicyTypes::PROB_WEIGHTED_CONTROL)
        {
//            weighted_cntrl_world_1.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
//            weighted_cntrl_world_2.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
//            const ControlVector ut_with_obs = weighted_cntrl_world_1.compute_first_control(xt); 
//            const ControlVector ut_without_obs = weighted_cntrl_world_2.compute_first_control(xt); 
//            ut = obs_probability[0] * ut_with_obs + obs_probability[1] * ut_without_obs ;
        }
        else
        {
            solver->solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            ut = solver->compute_first_control(xt); 
        }

        //PRINT("t=" << t << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

        rollout_cost += ct_true_world(xt, ut, t);
        const StateVector xt1 = dynamics(xt, ut);


        //get the user action, compute by solving for true chain
        true_chain_solver.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
        user_ut = true_chain_solver.compute_first_control(xt); 

        //update predictor probabilities
        std::vector<double> q_values(NUM_GOALS);
        std::vector<double> v_values(NUM_GOALS);
        if (policy != PolicyTypes::TRUE_ILQR)
        {
            //if not true ILQR, update the distribution
            std::vector<double> q_values(NUM_GOALS);
            std::vector<double> v_values(NUM_GOALS);
            for (size_t i=0; i < NUM_GOALS; i++)
            {
                double v_xt = solver->compute_value( i, xt, t);
                double v_xt1 = solver->compute_value( i, xt1, t+1);
                double c_xt = goals[i].ct_(xt, ut, t);
                double q_xt = c_xt + v_xt1;

                q_values[i] = q_xt;
                v_values[i] = v_xt;
            }
            //PRINT(q_values);
            //PRINT(v_values);

            goal_predictor.update_goal_distribution(q_values, v_values);
            std::vector<double> updated_goal_distribution = goal_predictor.get_goal_distribution();
            PRINT(updated_goal_distribution);

            for (size_t i=0; i < NUM_GOALS; i++)
            {
                goals[i].prob_ = updated_goal_distribution[i];
                solver->set_branch_probability(i, updated_goal_distribution[i]);
            }  //TODO handle argmax and branched seperately
            //maybe make function that sets all goal probabilities?
                
        }

        xt = xt1;
        states.push_back(xt);


        // Update parts required based on policy.
//        switch(policy)
//        {
//        case PolicyTypes::TRUE_ILQR:
//            break;
//        case PolicyTypes::HINDSIGHT:
//            solver->set_branch_probability(0, obs_probability[0]);
//            solver->set_branch_probability(1, obs_probability[1]);
//            break;
//        case PolicyTypes::ARGMAX_ILQR:
//        {
//            const int argmax_branch = get_argmax(obs_probability);
//            const int other_branch = (argmax_branch == 0) ? 1 : 0;
//            solver->set_branch_probability(argmax_branch, 1.0);
//            solver->set_branch_probability(other_branch, 0.0);
//            break;
//        }
//        case PolicyTypes::PROB_WEIGHTED_CONTROL:
//            // do nothing as this requires two separate solvers
//            break;
//        };

    }
    rollout_cost += cT_true_world(xt);
    DEBUG(" x_rollout(" << T-1 << ")= " << xt.transpose());
    DEBUG(" Total cost rollout: " << rollout_cost);

    std::string policy_fname = states_fname;
    switch(policy)
    {
    case PolicyTypes::TRUE_ILQR:
        policy_fname = "ilqr_true";
        break;
    case PolicyTypes::HINDSIGHT:
        policy_fname = "hindsight";// + std::to_string(static_cast<int>(OBS_PRIOR[0]*100)) + "-" + std::to_string(static_cast<int>(OBS_PRIOR[1]*100));
        break;
    case PolicyTypes::ARGMAX_ILQR:
    {
        policy_fname = "argmax";// + std::to_string(static_cast<int>(OBS_PRIOR[0]*100)) + "-" + std::to_string(static_cast<int>(OBS_PRIOR[1]*100));
        break;
    }
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
        // do nothing as this requires two separate solvers
        policy_fname = "weighted";// + std::to_string(static_cast<int>(OBS_PRIOR[0]*100)) + "-" + std::to_string(static_cast<int>(OBS_PRIOR[1]*100));
        break;
    };

    state_output_fname = state_output_fname + "_" +  policy_fname + "_" + states_fname;

    states_to_file(x0, xT, states, state_output_fname);
    SUCCESS("Wrote states to: " << state_output_fname);

    obstacle_output_fname = obstacle_output_fname + "_" + obstacles_fname;
    obstacles_to_file(world, obstacle_output_fname);
    SUCCESS("Wrote obstacles to: " << obstacle_output_fname);
    
    return rollout_cost;

}

