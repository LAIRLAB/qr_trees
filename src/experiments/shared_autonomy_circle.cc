#include <experiments/shared_autonomy_circle.hh>

#include <experiments/simulators/user_goal.hh>
#include <experiments/simulators/directdrive.hh>
#include <experiments/simulators/circle_world.hh>
#include <templated/iLQR_hindsight_value.hh>
#include <utils/math_utils_temp.hh>
#include <utils/debug_utils.hh>
#include <utils/print_helpers.hh>
#include <utils/helpers.hh>

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

StateVector x0; // Start state for 0th timestep.
ControlVector u_nominal; 

int T;


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

    // Initial linearization points are linearly interpolated states and zero
    // control.
    u_nominal[0] = 0.0;
    u_nominal[1] = 0.0;

    const std::array<double, 2> CONTROL_LIMS = {{-5, 5}};
    simulators::directdrive::DirectDrive system(dt, CONTROL_LIMS, world_dims);

    auto dynamics = system;

    //construct goal objects, which includes their cost functions
    std::vector<user_goal::User_Goal> goals;
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

  


    //setup the branches for different policies
    std::vector<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM>> solver_branches;
    switch(policy)
    {
    case PolicyTypes::TRUE_ILQR:
    {
        solver_branches.emplace_back(dynamics, goals[true_goal_ind].cT_, goals[true_goal_ind].ct_, 1.0);
        break;
    }
    case PolicyTypes::HINDSIGHT:
    {
        for (auto &goal: goals)
        {
            solver_branches.emplace_back(dynamics, goal.cT_, goal.ct_, goal.prob_);
        }
        break;
    }
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
    {
        for (auto &goal: goals)
        {
            solver_branches.emplace_back(dynamics, goal.cT_, goal.ct_, 1.0);
        }
        // do nothing as this requires two separate solvers
        break;
    }
    case PolicyTypes::ARGMAX_ILQR:
    {
        size_t argmax_branch = std::distance(goal_priors.begin(), std::max_element(goal_priors.begin(), goal_priors.end()));
        for (size_t i=0; i < goals.size(); i++)
        {
            auto &goal = goals[i];
            if (i == argmax_branch)
                solver_branches.emplace_back(dynamics, goal.cT_, goal.ct_, 1.);
            else
                solver_branches.emplace_back(dynamics, goal.cT_, goal.ct_, 0.);
        }
        break;
    }
    };

    std::vector<ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM>> solvers; 

    switch(policy)
    {
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
        //for prob weighted, one solver per branch
        for (size_t i=0; i < solver_branches.size(); ++i)
        {
            std::vector<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> > onebranch_vec = {solver_branches[i]};
            solvers.emplace_back( onebranch_vec);
        }
        break;
    default:
        //otherwise, one solver for all branches
        solvers.emplace_back(solver_branches);
        break;
    };

    // Setup a solver representing the user
    ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> user_branch(dynamics, goals[true_goal_ind].cT_, goals[true_goal_ind].ct_, 1.0);
    ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM> user_solver({user_branch});


    constexpr bool verbose = false;
    constexpr int max_iters = 300;
    constexpr double mu = 0.25;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 1;

    
    std::vector<StateVector> states;

    filters::GoalPredictor goal_predictor(goal_priors);  

    clock_t ilqr_begin_time = clock();
    for (auto &solver: solvers)
    {
        solver.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    }
    user_solver.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    //PRINT("Pre-solve" << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

    double rollout_cost = 0;
    StateVector xt = x0;
    // store initial state
    states.push_back(xt);
    //TODO: How to run this for full T?
    for (int t = 0; t < T-1; ++t)
    {
        const int t_offset = t >  0 ? 1 : 0;
        const int plan_horizon = T-t;
        //const int plan_horizon = std::min(T-t, MPC_HORIZON);
        
        ControlVector ut;
        ControlVector user_ut;
        ilqr_begin_time = clock();

        if (policy == PolicyTypes::PROB_WEIGHTED_CONTROL)
        {
            //solve all 
            ut.setZero();
            for (size_t i=0; i < solvers.size(); i++)
            {
                solvers[i].solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
                ControlVector ut_this_solver = solvers[i].compute_first_control(xt);
                ut += goal_predictor.get_prob_at_ind(i)*ut_this_solver;
            }
        } else {
            //assume only one solver for these cases
            solvers[0].solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            ut = solvers[0].compute_first_control(xt); 
        }

        

        //PRINT("t=" << t << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

        rollout_cost += ct_true_world(xt, ut, t);
        const StateVector xt1 = dynamics(xt, ut);

        //get the user action, compute by solving for true chain
        user_solver.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
        user_ut = user_solver.compute_first_control(xt); 

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
                double v_xt, v_xt1;
                if (policy == PolicyTypes::PROB_WEIGHTED_CONTROL)
                {
                    v_xt = solvers[i].compute_value( 0, xt, 0);   
                    v_xt1 = solvers[i].compute_value( 0, xt1, 1);
                } else {
                    v_xt = solvers[0].compute_value( i, xt, 0);   
                    v_xt1 = solvers[0].compute_value( i, xt1, 1);
                }
                double c_xt = goals[i].ct_(xt, ut, t);
                double q_xt = c_xt + v_xt1;

                q_values[i] = q_xt;
                v_values[i] = v_xt;
            }
            //PRINT(q_values);
            //PRINT(v_values);

            goal_predictor.update_goal_distribution(q_values, v_values);
            std::vector<double> updated_goal_distribution = goal_predictor.get_goal_distribution();
            size_t argmax_goal = argmax(updated_goal_distribution);

            PRINT("goal probabilities: " << updated_goal_distribution);

            //update goal probabilities
            for (size_t i=0; i < NUM_GOALS; i++)
            { 
                goals[i].prob_ = updated_goal_distribution[i];
                if (policy == PolicyTypes::HINDSIGHT)
                {
                    solvers[0].set_branch_probability(i, updated_goal_distribution[i]);
                } else if (policy == PolicyTypes::ARGMAX_ILQR) {
                    if (i == argmax_goal)
                        solvers[0].set_branch_probability(i, 1.0);
                    else
                        solvers[0].set_branch_probability(i, 0.0);
                }
            }
                
        }

        xt = xt1;
        states.push_back(xt);



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

    states_to_file(x0, goals[true_goal_ind].goal_state_, states, state_output_fname);
    SUCCESS("Wrote states to: " << state_output_fname);

    obstacle_output_fname = obstacle_output_fname + "_" + obstacles_fname;
    obstacles_to_file(world, obstacle_output_fname);
    SUCCESS("Wrote obstacles to: " << obstacle_output_fname);
    
    return rollout_cost;

}

