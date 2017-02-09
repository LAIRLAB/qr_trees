//
// Shervin Javdani (sjavdani@cs.cmu.edu)
// January 2017
//


#include <experiments/shared_autonomy_circle_class.hh>

#include <utils/math_utils_temp.hh>
#include <utils/debug_utils.hh>
#include <utils/print_helpers.hh>
#include <utils/helpers.hh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace experiments
{

const double dt = 1.0/6.0;
const std::array<double, 2> CONTROL_LIMS = {{-5, 5}};

constexpr bool verbose = false;
constexpr int max_iters = 300;
constexpr double mu = 0.25;
constexpr double convg_thresh = 1e-4;
constexpr double start_alpha = 1;



SharedAutonomyCircle::SharedAutonomyCircle(const PolicyTypes policy, const CircleWorld &world, const std::vector<StateVector>& goal_states, const std::vector<double>& goal_priors, const int true_goal_ind, const int timesteps)
    : policy_type_(policy), world_(world), true_goal_ind_(true_goal_ind), timesteps_(timesteps), dynamics_(dt, CONTROL_LIMS, world.dimensions()), goal_predictor_(goal_priors), NUM_GOALS(goal_states.size())
{
    using namespace std::placeholders;

    const std::array<double, 4> world_dims = world.dimensions();
    IS_TRUE(std::equal(world_dims.begin(), world_dims.begin(), world.dimensions().begin()));

    // Currently each can only have 1 obstacle.
    IS_LESS_EQUAL(world.obstacles().size(), 1);

    IS_GREATER(timesteps_, 1);
    IS_GREATER(dt, 0);

    DEBUG("Initializing with policy \"" << to_string(policy)
            << "\" with num obs in true= \"" 
            << world.obstacles().size() << "\"");

    //initialize first state
    states_.push_back(StateVector::Zero());
    states_[0][State::POS_X] = 0;
    states_[0][State::POS_Y] = -25;

    // Initial linearization points are linearly interpolated states and zero
    // control.
    u_nominal[0] = 0.0;
    u_nominal[1] = 0.0;

    //simulators::directdrive::DirectDrive system(dt, CONTROL_LIMS, world_dims);

    //construct goal objects, which includes their cost functions
    for (size_t i=0; i < goal_states.size(); ++i)
    {
        goals_.emplace_back(goal_states[i], goal_priors[i], world);
    }

    // set the true cost function based on user goal ind


    //construct the branches
    switch(policy_type_)
    {
    case PolicyTypes::TRUE_ILQR:
    {
        solver_branches_.emplace_back(dynamics_, goals_[true_goal_ind_].cT_, goals_[true_goal_ind_].ct_, 1.0);
        break;
    }
    case PolicyTypes::HINDSIGHT:
    {
        for (auto &goal: goals_)
        {
            solver_branches_.emplace_back(dynamics_, goal.cT_, goal.ct_, goal.prob_);
        }
        break;
    }
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
    {
        for (auto &goal: goals_)
        {
            solver_branches_.emplace_back(dynamics_, goal.cT_, goal.ct_, 1.0);
        }
        // do nothing as this requires two separate solvers
        break;
    }
    case PolicyTypes::AVG_COST:
    {
        solver_branches_.emplace_back(dynamics_, avg_final_cost(goals_), avg_cost(goals_), 1.0);
        break;
    }
    case PolicyTypes::ARGMAX_ILQR:
    {
        size_t argmax_branch = std::distance(goal_priors.begin(), std::max_element(goal_priors.begin(), goal_priors.end()));
        for (size_t i=0; i < goals_.size(); i++)
        {
            auto &goal = goals_[i];
            if (i == argmax_branch)
                solver_branches_.emplace_back(dynamics_, goal.cT_, goal.ct_, 1.);
            else
                solver_branches_.emplace_back(dynamics_, goal.cT_, goal.ct_, 0.);
        }
        break;
    }
    };

    //construct the tree solvers from branches
    switch(policy_type_)
    {
    case PolicyTypes::PROB_WEIGHTED_CONTROL:
        //for prob weighted, one solver per branch
        for (size_t i=0; i < solver_branches_.size(); ++i)
        {
            std::vector<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> > onebranch_vec = {solver_branches_[i]};
            solvers_.emplace_back( onebranch_vec);
        }
        break;
    default:
        //otherwise, one solver for all branches
        solvers_.emplace_back(solver_branches_);
        break;
    };

    //solve initial problem
    for (auto &solver: solvers_)
    {
        solver.solve(timesteps_, states_.back(), u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    }

    //construct the prediction solvers, one per goal
    for (auto &goal: goals_)
    {
        prediction_solver_branches_.emplace_back(dynamics_, goal.cT_, goal.ct_, 1.0);
        std::vector<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM> > onebranch_vec = {prediction_solver_branches_.back()};
        prediction_solvers_.emplace_back( onebranch_vec);
        prediction_solvers_.back().solve(timesteps_, states_.back(), u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    }

    //construct a solver representing the user
    user_solver_branch_.reset(new ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM>(dynamics_, goals_[true_goal_ind_].cT_, goals_[true_goal_ind_].ct_, 1.0));
    user_solver_.reset(new ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM>({*user_solver_branch_}));
    user_solver_->solve(timesteps_, states_.back(), u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    
}

void SharedAutonomyCircle::run_control(int num_timesteps)
{
    const int loop_limit = std::min(get_num_timesteps_remaining(), num_timesteps) + current_timestep_;

    DEBUG("Running up to timestep " << loop_limit << ". Currently at timestep " << current_timestep_);
    for (; current_timestep_ < loop_limit; ++current_timestep_)
    {
        const int t_offset = current_timestep_ >  0 ? 1 : 0;
        const int plan_horizon = get_num_timesteps_remaining();
        //const int plan_horizon = std::min(T-t, MPC_HORIZON);
        
        ControlVector ut;
        ControlVector user_ut;
        StateVector xt = get_last_state();

        // solve user action
        user_solver_->solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
        user_ut = user_solver_->compute_first_control(xt);
        const StateVector user_xt1 = dynamics_(xt, user_ut);


        //update predictor probabilities
        std::vector<double> q_values(NUM_GOALS);
        std::vector<double> v_values(NUM_GOALS);
        //for (auto& solver: prediction_solvers_)
        for (size_t i=0; i < NUM_GOALS; i++)
        {
            prediction_solvers_[i].solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            double v_xt = prediction_solvers_[i].compute_value( 0, xt, 0);   
            double v_xt1 = prediction_solvers_[i].compute_value( 0, user_xt1, 1);
            double c_xt = goals_[i].ct_(xt, user_ut, current_timestep_);
            double q_xt = c_xt + v_xt1;

            q_values[i] = q_xt;
            v_values[i] = v_xt;
        }

        //PRINT(q_values);
        //PRINT(v_values);

        goal_predictor_.update_goal_distribution(q_values, v_values, 0.00001);
        std::vector<double> updated_goal_distribution = goal_predictor_.get_goal_distribution();
        size_t argmax_goal = argmax(updated_goal_distribution);

        DEBUG("iter " << current_timestep_ << " goal probabilities: " << updated_goal_distribution);

        //update goal probabilities
        for (size_t i=0; i < NUM_GOALS; i++)
        { 
            goals_[i].prob_ = updated_goal_distribution[i];
            if (policy_type_ == PolicyTypes::HINDSIGHT)
            {
                solvers_[0].set_branch_probability(i, updated_goal_distribution[i]);
            } else if (policy_type_ == PolicyTypes::ARGMAX_ILQR) {
                if (i == argmax_goal)
                    solvers_[0].set_branch_probability(i, 1.0);
                else
                    solvers_[0].set_branch_probability(i, 0.0);
            }
        }
                

        if (policy_type_ == PolicyTypes::PROB_WEIGHTED_CONTROL)
        {
            //solve all 
            ut.setZero();
            for (size_t i=0; i < solvers_.size(); i++)
            {
                solvers_[i].solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
                ControlVector ut_this_solver = solvers_[i].compute_first_control(xt);
                ut += goal_predictor_.get_prob_at_ind(i)*ut_this_solver;
            }
        } else {
            //assume only one solver for these cases
            solvers_[0].solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
            ut = solvers_[0].compute_first_control(xt); 
        }


        //PRINT("t=" << t << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

        const StateVector xt1 = dynamics_(xt, ut);

        //update state, control, cost
        states_.push_back(xt1);
        controls_.push_back(ut);
        rollout_cost_ += goals_[true_goal_ind_].ct_(xt, ut, current_timestep_);

        //get the user action, compute by solving for true chain
        //user_solver.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
        //user_ut = user_solver.compute_first_control(xt); 


    }

    if (is_done())
    {
        rollout_cost_ += goals_[true_goal_ind_].cT_(get_last_state());
    }



}


StateVector SharedAutonomyCircle::get_state_at_ind(int ind)
{
    IS_GREATER(states_.size(), ind);
    return states_[ind];
}

ControlVector SharedAutonomyCircle::get_control_at_ind(int ind)
{
    IS_GREATER(controls_.size(), ind);
    return controls_[ind];
}


std::vector<double> SharedAutonomyCircle::get_values_at_positions(const std::vector<Eigen::Vector2d>& positions, int num_timesteps_future)
{
    if (policy_type_ == PolicyTypes::TRUE_ILQR || policy_type_ == PolicyTypes::AVG_COST)
    {
        return get_values_at_positions_onebranch(positions, 0, num_timesteps_future);
    } else if (policy_type_ == PolicyTypes::ARGMAX_ILQR) {
        std::vector<double> goal_distribution = goal_predictor_.get_goal_distribution();
        int argmax_branch = std::distance(goal_distribution.begin(), std::max_element(goal_distribution.begin(), goal_distribution.end()));
        return get_values_at_positions_onebranch(positions, argmax_branch, num_timesteps_future);
    } else {
        //take expected value for hindsight and prob_weighted
        std::vector<double> goal_distribution = goal_predictor_.get_goal_distribution();
        std::vector<double> exp_values(positions.size(), 0.0);

        for (int branch_num=0; branch_num < NUM_GOALS; branch_num++)
        {
            std::vector<double> vals_this_branch = get_values_at_positions_onebranch(positions, branch_num, num_timesteps_future);

            //multiply by prob of goal
            std::transform(vals_this_branch.begin(), vals_this_branch.end(), vals_this_branch.begin(),
               std::bind1st(std::multiplies<double>(),goal_distribution[branch_num]));

            //add to output
            std::transform(exp_values.begin(), exp_values.end(), vals_this_branch.begin(), exp_values.begin(), std::plus<double>());
        }

        return exp_values;
    }


}


std::vector<double> SharedAutonomyCircle::get_values_at_positions_onebranch(const std::vector<Eigen::Vector2d>& positions, int branch_num, int num_timesteps_future)
{
    std::vector<double> values(positions.size());
    for (size_t i=0; i < positions.size(); i++)
    {
        StateVector s;
        s[State::POS_X] = positions[i][0];
        s[State::POS_Y] = positions[i][1];
        if (policy_type_ == PolicyTypes::PROB_WEIGHTED_CONTROL)
        {
            values[i] = solvers_[branch_num].compute_value( 0, s, num_timesteps_future);   
        } else {
            values[i] = solvers_[0].compute_value( branch_num, s, num_timesteps_future);   
        }
    }

    return values;
}

//return a cost function that is the weighted average of the specified function
user_goal::CostFunction avg_cost(const std::vector<user_goal::CostFunction>& cost_functions, const filters::GoalPredictor& goal_predictor)
{
    return [&goal_predictor, &cost_functions](const StateVector& s, const ControlVector& u, const int t)
    {
        std::vector<double> weights = goal_predictor.get_goal_distribution();
        double ret_val = 0;
        for (size_t i=0; i < cost_functions.size(); i++)
        {
            ret_val += weights[i]*cost_functions[i](s, u, t);
        }
        return ret_val;
    };
}


user_goal::CostFunction avg_cost(const std::vector<user_goal::User_Goal>& user_goals)
{
    return [&user_goals](const StateVector& s, const ControlVector& u, const int t)
    {
        double ret_val = 0;
        for (auto& goal: user_goals)
        {
            ret_val += goal.prob_*goal.ct_(s, u, t);
        }
        return ret_val;
    };


}


user_goal::FinalCostFunction avg_final_cost(const std::vector<user_goal::User_Goal>& user_goals)
{
    return [&user_goals](const StateVector& s)
    {
        double ret_val = 0;
        for (auto& goal: user_goals)
        {
            ret_val += goal.prob_*goal.cT_(s);
        }
        return ret_val;
    };

}


} //namespace experiments
