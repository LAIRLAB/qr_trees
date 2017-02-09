//
// Shervin Javdani (sjavdani@cs.cmu.edu)
// January 2017
//

#pragma once

#include <experiments/simulators/circle_world.hh>
#include <experiments/simulators/directdrive.hh>
#include <experiments/simulators/user_goal.hh>
#include <templated/iLQR_hindsight_value.hh>
#include <filters/goal_predictor.hh>

#include <memory>

#include <array>
#include <string>


namespace experiments
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


enum class PolicyTypes
{
    // Compute an iLQR policy from a tree that splits only at the first timestep.
    HINDSIGHT = 0,
    // Compute the iLQR chain policy under the true dynamics. This should be the best solution.
    TRUE_ILQR,
    // Compute iLQR chain using the argmax(probabilities_from_filter) dynamics.
    // This should perfectly when there is no noise in the observations.
    ARGMAX_ILQR,
    // Compute iLQR chain for each probabilistic split and take a weighted average of the controls.
    PROB_WEIGHTED_CONTROL,
    // Compute iLQR as single chain with the cost function as the weighted average of all cost functions
    AVG_COST,
};

std::string to_string(const PolicyTypes policy_type)
{
    switch(policy_type)
    {
        case PolicyTypes::HINDSIGHT:
            return "hindsight";
        case PolicyTypes::TRUE_ILQR:
            return "ilqr_true";
        case PolicyTypes::ARGMAX_ILQR:
            return "argmax";
        case PolicyTypes::PROB_WEIGHTED_CONTROL:
            return "weighted";
        case PolicyTypes::AVG_COST:
            return "average_cost";
    };
    return "Unrecognized policy type. Error.";
}

class SharedAutonomyCircle
{
public:
    SharedAutonomyCircle(const PolicyTypes policy,
        const CircleWorld &world,
        const std::vector<StateVector>& goal_states,
        const std::vector<double>& goal_priors,
        const int true_goal_ind,
        const int timesteps);


    void run_control(int num_timesteps);

    StateVector get_state_at_ind(int ind);
    ControlVector get_control_at_ind(int ind);

    std::vector<double> get_values_at_positions(const std::vector<Eigen::Vector2d>& positions, int num_timesteps_future=0);
    std::vector<double> get_values_at_positions_onebranch(const std::vector<Eigen::Vector2d>& positions, int branch_num, int num_timesteps_future=0);

    inline const PolicyTypes get_policy_type(){return policy_type_;}

    inline StateVector get_last_state(){return states_.back();}
    inline ControlVector get_last_control(){return controls_.back();}
    inline int get_num_timesteps_remaining(){return timesteps_ - current_timestep_;}
    inline bool is_done(){return current_timestep_ >= timesteps_;}
    inline double get_rollout_cost(){return rollout_cost_;}
    inline int get_num_states_computed(){return states_.size();}

    std::vector<StateVector> const get_states(){return states_;}
    std::vector<ControlVector> const get_controls(){return controls_;}

private:
    PolicyTypes policy_type_;
    CircleWorld world_;
    std::vector<user_goal::User_Goal> goals_;
    int true_goal_ind_;
    int timesteps_;
    simulators::directdrive::DirectDrive dynamics_;
    filters::GoalPredictor goal_predictor_;  
    const int NUM_GOALS;

    std::vector<StateVector> states_;
    std::vector<ControlVector> controls_;
    ControlVector u_nominal_;

    int current_timestep_ = 0;
    double rollout_cost_ = 0.;

    //solvers for computing control
    std::vector<ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM>> solvers_;
    std::vector<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM>> solver_branches_;
    //solvers for computing prediction values. Always one per goal
    std::vector<ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM>> prediction_solvers_;
    std::vector<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM>> prediction_solver_branches_;
    //solver for user. Always one for true goal
    std::unique_ptr<ilqr::iLQRHindsightValueSolver<STATE_DIM,CONTROL_DIM>> user_solver_;
    std::unique_ptr<ilqr::HindsightBranchValue<STATE_DIM,CONTROL_DIM>> user_solver_branch_;
};


user_goal::CostFunction avg_cost(const std::vector<user_goal::User_Goal>& user_goals);
user_goal::FinalCostFunction avg_final_cost(const std::vector<user_goal::User_Goal>& user_goals);



} //namespace experiments
