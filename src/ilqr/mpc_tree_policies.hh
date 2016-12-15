//
// Helper functions to construct MPC-style (continuous replanning) style policies using the iLQRTree 
// class.
//

#pragma once

#include <ilqr/ilqr_taylor_expansions.hh>
#include <ilqr/ilqr_tree.hh>

#include <Eigen/Dense>

#include <vector>

namespace policy
{

// Does hindsight optimization over a probabilistic split in dynamics.
// Returns a control vector "u" by constructing a hindsight_split style tree. 
// Linearly interpolates from xt to xT for the initialization for iLQR with nominal_control for all
// timesteps. 
Eigen::VectorXd hindsight_tree_policy(const int t,
                                      const Eigen::VectorXd& xt, 
                                      const int T,
                                      const Eigen::VectorXd& xT, 
                                      const Eigen::VectorXd& nominal_control,
                                      const std::vector<double> &probabilities, 
                                      const std::vector<ilqr::DynamicsFunc> &dynamics_funcs, 
                                      const ilqr::CostFunc &cost,
                                      ilqr::iLQRTree& ilqr_tree);

// Does hindsight optimization over a probabilistic split in cost functions.
// Returns a control vector "u" by constructing a hindsight_split style tree. 
// Linearly interpolates from xt to xT for the initialization for iLQR with nominal_control for all
// timesteps. 
Eigen::VectorXd hindsight_tree_policy(const int t,
                                      const Eigen::VectorXd& xt, 
                                      const int T,
                                      const Eigen::VectorXd& xT, 
                                      const Eigen::VectorXd& nominal_control,
                                      const std::vector<double> &probabilities, 
                                      const ilqr::DynamicsFunc &dynamics_func,
                                      const std::vector<ilqr::CostFunc> &cost_funcs, 
                                      ilqr::iLQRTree& ilqr_tree);

// Returns a control vector "u" by constructing a chain tree structure.
Eigen::VectorXd chain_policy(const int t, 
        const Eigen::VectorXd& xt, 
        const int T,
        const Eigen::VectorXd& xT, 
        const Eigen::VectorXd& nominal_control,
        const ilqr::DynamicsFunc &dynamics, 
        const ilqr::CostFunc &cost,
        ilqr::iLQRTree& ilqr_tree);


// Probability weighted controller over a probabilistic split in dynamics functions.
Eigen::VectorXd probability_weighted_policy(const int t, 
        const Eigen::VectorXd& xt, 
        const int T,
        const Eigen::VectorXd& xT, 
        const Eigen::VectorXd& nominal_control,
        const std::vector<double> &probabilities, 
        const std::vector<ilqr::DynamicsFunc> &dynamics_funcs, 
        const ilqr::CostFunc &cost);

// Probability weighted controller over a probabilistic split in cost functions.
Eigen::VectorXd probability_weighted_policy(const int t, 
        const Eigen::VectorXd& xt, 
        const int T,
        const Eigen::VectorXd& xT, 
        const Eigen::VectorXd& nominal_control,
        const std::vector<double> &probabilities, 
        const ilqr::DynamicsFunc &dynamics_func, 
        const std::vector<ilqr::CostFunc> &cost_funcs);

} // namespace policy
