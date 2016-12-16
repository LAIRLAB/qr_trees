#pragma once

#include <ilqr/ilqr_taylor_expansions.hh>
#include <ilqr/ilqr_tree.hh>

#include <Eigen/Dense>

#include <vector>

namespace ilqr
{


// Constructs a normal iLQR chain structure using the iLQRTree class.
// If final cost is null, it will not be used on the last timestep.
std::vector<ilqr::TreeNodePtr> construct_chain(
        const std::vector<Eigen::VectorXd>& xstars, 
        const std::vector<Eigen::VectorXd>& ustars, 
        const ilqr::DynamicsFunc &dyn, 
        const ilqr::CostFunc &cost,  
        const ilqr::CostFunc &final_cost,  
        ilqr::iLQRTree& ilqr_tree);

// Constructs a normal iLQR chain structure using the iLQRTree class.
// The cost is used as the final_cost (in the function above) as well.
std::vector<ilqr::TreeNodePtr> construct_chain(
        const std::vector<Eigen::VectorXd>& xstars, 
        const std::vector<Eigen::VectorXd>& ustars, 
        const ilqr::DynamicsFunc &dyn, 
        const ilqr::CostFunc &cost,  
        ilqr::iLQRTree& ilqr_tree);

// Extracts the forward pass states, controls, and computes the costs 
// from a chain structure (vector of tree nodes).
void get_forward_pass_info(const std::vector<ilqr::TreeNodePtr> &chain,
                           const ilqr::CostFunc &cost,  
                           std::vector<Eigen::VectorXd> &states, 
                           std::vector<Eigen::VectorXd> &controls, 
                           std::vector<double> &costs);

// Construct an iLQRTree that splits with b-number of dynamics function branches at the first 
// time step. The probabilities and dynamics function std::vectors are both size b. 
// The time steps in the tree is the size of the std::vectors xstars and ustars.
void construct_hindsight_split_tree(const std::vector<Eigen::VectorXd>& xstars, 
        const std::vector<Eigen::VectorXd>& ustars, 
        const std::vector<double> &probabilities, 
        const std::vector<ilqr::DynamicsFunc> &dynamics_funcs, 
        const ilqr::CostFunc &cost,
        ilqr::iLQRTree& ilqr_tree);

// Construct an iLQRTree that splits with b-number of cost function branches at the first 
// time step. The probabilities and cost function std::vectors are both size b. 
// The time steps in the tree is the size of the std::vectors xstars and ustars.
// final_cost_funcs should either be size b or empty if the same cost_func is to be used on the last
// timestep.
void construct_hindsight_split_tree(const std::vector<Eigen::VectorXd>& xstars, 
        const std::vector<Eigen::VectorXd>& ustars, 
        const std::vector<double> &probabilities, 
        const ilqr::DynamicsFunc &dynamics_func,
        const std::vector<ilqr::CostFunc> &cost_funcs, 
        const std::vector<ilqr::CostFunc> &final_cost_funcs, 
        ilqr::iLQRTree& ilqr_tree);

} // namespace ilqr

