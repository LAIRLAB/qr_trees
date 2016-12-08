#pragma once

#include <ilqr/ilqr_taylor_expansions.hh>
#include <ilqr/ilqr_tree.hh>

#include <Eigen/Dense>

#include <vector>

namespace ilqr
{

// Constructs a normal iLQR chain structure using the iLQRTree class.
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

// Construct an iLQRTree that splits with b-number of branches at the first time
// step. The probabilities and dynamics function std::vectors are both size b. 
// The time steps in the tree is the size of the std::vectors xstars and ustars.
void construct_hindsight_split_tree(const std::vector<Eigen::VectorXd>& xstars, 
        const std::vector<Eigen::VectorXd>& ustars, 
        const std::vector<double> &probabilities, 
        const std::vector<ilqr::DynamicsFunc> &dynamics_funcs, 
        const ilqr::CostFunc &cost,
        ilqr::iLQRTree& ilqr_tree);

} // namespace ilqr

