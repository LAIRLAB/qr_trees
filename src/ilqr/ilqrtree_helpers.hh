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
} // namespace ilqr

