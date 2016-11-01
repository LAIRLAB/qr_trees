//
// Implements Tree-iLQR.
//

#pragma once

#include <ilqr/tree.hh>
#include <ilqr/types.hh>

#include <Eigen/Dense>

#include <memory>
#include <vector>

// Shared pointer to a Node in the underlying Tree structure of the iLQR-Tree. Calling ->item()
// returns a shared pointer to the PlanNode that holds the state/control/dynamics/etc.
// information.
using TreeNodePtr = std::shared_ptr<data::Node<PlanNode>>;

class ilqr_tree 
{
public:
    ilqr_tree(int state_dim, int control_dim);
    virtual ~ilqr_tree() = default;

    //TreeNodePtr add_root(const Eigen::VectorXd &xstar, const Eigen::VectorXd &ustar, 
    //        const DynamicsFunc &dynamics, const CostFunc &cost);

    TreeNodePtr add_node(const Eigen::VectorXd &xstar, const Eigen::VectorXd &ustar, 
            const DynamicsFunc &dynamics, const CostFunc &cost, const TreeNodePtr &parent);

private:
    data::Tree<PlanNode> tree;

    int state_dim_ = 0;
    int control_dim_ = 0;

};
