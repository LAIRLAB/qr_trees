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

class iLQRTree 
{
public:
    iLQRTree(int state_dim, int control_dim);
    virtual ~iLQRTree() = default;

    std::shared_ptr<PlanNode> make_plan_node(const Eigen::VectorXd &x_star,
                                             const Eigen::VectorXd &u_star,
                                             const DynamicsFunc &dynamics,
                                             const CostFunc &cost,
                                             const double probablity);

    TreeNodePtr add_root(const Eigen::VectorXd &x_star, const Eigen::VectorXd &u_star, 
            const DynamicsFunc &dynamics, const CostFunc &cost);

    std::vector<TreeNodePtr> add_nodes(const std::vector<std::shared_ptr<PlanNode>> &plan_nodes, 
            TreeNodePtr &parent);

private:
    data::Tree<PlanNode> tree_;

    int state_dim_ = 0;
    int control_dim_ = 0;

};
