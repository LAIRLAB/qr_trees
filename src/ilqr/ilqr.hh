//
// Implements Tree-iLQR.
//

#pragma once

#include <ilqr/tree.hh>
#include <ilqr/types.hh>

#include <Eigen/Dense>

#include <memory>
#include <list>
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

    // Construct a PlanNode which represents the dynamics, cost functions as well as the state
    // and control policy at a specific time step along the tree. Nominal state and control inputs
    // are passed and used for the initial Taylor expansions of the cost and dynamics functions.
    std::shared_ptr<PlanNode> make_plan_node(const Eigen::VectorXd &x_star,
                                             const Eigen::VectorXd &u_star,
                                             const DynamicsFunc &dynamics,
                                             const CostFunc &cost,
                                             const double probablity);

    // Add the root node to the iLQR Tree with similar arguments to the make_plan_node function.
    TreeNodePtr add_root(const Eigen::VectorXd &x_star, const Eigen::VectorXd &u_star, 
            const DynamicsFunc &dynamics, const CostFunc &cost);
    
    // Add a PlanNode as the root node to the iLQR Tree. Requires the probability to be 1. 
    TreeNodePtr add_root(const std::shared_ptr<PlanNode> &plan_node);

    // Add a list of PlanNodes as children under a parent. The probabilities must sum to 1.
    std::vector<TreeNodePtr> add_nodes(const std::vector<std::shared_ptr<PlanNode>> &plan_nodes, 
            TreeNodePtr &parent);

    // Get the root node of the Tree.
    TreeNodePtr root();

    // Do a full bellman backup on the tree
    void bellman_tree_backup();
private:
    int state_dim_ = 0;
    int control_dim_ = 0;

    data::Tree<PlanNode> tree_;

    const Eigen::MatrixXd ZeroValueMatrix_; 

    // Backups the value matrix and control gains matrix from the children of the node.
    std::list<TreeNodePtr> backup_to_parents(const std::list<TreeNodePtr> &all_children);

    // Special case for just the leaves of the tree. We can compute this by giving the leaves
    // synthetic children with $V_{T+1} = 0$.
    void control_and_value_for_leaves();

    // Helper to compute the value matrix for plan_node given a child's V_{t+1} value matrix.
    // Returns the result and does not store the result in the PlanNode.
    Eigen::MatrixXd compute_value_matrix(const std::shared_ptr<PlanNode> &node, 
            const Eigen::MatrixXd &Vt1);

    // Helper to compute the feedback and feedforward control policies. Stores them in the PlanNode.
    void compute_control_policy(std::shared_ptr<PlanNode> &node, const Eigen::MatrixXd &Vt1);
};
