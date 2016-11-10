//
// Implements Tree-iLQR.
//

#pragma once

#include <ilqr/tree.hh>
#include <lqr/lqr_types.hh>

#include <Eigen/Dense>

#include <memory>
#include <list>
#include <vector>

namespace lqr
{

// Shared pointer to a Node in the underlying Tree structure of the iLQR-Tree. Calling ->item()
// returns a shared pointer to the PlanNode that holds the state/control/dynamics/etc.
// information.
using TreeNodePtr = std::shared_ptr<data::Node<PlanNode>>;

class LQRTree 
{
public:
    LQRTree(int state_dim, int control_dim);
    virtual ~LQRTree() = default;

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

    // Forward pass to generate new nominal points for iLQR.
    // The step-size is controlled by alpha (alpha * K).
    void forward_pass(const double alpha);

    // Do a full bellman backup on the tree.
    void bellman_tree_backup();

private:
    int state_dim_ = 0;
    int control_dim_ = 0;

    data::Tree<PlanNode> tree_;

    // Zeros matrix of size [state_dim +1] x [state_dim +1]. Used as
    // initialization for the zeros matrix.
    const Eigen::MatrixXd ZERO_VALUE_MATRIX_; 

    // The alpha is the step size to use in the control application.
    Eigen::MatrixXd forward_node(std::shared_ptr<PlanNode> node, 
                                 const Eigen::MatrixXd &xt, 
                                 const double alpha);

    // Special case for just the leaves of the tree. We can compute this by giving the leaves
    // synthetic children with $V_{T+1} = 0$.
    void control_and_value_for_leaves();

    // Backups the value matrix and control gains matrix from the children of the node to get 
    // the value and control policies for the parents of the children. Returns a list of all the
    // parents.
    std::list<TreeNodePtr> backup_to_parents(const std::list<TreeNodePtr> &all_children);

    // Helper to compute the value matrix for plan_node given a child's V_{t+1} value matrix.
    // Returns the result and does not store the result in the PlanNode.
    Eigen::MatrixXd compute_value_matrix(const std::shared_ptr<PlanNode> &node, 
            const Eigen::MatrixXd &Vt1);

    // Helper to compute the feedback and feedforward control policies. Stores them in the PlanNode.
    void compute_control_policy(std::shared_ptr<PlanNode> &node, const Eigen::MatrixXd &Vt1);
};

} // namespace lqr
