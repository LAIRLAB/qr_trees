//
// Implements Tree-iLQR.
//

#pragma once

#include <ilqr/tree.hh>
#include <ilqr/ilqr_node.hh>

#include <Eigen/Dense>

#include <memory>
#include <list>
#include <vector>

namespace ilqr
{

// Shared pointer to a Node in the underlying Tree structure of the iLQR-Tree. Calling ->item()
// returns a shared pointer to the iLQRNode that holds the state/control/dynamics/etc.
// information.
using TreeNodePtr = std::shared_ptr<data::Node<iLQRNode>>;

class iLQRTree 
{
public:
    iLQRTree(int state_dim, int control_dim);
    virtual ~iLQRTree() = default;

    // Construct a iLQRNode which represents the dynamics, cost functions as
    // well as the state and control policy at a specific time step along the
    // tree. Nominal state and control inputs are passed and used for the
    // initial Taylor expansions of the cost and dynamics functions.
    std::shared_ptr<iLQRNode> make_ilqr_node(const Eigen::VectorXd &x_star,
            const Eigen::VectorXd &u_star, const DynamicsFunc &dynamics, const
            CostFunc &cost, const double probablity);

    // Add the root node to the iLQR Tree with similar arguments to the
    // make_ilqr_node function.
    TreeNodePtr add_root(const Eigen::VectorXd &x_star, const Eigen::VectorXd
            &u_star, const DynamicsFunc &dynamics, const CostFunc &cost);
    
    // Add a iLQRNode as the root node to the iLQR Tree. Requires the
    // probability to be 1. 
    TreeNodePtr add_root(const std::shared_ptr<iLQRNode> &ilqr_node);

    // Add a list of iLQRNodes as children under a parent. The probabilities
    // must sum to 1.
    std::vector<TreeNodePtr> add_nodes(const
            std::vector<std::shared_ptr<iLQRNode>> &ilqr_nodes, TreeNodePtr
            &parent);

    // Get the root node of the Tree.
    TreeNodePtr root();

    // Forward pass to generate new nominal points for iLQR.
    // The step-size is controlled by alpha (alpha * K).
    void forward_tree_update(const double alpha);

    // Do a full bellman backup on the tree.
    void bellman_tree_backup();

    // TODO: Should make a function that returns a Tree of states with expected cost-to-gos.
    //void forward_pass(const Eigen::VectorXd &x0, 
    //        std::vector<Eigen::VectorXd> &states, 
    //        std::vector<Eigen::VectorXd> &controls, 
    //        std::vector<double> &costs, 
    //        const TreeNodePtr &start_node = nullptr);

private:
    int state_dim_ = 0;
    int control_dim_ = 0;

    data::Tree<iLQRNode> tree_;

    // Zeros matrix of size [state_dim +1] x [state_dim +1]. Used as
    // initialization for the zeros matrix.
    const QuadraticValue ZERO_VALUE_; 

    // The alpha is the step size to use in the control application.
    void forward_node(const std::shared_ptr<iLQRNode>& parent_t,
                      std::shared_ptr<iLQRNode>& child_tp1, 
                      const Eigen::MatrixXd &xt,
                      const double alpha,
                      const bool update_expansion,
                      Eigen::VectorXd &ut,
                      Eigen::VectorXd &xt1
                      );

    // Special case for just the leaves of the tree. We can compute this by
    // giving the leaves synthetic children with $V_{T+1} = 0$.
    void control_and_value_for_leaves();

    // Backups the value matrix and control gains matrix from the children of
    // the node to get the value and control policies for the parents of the
    // children. Returns a list of all the parents.
    std::list<TreeNodePtr> backup_to_parents(const std::list<TreeNodePtr>
            &all_children);

    // Computes the added weighted quadratic value 
    // b += probability*a. Each term in 'a' is scaled linearily by probability. 
    static void add_weighted_value(const double probability, 
            const QuadraticValue &a, QuadraticValue &b);
};

} // namespace ilqr
