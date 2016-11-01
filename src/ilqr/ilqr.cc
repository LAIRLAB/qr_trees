#include <ilqr/ilqr.hh>

#include <utils/debug_utils.hh>

#include <algorithm>
#include <iterator>

namespace
{
    // Double precision equality checking epsilon.
    constexpr double EPS = 1e-5;

} // namespace

iLQRTree::iLQRTree(int state_dim, int control_dim)
    : state_dim_(state_dim),
      control_dim_(control_dim)
{

}

TreeNodePtr iLQRTree::add_root(const Eigen::VectorXd &x_star, const Eigen::VectorXd &u_star, 
        const DynamicsFunc &dynamics_func, const CostFunc &cost_func)
{
    std::shared_ptr<PlanNode> plan_node = make_plan_node(x_star, u_star, dynamics_func, cost_func, 1.0); 
    tree_ = data::Tree<PlanNode>(plan_node);
    return tree_.root();
}

std::vector<TreeNodePtr> iLQRTree::add_nodes(const std::vector<std::shared_ptr<PlanNode>> &plan_nodes, 
        TreeNodePtr &parent)
{
    // Confirm the probabilities in the plan nodes sum to 1.
    const double probability_sum = 
        std::accumulate(plan_nodes.begin(), plan_nodes.end(), 0.0,
            [](const double a, const std::shared_ptr<PlanNode> &node) 
            {
                return a + node->probability_;
            }
            );
    IS_ALMOST_EQUAL(probability_sum, 1.0, EPS); // Throw error if sum is not close to 1.0

    // Create tree nodes from the plan nodes and add them to the tree.
    std::vector<TreeNodePtr> children;
    children.reserve(plan_nodes.size());
    for (const auto &plan_node : plan_nodes)
    {
        children.emplace_back(tree_.add_child(parent, plan_node));
    }
    return children;
}

std::shared_ptr<PlanNode> iLQRTree::make_plan_node(const Eigen::VectorXd &x_star, 
                                const Eigen::VectorXd &u_star, 
                                const DynamicsFunc &dynamics_func, 
                                const CostFunc &cost_func,
                                const double probability)
{
    IS_EQUAL(x_star.size(), state_dim_);
    IS_EQUAL(u_star.size(), control_dim_);

    std::shared_ptr<PlanNode> plan_node 
        = std::make_shared<PlanNode>(state_dim_, control_dim_, dynamics_func, cost_func, probability);

    // Add node creation, the nominal state/control and the forward pass state/control used for 
    // differentiating the Dynamics and cost function are the same.
    plan_node->set_xstar(x_star);
    plan_node->set_ustar(u_star);
    plan_node->set_x(x_star);
    plan_node->set_u(u_star);

    // Update the linearization and quadraticization of the dynamics and cost respectively.
    plan_node->update_dynamics();
    plan_node->update_cost();

    return plan_node;
}
