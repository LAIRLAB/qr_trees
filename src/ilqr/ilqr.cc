#include <ilqr/ilqr.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

ilqr_tree::ilqr_tree(int state_dim, int control_dim)
    : state_dim_(state_dim),
      control_dim_(control_dim)
{

}

TreeNodePtr ilqr_tree::add_node(const Eigen::VectorXd &x_star, 
                                const Eigen::VectorXd &u_star, 
                                const DynamicsFunc &dynamics, 
                                const CostFunc &cost, 
                                const TreeNodePtr &parent)

{
    IS_EQUAL(xstar.size(), state_dim_);
    IS_EQUAL(ustar.size(), control_dim_);

    std::shared_ptr<PlanNode> plan_node = std::make_shared<PlanNode>(state_dim_, control_dim_);

    // Add node creation, the nominal state/control and the forward pass state/control used for 
    // differentiating the Dynamics and cost function are the same.
    plan_node->set_xstar(x_star);
    plan_node->set_ustar(u_star);
    plan_node->x = x_star;
    plan_node->u = u_star;

    // If we are creating the root node.
    if (!parent) 
    {
    }
    TreeNodePtr tree_node = std::make_shared<data::Node<PlanNode>>(plan_node);
    return tree_node;
}

