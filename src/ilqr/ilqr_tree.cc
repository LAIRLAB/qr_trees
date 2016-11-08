#include <ilqr/ilqr_tree.hh>

#include <utils/debug_utils.hh>

#include <iterator>
#include <numeric>
#include <unordered_map>

namespace
{
    // Double precision equality checking epsilon.
    constexpr double EPS = 1e-5;

} // namespace

namespace ilqr
{

iLQRTree::iLQRTree(int state_dim, int control_dim)
    : state_dim_(state_dim),
      control_dim_(control_dim),
      ZeroValueMatrix_(Eigen::MatrixXd::Zero(state_dim + 1, state_dim + 1))
{
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

TreeNodePtr iLQRTree::add_root(const Eigen::VectorXd &x_star, const Eigen::VectorXd &u_star, 
        const DynamicsFunc &dynamics_func, const CostFunc &cost_func)
{
    return add_root(make_plan_node(x_star, u_star, dynamics_func, cost_func, 1.0)); 
}

TreeNodePtr iLQRTree::add_root(const std::shared_ptr<PlanNode> &plan_node)
{
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

TreeNodePtr iLQRTree::root()
{
    return tree_.root();
}

void iLQRTree::forward_pass(const double alpha)
{
    // Process from the end of the list, but start at the beginning.
    std::list<std::pair<TreeNodePtr, Eigen::MatrixXd>> to_process;
    // First x linearization is just from the root, not from rolling out dynamics.
    Eigen::VectorXd xt = tree_.root()->item()->x();
    TreeNodePtr tree_node = tree_.root();
    to_process.emplace_front(tree_.root(), xt);
    while (!to_process.empty())
    {
        auto &process_pair = to_process.back();
        //TODO: Implement line search over this function for the forward pass.
        const Eigen::MatrixXd xt1 = forward_node(process_pair.first->item(), process_pair.second, alpha); 
        for (auto child : process_pair.first->children())
        {
            to_process.emplace_front(child, xt1);
        }

        to_process.pop_back();
    }
}

Eigen::MatrixXd iLQRTree::forward_node(std::shared_ptr<PlanNode> node, 
        const Eigen::MatrixXd &xt, 
        const double alpha)
{
    // Compute difference from where the node is at now during the forward pass versus the 
    // linearization point from before.
    const Eigen::VectorXd dx_t = (xt - node->x());
    const Eigen::VectorXd du_t = node->K_ * dx_t + node->k_;

    // Set the new linearization point at the new xt for the node.
    node->set_x(xt);
    node->set_u(alpha*du_t + node->u());
    node->update_dynamics(); 
    node->update_cost();

    // The upper right corner of A has x_{t+1} = f(x_t,u_t), i.e. the evaluation of the dynamics at x_t,u_t
    Eigen::VectorXd xt1 = Eigen::VectorXd::Zero(state_dim_+1);
    xt1.topRows(state_dim_) = node->dynamics_.A.topRightCorner(state_dim_, 1);
    return xt1;
}



void iLQRTree::bellman_tree_backup()
{
   // Special case to compute the control policy and value matrices for the leaf nodes.
   control_and_value_for_leaves();

   // Start at all the leaf nodes (currently assume they are all at the same depth) and work our
   // way up the tree until we get the root node (single node at depth = 0).
   auto all_children = tree_.leaf_nodes();
   while (!(all_children.size() == 1 && all_children.front()->depth() == 0))
   {
       // For all the children, back up their value function to their parents. 
       // To work up the tree, the parents become the new children.
       all_children = backup_to_parents(all_children);
   } 
}

void iLQRTree::control_and_value_for_leaves()
{
   auto leaf_nodes = tree_.leaf_nodes();

   // Confirm all leaves are at the same depth in the tree. This isn't necessary for the
   // general algorithm, but we need it for the current implementation.
   const int FIRST_DEPTH = leaf_nodes.size() > 0 ? leaf_nodes.front()->depth() : -1;
   for (auto &leaf: leaf_nodes)
   {
       IS_EQUAL(leaf->depth(), FIRST_DEPTH);
       std::shared_ptr<PlanNode> node = leaf->item();
       // Compute the leaf node's control policy K, k using a Zero Value matrix for the future.
       compute_control_policy(node, ZeroValueMatrix_);
       node->V_ = compute_value_matrix(node, ZeroValueMatrix_);
   }
}

std::list<TreeNodePtr> iLQRTree::backup_to_parents(const std::list<TreeNodePtr> &all_children)
{
   // Hash the leaves by their parent so we can process all the children for a parent.    
   std::unordered_map<TreeNodePtr, std::list<TreeNodePtr>> parent_map;

   // Confirm all children are at the same depth in the tree. This isn't necessary for the
   // general algorithm, but we need it for the current implementation.
   const int FIRST_DEPTH = all_children.size() > 0 ? all_children.front()->depth() : -1;
   for (auto &child : all_children)
   {
       IS_EQUAL(child->depth(), FIRST_DEPTH);
       parent_map[child->parent()].push_back(child);
   }

   std::list<TreeNodePtr> parents;
   for (auto &parent_children_pair : parent_map)
   {
       // Compute the weighted \tilde{V}_{t+1} = \sum_k p_k V_{T+1}^{(k)} matrix by computing the
       // probability-weighted average 
       Eigen::MatrixXd Vtilde;
       auto &children = parent_children_pair.second;
       for (auto &child : children)
       {
           const std::shared_ptr<PlanNode> &child_node = child->item();
           Vtilde += child_node->probability_ * child_node->V_;
       }

       std::shared_ptr<PlanNode> parent_node = parent_children_pair.first->item();
       // Compute the parent node's control policy K, k using Vtilde.
       compute_control_policy(parent_node, Vtilde);
       // Compute parent's Vt from this the Vtilde (from T+1) and the control policy K,k computed above.
       parent_node->V_ = compute_value_matrix(parent_node, Vtilde);

       parents.push_back(parent_children_pair.first);
   }

   return parents;
}

Eigen::MatrixXd iLQRTree::compute_value_matrix(const std::shared_ptr<PlanNode> &node, 
                                               const Eigen::MatrixXd &Vt1)
{
    // Extract dynamics terms.
    const Eigen::MatrixXd &A = node->dynamics_.A;
    const Eigen::MatrixXd &B = node->dynamics_.B;
    // Extract cost terms.
    const Eigen::MatrixXd &Q = node->cost_.Q;
    const Eigen::MatrixXd &P = node->cost_.P;
    const Eigen::MatrixXd &b_u = node->cost_.b_u;
    // Extract control policy terms.
    const Eigen::MatrixXd &K = node->K_;
    const Eigen::MatrixXd &k = node->k_;

    const Eigen::MatrixXd cntrl_cross_term = P + A.transpose() * Vt1 * B;
    Eigen::MatrixXd quadratic_term = Q + A.transpose() * Vt1 * A + cntrl_cross_term*K;
    IS_EQUAL(quadratic_term.rows(), state_dim_ + 1);
    IS_EQUAL(quadratic_term.cols(), state_dim_ + 1);

    Eigen::MatrixXd linear_term = cntrl_cross_term*k;
    IS_EQUAL(quadratic_term.rows(), state_dim_ + 1);
    IS_EQUAL(quadratic_term.cols(), 1);

    Eigen::MatrixXd constant_term = b_u.transpose() * k;

    Eigen::MatrixXd Vt = quadratic_term; 
    Vt.topRightCorner(state_dim_, 1) += linear_term.topRows(state_dim_);
    Vt.bottomLeftCorner(1, state_dim_) += linear_term.topRows(state_dim_).transpose();
    Vt.bottomRightCorner(1, 1) += constant_term;

    return Vt;
}

void iLQRTree::compute_control_policy(std::shared_ptr<PlanNode> &node, const Eigen::MatrixXd &Vt1)
{
    const Eigen::MatrixXd &A = node->dynamics_.A;
    const Eigen::MatrixXd &B = node->dynamics_.B;
    // Extract cost terms.
    const Eigen::MatrixXd &P = node->cost_.P;
    const Eigen::MatrixXd &R = node->cost_.R;
    const Eigen::MatrixXd &b_u = node->cost_.b_u;

    const Eigen::MatrixXd inv_cntrl_term = (R + B.transpose()*Vt1*B).inverse();

    node->K_ = -1.0 * inv_cntrl_term * (P.transpose() + B.transpose() * Vt1 * A);
    node->k_ = -1.0 * inv_cntrl_term * b_u; 
}

} // namespace ilqr
