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
      ZERO_VALUE_(state_dim)
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
        = std::make_shared<PlanNode>(x_star, u_star, 
                                     dynamics_func, cost_func, 
                                     probability);

    return plan_node;
}

TreeNodePtr iLQRTree::add_root(const Eigen::VectorXd &x_star, 
                               const Eigen::VectorXd &u_star, 
                               const DynamicsFunc &dynamics_func, 
                               const CostFunc &cost_func)
{
    return add_root(make_plan_node(x_star, u_star, dynamics_func, cost_func, 1.0)); 
}

TreeNodePtr iLQRTree::add_root(const std::shared_ptr<PlanNode> &plan_node)
{
    tree_ = data::Tree<PlanNode>(plan_node);
    return tree_.root();
}

std::vector<TreeNodePtr> iLQRTree::add_nodes(
        const std::vector<std::shared_ptr<PlanNode>> &plan_nodes, 
        TreeNodePtr &parent)
{
    // Confirm the probabilities in the plan nodes sum to 1.
    const double probability_sum = 
        std::accumulate(plan_nodes.begin(), plan_nodes.end(), 0.0,
            [](const double a, const std::shared_ptr<PlanNode> &node) 
            {
                return a + node->probability();
            }
            );
    // Throw error if sum is not close to 1.0
    IS_ALMOST_EQUAL(probability_sum, 1.0, EPS); 

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
    std::list<std::pair<TreeNodePtr, Eigen::VectorXd>> to_process;
    // First x linearization is just from the root, not from rolling out dynamics.
    to_process.emplace_front(tree_.root(), tree_.root()->item()->x());
    while (!to_process.empty())
    {
        auto &process_pair = to_process.back();
        //TODO: Implement line search over this function for the forward pass.
        Eigen::VectorXd xt1, ut;
        const Eigen::VectorXd &xt = process_pair.second;
        auto plan_node = process_pair.first->item();

        forward_node(plan_node, xt, alpha, true, ut, xt1); 
        for (auto child : process_pair.first->children())
        {
            to_process.emplace_front(child, xt1);
        }

        to_process.pop_back();
    }
}

void iLQRTree::forward_node(std::shared_ptr<PlanNode>& node, 
                            const Eigen::MatrixXd &xt, 
                            const double alpha, 
                            const bool update_expansion,
                            Eigen::VectorXd &ut,
                            Eigen::VectorXd &xt1)
{
    // Compute difference from where the node is at now during the forward pass
    // versus the linearization point from before.
    const Eigen::VectorXd zt = (xt - node->x()); const Eigen::VectorXd vt =
        (node->K()*zt) + node->k();

    // Set the new linearization point at the new xt for the node.
    ut = alpha*vt + node->u();

    xt1 = node->dynamics_func()(xt, ut);

    if (update_expansion) 
    {
        node->update_expansion(xt, ut); 
    }
}



void iLQRTree::bellman_tree_backup()
{
   // Special case to compute the control policy and value matrices for the 
   // leaf nodes.
   control_and_value_for_leaves();

   // Start at all the leaf nodes (currently assume they are all at the same
   // depth) and work our way up the tree until we get the root node (single
   // node at depth = 0).
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

   // Confirm all leaves are at the same depth in the tree. This isn't necessary
   // for the general algorithm, but we need it for the current implementation.
   const int first_depth = leaf_nodes.size() > 0 
                               ? leaf_nodes.front()->depth() 
                               : -1;

   for (auto &leaf: leaf_nodes)
   {
       IS_EQUAL(leaf->depth(), first_depth);
       std::shared_ptr<PlanNode> node = leaf->item();
       // Compute the leaf node's control policy K, k using a Zero Value matrix for 
       // the future.
       node->bellman_backup(ZERO_VALUE_);
   }
}

std::list<TreeNodePtr> iLQRTree::backup_to_parents(const std::list<TreeNodePtr> &all_children)
{

   // Hash the leaves by their parent so we can process all the children for a
   // parent.    
   std::unordered_map<TreeNodePtr, std::list<TreeNodePtr>> parent_map;

   // Confirm all children are at the same depth in the tree. This isn't
   // necessary for the general algorithm, but we need it for the current
   // implementation.
   const int first_depth = all_children.size() > 0 
                           ? all_children.front()->depth() 
                           : -1;

   for (auto &child : all_children)
   {
       IS_EQUAL(child->depth(), first_depth);
       parent_map[child->parent()].push_back(child);
   }

   std::list<TreeNodePtr> parents;
   for (auto &parent_children_pair : parent_map)
   {
       // Compute the weighted \tilde{J}_{t+1} = \sum_k p_k J_{T+1}^{(k)} matrix
       // by computing the probability-weighted average 
       QuadraticValue Jtilde = ZERO_VALUE_;
       auto &children = parent_children_pair.second;
       for (auto &child : children)
       {
           const std::shared_ptr<PlanNode> &node = child->item();
           add_weighted_value(node->probability(), node->value(), Jtilde);
       }

       std::shared_ptr<PlanNode> parent_node = parent_children_pair.first->item();
       // Compute the parent node's control policy and value function from the 
       // linearly weighted value functions of the child nodes.
       parent_node->bellman_backup(Jtilde);
       parents.push_back(parent_children_pair.first);
   }

   return parents;
}
void iLQRTree::add_weighted_value(const double probability, 
        const QuadraticValue &a, QuadraticValue &b) 
{
    b.V() += probability * a.V();
    b.G() += probability * a.G();
    b.W() += probability * a.W();
}

} // namespace ilqr
