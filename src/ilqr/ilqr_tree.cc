#include <ilqr/ilqr_tree.hh>

#include <utils/debug_utils.hh>

#include <iterator>
#include <numeric>
#include <tuple>
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

std::shared_ptr<iLQRNode> iLQRTree::make_ilqr_node(const Eigen::VectorXd &x_star, 
                                const Eigen::VectorXd &u_star, 
                                const DynamicsFunc &dynamics_func, 
                                const CostFunc &cost_func,
                                const double probability)
{
    IS_EQUAL(x_star.size(), state_dim_);
    IS_EQUAL(u_star.size(), control_dim_);

    std::shared_ptr<iLQRNode> ilqr_node 
        = std::make_shared<iLQRNode>(x_star, u_star, 
                                     dynamics_func, cost_func, 
                                     probability);

    return ilqr_node;
}

TreeNodePtr iLQRTree::add_root(const Eigen::VectorXd &x_star, 
                               const Eigen::VectorXd &u_star, 
                               const DynamicsFunc &dynamics_func, 
                               const CostFunc &cost_func)
{
    return add_root(make_ilqr_node(x_star, u_star, dynamics_func, cost_func, 1.0)); 
}

TreeNodePtr iLQRTree::add_root(const std::shared_ptr<iLQRNode> &ilqr_node)
{
    tree_ = data::Tree<iLQRNode>(ilqr_node);
    return tree_.root();
}

std::vector<TreeNodePtr> iLQRTree::add_nodes(
        const std::vector<std::shared_ptr<iLQRNode>> &ilqr_nodes, 
        TreeNodePtr &parent)
{
    // Confirm the probabilities in the plan nodes sum to 1.
    const double probability_sum = 
        std::accumulate(ilqr_nodes.begin(), ilqr_nodes.end(), 0.0,
            [](const double a, const std::shared_ptr<iLQRNode> &node) 
            {
                return a + node->probability();
            }
            );
    // Throw error if sum is not close to 1.0
    IS_ALMOST_EQUAL(probability_sum, 1.0, EPS); 

    // Create tree nodes from the plan nodes and add them to the tree.
    std::vector<TreeNodePtr> children;
    children.reserve(ilqr_nodes.size());
    for (const auto &ilqr_node : ilqr_nodes)
    {
        children.emplace_back(tree_.add_child(parent, ilqr_node));
    }
    return children;
}

TreeNodePtr iLQRTree::root()
{
    return tree_.root();
}

void iLQRTree::forward_tree_update(const double alpha)
{
    // Process from the end of the list, but start at the beginning. Each tuple is (parent_node_t, child_node_{t+1}, x_t).
    std::list<std::tuple<TreeNodePtr, TreeNodePtr, Eigen::VectorXd>> to_process;
    const Eigen::VectorXd x0 = tree_.root()->item()->x();
    auto parent = tree_.root();
    for (auto child : tree_.root()->children())
    {
        // First x linearization is just from the root, not from rolling out dynamics.
        to_process.emplace_front(parent, child, x0);
    }
    while (!to_process.empty())
    {
        auto &process_tuple = to_process.back();
        //TODO: Implement line search over this function for the forward pass.
        const std::shared_ptr<iLQRNode> parent_ilqr_node = std::get<0>(process_tuple)->item();
        std::shared_ptr<iLQRNode> child_ilqr_node = std::get<1>(process_tuple)->item();
        const Eigen::VectorXd &xt = std::get<2>(process_tuple);

        Eigen::VectorXd xt1, ut;
        forward_node(parent_ilqr_node, child_ilqr_node, xt, alpha, true, ut, xt1); 


        TreeNodePtr new_parent_node = std::get<1>(process_tuple);
        if (new_parent_node->num_children() == 0)
        {
            // There are no dynamics really from this node on so we give zero control.
            Eigen::VectorXd xt1_null, ut_null;
            forward_node(child_ilqr_node, child_ilqr_node, xt1, alpha, true, ut_null, xt1_null); 
        }
        for (auto &child : new_parent_node->children())
        {
            to_process.emplace_front(new_parent_node, child, xt1);
        }

        to_process.pop_back();
    }
}

void iLQRTree::forward_node(const std::shared_ptr<iLQRNode>& parent_t,
                            std::shared_ptr<iLQRNode>& child_tp1, 
                            const Eigen::MatrixXd &xt, 
                            const double alpha, 
                            const bool update_expansion,
                            Eigen::VectorXd &ut,
                            Eigen::VectorXd &xt1)
{
    // Roll forward the dynamics stored in the child from xt. 
    // The control policy is from the parent.
    
    // Compute difference from where the node is at now during the forward pass
    // versus the linearization point from before.
    const Eigen::VectorXd zt = (xt - parent_t->x()); 
    const Eigen::VectorXd vt = (parent_t->K()*zt) + parent_t->k();

    // Set the new linearization point at the new xt for the node.
    ut = alpha*vt + parent_t->u();

    if (update_expansion) 
    {
        parent_t->update_expansion(xt, ut); 
    }

    xt1 = child_tp1->dynamics_func()(xt, ut);
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
    auto identity_dynamics = [](const Eigen::VectorXd &x, const Eigen::VectorXd &u) { return static_cast<Eigen::VectorXd>(x); };
    auto zero_cost = [](const Eigen::VectorXd &x, const Eigen::VectorXd &u) { return 0.0; };
   std::shared_ptr<iLQRNode> first_ilqr_node = leaf_nodes.front()->item();
   const std::shared_ptr<ilqr::iLQRNode> zero_value_node 
       = std::make_shared<ilqr::iLQRNode>(first_ilqr_node->x(),  first_ilqr_node->u(), identity_dynamics, zero_cost, 1.0);
   for (auto &leaf: leaf_nodes)
   {
       IS_EQUAL(leaf->depth(), first_depth);
       std::shared_ptr<iLQRNode> node = leaf->item();
       // Compute the leaf node's control policy K, k using a Zero Value matrix for 
       // the future.
       node->bellman_backup({zero_value_node});
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
       std::list<TreeNodePtr> &children = parent_children_pair.second;
       std::vector<std::shared_ptr<ilqr::iLQRNode>> child_ilqr_nodes(children.size());
       std::transform(children.begin(), children.end(), child_ilqr_nodes.begin(), 
               [](const TreeNodePtr &node)
               {
                   return node->item();
               });

       std::shared_ptr<iLQRNode> parent_node = parent_children_pair.first->item();
       // Compute the parent node's control policy and value function from the 
       // linearly weighted value functions of the child nodes.
       parent_node->bellman_backup(child_ilqr_nodes);
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
