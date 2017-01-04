//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// December 2016
//

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

    const std::shared_ptr<iLQRNode> parent_ilqr_node = parent->item();
    // Create tree nodes from the plan nodes and add them to the tree.
    std::vector<TreeNodePtr> children;
    children.reserve(ilqr_nodes.size());
    for (const auto &ilqr_node : ilqr_nodes)
    {
        TreeNodePtr child_node = tree_.add_child(parent, ilqr_node);
        children.push_back(child_node);
    }
    return children;
}

TreeNodePtr iLQRTree::root()
{
    return tree_.root();
}

void iLQRTree::forward_tree_update(const double alpha)
{
    // Process from the end of the list, but start at the beginning. 
    // Each tuple is (parent_node_t, x_t).
    std::list<std::tuple<TreeNodePtr, Eigen::VectorXd>> to_process;
    const Eigen::VectorXd x0 = tree_.root()->item()->x();
    to_process.emplace_front(tree_.root(), x0);
    while (!to_process.empty())
    {
        auto &process_tuple = to_process.back();
        //TODO: Implement line search over this function for the forward pass.
        const TreeNodePtr& parent = std::get<0>(process_tuple);
        const std::shared_ptr<iLQRNode> parent_ilqr_node = parent->item();
        const Eigen::VectorXd &xt = std::get<1>(process_tuple);

        const Eigen::VectorXd ut = parent_ilqr_node->compute_control(xt, alpha);
        parent_ilqr_node->x() = xt;
        parent_ilqr_node->u() = ut;
        for (auto child : parent->children()) 
        {
            std::shared_ptr<iLQRNode> child_ilqr_node = child->item();
            const Eigen::VectorXd xt1 = child_ilqr_node->dynamics_func()(xt, ut);
            to_process.emplace_front(child, xt1);
        }

        to_process.pop_back();
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

data::Tree<SACV> iLQRTree::forward_pass(const Eigen::VectorXd &x0, const TreeNodePtr &top, const double top_alpha)
{
    data::Tree<SACV> output_tree; 

    // Tuple of parent, child, xt.
    using ProcessTuple = std::tuple<decltype(output_tree.root()), TreeNodePtr, Eigen::VectorXd>;
    std::list<ProcessTuple> to_process;
    static auto get_parent = [](const ProcessTuple& t) { return std::get<0>(t); };
    static auto get_node = [](const ProcessTuple &t) { return std::get<1>(t); };
    static auto get_x = [](const ProcessTuple &t) { return std::get<2>(t); };


    // If the passed in top pointer is null, then it will start at the top node. 
    const TreeNodePtr &start = top ? top : tree_.root();
    to_process.emplace_front(nullptr, start, x0);

    while (!to_process.empty())
    {
        ProcessTuple &process_tuple = to_process.back();
        const std::shared_ptr<iLQRNode> ilqr_node = get_node(process_tuple)->item();
        const Eigen::VectorXd &xt = get_x(process_tuple);

        Eigen::VectorXd ut;
        if (get_node(process_tuple) == start)
        {
            ut = ilqr_node->compute_control(xt, top_alpha);
        }
        else
        {
            ut = ilqr_node->compute_control(xt, 1.0);
        }
        std::shared_ptr<SACV> output(new SACV());;
        output->x = xt;
        output->u = ut;
        output->c = ilqr_node->cost_func()(xt, ut);
        output->value = output->c; // we will back up the child costs to this once we have them.
        output->probability = ilqr_node->probability();
        decltype(output_tree.root()) new_parent;
        // If the tree does not have a root yet, then construct a new tree with this as the root.
        if (!output_tree.root())
        {
            output_tree = data::Tree<SACV>(output);
            new_parent = output_tree.root();
        }
        else // Otherwise add this to its parent.
        {
            auto parent = get_parent(process_tuple);
            new_parent = output_tree.add_child(parent, output);
        }
        for (const auto &child : get_node(process_tuple)->children()) 
        {
            std::shared_ptr<iLQRNode> child_ilqr_node = child->item();
            const Eigen::VectorXd xt1 = child_ilqr_node->dynamics_func()(xt, ut);
            to_process.emplace_front(new_parent, child, xt1);
        }

        to_process.pop_back();
    }

    // Finally, run through the output tree backwards to compute the cost to go (bellman-backup) the costs.
    std::list<decltype(output_tree.root())> backwards_process = output_tree.leaf_nodes();
    while (!backwards_process.empty())
    {

        auto child = backwards_process.back();
        auto parent = child->parent();
        // This child's cost gets added to the parent's value.
        parent->item()->value += child->item()->probability*child->item()->value;
        // Add the parent to the front of the queue
        backwards_process.push_front(parent);
        backwards_process.pop_back();
    }

    return output_tree;
}

} // namespace ilqr
