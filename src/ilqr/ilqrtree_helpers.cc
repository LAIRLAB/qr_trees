
#include <ilqr/ilqrtree_helpers.hh>

namespace ilqr
{

std::vector<ilqr::TreeNodePtr> construct_chain(
        const std::vector<Eigen::VectorXd>& xstars, 
        const std::vector<Eigen::VectorXd>& ustars, 
        const ilqr::DynamicsFunc &dyn, 
        const ilqr::CostFunc &cost,  
        ilqr::iLQRTree& ilqr_tree)
{
    const int T = xstars.size();
    IS_GREATER(T, 0);
    IS_EQUAL(xstars.size(), ustars.size());
    std::vector<ilqr::TreeNodePtr> tree_nodes(T);
    auto plan_node = ilqr_tree.make_ilqr_node(xstars[0], ustars[0], dyn, cost, 1.0);
    ilqr::TreeNodePtr last_tree_node = ilqr_tree.add_root(plan_node);
    tree_nodes[0] = last_tree_node;
    IS_TRUE(tree_nodes[0])
    for (int t = 1; t < T; ++t)
    {
        // Everything has probability 1.0 since we are making a chain.
        auto plan_node = ilqr_tree.make_ilqr_node(xstars[t], ustars[t], dyn, cost, 1.0);
        auto new_nodes = ilqr_tree.add_nodes({plan_node}, last_tree_node);
        IS_EQUAL(new_nodes.size(), 1);
        last_tree_node = new_nodes[0];
        tree_nodes[t] = last_tree_node;
        IS_TRUE(tree_nodes[t]);
    }

    return tree_nodes;
}

void get_forward_pass_info(const std::vector<ilqr::TreeNodePtr> &chain,
                           const ilqr::CostFunc &cost,  
                           std::vector<Eigen::VectorXd> &states, 
                           std::vector<Eigen::VectorXd> &controls, 
                           std::vector<double> &costs)
{
    const int T = chain.size();
    IS_GREATER(T, 0);
    states.clear(); states.reserve(T);
    controls.clear(); controls.reserve(T);
    costs.clear(); costs.reserve(T);
    for (const auto& tree_node : chain)
    {
        IS_TRUE(tree_node);
        const auto ilqr_node = tree_node->item();
        IS_TRUE(ilqr_node);
        states.push_back(ilqr_node->x());
        controls.push_back(ilqr_node->u());
        costs.push_back(cost(ilqr_node->x(), ilqr_node->u()));
    }
}

} // namespace ilqr

