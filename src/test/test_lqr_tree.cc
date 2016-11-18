//
// Tests the LQR Tree with LQR parameters to confirm it gives the same answer.
//

#include <lqr/lqr_tree.hh>
#include <lqr/LQR.hh>
#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace
{

}

// Simplest LQR formulation with a static linear dynamics and quadratic cost.
void simple_lqr()
{
    // Time horizon.
    constexpr int T = 6;
    // State and control dimensions.
    constexpr int state_dim = 3;
    constexpr int control_dim = 3;

    // Define the dynamics.
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim);
    //A(1, 1) = -0.5;
    //A(2, 2) = 0.25;
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(state_dim, control_dim);
    //B(0, 1) = 0.25;
    //B(1, 1) = -0.3;
    //B(2, 0) = 0.5;
    //B(2, 1) = -0.1;

    // Define the cost.
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    //Q(0, 0) = 5.0;
    //Q(2, 2) = 10.0;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    //R(0, 0) = 0.1;
    //R(1, 1) = 1.0;

    // Add them to the tree.
    lqr::LQRTree lqr_tree(state_dim, control_dim); 
    lqr::TreeNodePtr last_tree_node = nullptr;
    std::vector<lqr::TreeNodePtr> tree_nodes;
    for (int t = 0; t < T; ++t)
    {
        // Everything has probability 1.0 since we are making a chain.
        auto plan_node = lqr_tree.make_plan_node(A, B, Q, R, 1.0);
        if (t == 0)
        {
            last_tree_node = lqr_tree.add_root(plan_node);
        }
        else
        {
            auto new_nodes = lqr_tree.add_nodes({plan_node}, last_tree_node);
            IS_EQUAL(new_nodes.size(), 1);
            last_tree_node = new_nodes[0];
        }
        tree_nodes.push_back(last_tree_node);
    }

    Eigen::VectorXd x0(state_dim);
    x0 << 3, 2, 1;
    //x0.setZero();
    
    // Optimize the Tree
    lqr_tree.bellman_tree_backup();
    lqr_tree.forward_pass(x0);

    // Compute the true LQR result.
    lqr::LQR lqr(A, B, Q, R, T);
    lqr.solve();

    std::vector<Eigen::VectorXd> lqr_states;
    std::vector<Eigen::VectorXd> lqr_controls;
    std::vector<double> lqr_costs;
    lqr.forward_pass(x0, lqr_costs, lqr_states, lqr_controls);
    for (int t = 0; t < T; ++t)
    {
        auto plan_node = tree_nodes[t]->item();
        const auto node_x = plan_node->x().topRows(state_dim);
        const auto true_x = lqr_states[t];
        const auto node_u = plan_node->u();
        const auto true_u = lqr_controls[t];
        IS_TRUE(math::is_equal(node_x, true_x, 1e-8));
        IS_TRUE(math::is_equal(node_u, true_u, 1e-8));
        WARN("t=" << t << ", xtrue: " << true_x.transpose());
        WARN("   " << t << ", xtree: " << node_x.transpose());
        WARN("t=" << t << ", utrue: " << node_u.transpose());
        WARN("  " << t << ", utree: " << true_u.transpose());
    }

}

int main()
{
    simple_lqr();

    return 0;
}
