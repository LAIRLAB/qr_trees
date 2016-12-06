//
// Tests the iLQR Tree with LQR parameters to confirm it gives the same answer.
//

#include <ilqr/ilqr_helpers.hh>
#include <ilqr/ilqr_tree.hh>
#include <lqr/LQR.hh>
#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

#include <numeric>

namespace
{

// Tolerance for checking double/floating point equality.
constexpr double WEAKER_TOL = 5e-3;
constexpr double TOL = 1e-5;
constexpr double TIGHTER_TOL = 1e-7;

ilqr::DynamicsFunc create_linear_dynamics(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
    return [&A, &B](const Eigen::VectorXd &x, const Eigen::VectorXd& u) -> Eigen::VectorXd
    {
        const int state_dim = x.size();
        IS_EQUAL(A.cols(), state_dim);
        IS_EQUAL(A.rows(), state_dim);
        IS_EQUAL(B.rows(), state_dim);
        IS_EQUAL(B.cols(), u.size());
        const Eigen::VectorXd x_next = A*x + B*u;
        IS_EQUAL(x_next.size(), state_dim);
        return x_next;
    };
}

ilqr::CostFunc create_quadratic_cost(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
{
    return [&Q, &R](const Eigen::VectorXd &x, const Eigen::VectorXd& u) -> double
    {
        const int state_dim = x.size();
        const int control_dim = u.size();
        IS_EQUAL(Q.cols(), state_dim);
        IS_EQUAL(Q.rows(), state_dim);
        IS_EQUAL(R.rows(), control_dim);
        IS_EQUAL(R.cols(), control_dim);
        const Eigen::VectorXd cost = 0.5*(x.transpose()*Q*x + u.transpose()*R*u);
        IS_EQUAL(cost.size(), 1);
        const double c = cost(0);
        return c;
    };
}

Eigen::MatrixXd make_random_psd(const int dim, const double min_eig_val)
{
    constexpr double MIN_CON = 1e1;
    Eigen::MatrixXd tmp = 10.*Eigen::MatrixXd::Random(dim, dim);
    Eigen::MatrixXd symmetric_mat = (tmp + tmp.transpose())/2.0;

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetric_mat);
    Eigen::VectorXd evals = es.eigenvalues();
    const Eigen::MatrixXd evecs = es.eigenvectors();
    evals = evals.cwiseMax(Eigen::VectorXd::Constant(dim, min_eig_val));
    const double condition = evals(evals.size()-1) / evals(0);
    if (condition < MIN_CON)
    {
        evals(evals.size()-1) = evals(0)*MIN_CON;
    }
    return evecs * evals.asDiagonal() * evecs.transpose();
}


std::vector<ilqr::TreeNodePtr> construct_chain(const std::vector<Eigen::VectorXd>& xstars, 
        const std::vector<Eigen::VectorXd>& ustars, const ilqr::DynamicsFunc &dyn, const ilqr::CostFunc &cost,  
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

}

// Initialize iLQR with LQR intialization on a linear dynamics and quadratic
// cost. 
void test_with_lqr_initialization(const int state_dim, 
        const int control_dim, 
        const int T)
{
    std::srand(1);

    // Define the dynamics.
    const Eigen::MatrixXd A = Eigen::MatrixXd::Random(state_dim, state_dim);
    const Eigen::MatrixXd B = Eigen::MatrixXd::Random(state_dim, control_dim);
    const ilqr::DynamicsFunc linear_dyn = create_linear_dynamics(A, B);

    // Define the cost.
    const Eigen::MatrixXd Q = make_random_psd(state_dim, 1e-11);
    const Eigen::MatrixXd R = make_random_psd(control_dim, 1e-1);
    const ilqr::CostFunc quad_cost = create_quadratic_cost(Q, R);

    // Create a list of initial states for the iLQR. 
    Eigen::VectorXd x0 = Eigen::VectorXd::Random(state_dim);
    
    // Storage for regular LQR.
    std::vector<Eigen::VectorXd> lqr_states, lqr_controls;
    std::vector<double> lqr_costs;

    // Compute the true LQR result.
    lqr::LQR lqr(A, B, Q, R, T);
    lqr.solve();
    lqr.forward_pass(x0, lqr_costs, lqr_states, lqr_controls);

    // Storage for iLQR
    std::vector<Eigen::VectorXd> ilqr_states, ilqr_controls;
    std::vector<double> ilqr_costs;
    ilqr::iLQRTree ilqr_tree(state_dim, control_dim);
    std::vector<ilqr::TreeNodePtr> ilqr_chain = construct_chain(lqr_states, lqr_controls,
        linear_dyn, quad_cost, ilqr_tree);
    ilqr_tree.bellman_tree_backup();
    ilqr_tree.forward_tree_update(1.0);
    get_forward_pass_info(ilqr_chain, quad_cost, ilqr_states, ilqr_controls, ilqr_costs);

    // Recompute costs to check that the costs being returned are correct.
    double ilqr_cost = 0;
    double lqr_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        const Eigen::VectorXd lqr_x = lqr_states[t];
        const Eigen::VectorXd lqr_u = lqr_controls[t];
        const Eigen::VectorXd ilqr_x = ilqr_states[t];
        const Eigen::VectorXd ilqr_u = ilqr_controls[t];
        //PRINT("t=" << t << ": ilqr_x " << ilqr_x.transpose());
        //WARN("   : lqr_x " << lqr_x.transpose());
        //WARN("   : ilqr_u " << ilqr_u.transpose());
        //WARN("   : lqr_u " << lqr_u.transpose());

        IS_TRUE(math::is_equal(lqr_x, ilqr_x, TOL));
        IS_TRUE(math::is_equal(lqr_u, ilqr_u, TOL));

        ilqr_cost += quad_cost(ilqr_x, ilqr_u);
        lqr_cost += quad_cost(lqr_x, lqr_u);
        IS_ALMOST_EQUAL(quad_cost(ilqr_x, ilqr_u), ilqr_costs[t], TOL);
        IS_ALMOST_EQUAL(ilqr_costs[t], lqr_costs[t], TOL);
        IS_ALMOST_EQUAL(ilqr_cost, lqr_cost, TOL);
    }

    const double lqr_total_cost = std::accumulate(lqr_costs.begin(), lqr_costs.end(), 0.0);
    const double ilqr_total_cost = std::accumulate(ilqr_costs.begin(), ilqr_costs.end(), 0.0);
    IS_ALMOST_EQUAL(lqr_total_cost, ilqr_total_cost, TOL);
    IS_ALMOST_EQUAL(lqr_total_cost, lqr_cost, TOL);
    IS_ALMOST_EQUAL(ilqr_total_cost, ilqr_cost, TOL);

    // Confirm another backwards and forwards pass does not change the results.
    std::vector<Eigen::VectorXd> ilqr_states_2, ilqr_controls_2;
    std::vector<double> ilqr_costs_2;
    ilqr_tree.bellman_tree_backup();
    ilqr_tree.forward_tree_update(1.0);
    get_forward_pass_info(ilqr_chain, quad_cost, ilqr_states_2, ilqr_controls_2, ilqr_costs_2);
    const double ilqr_total_cost_2 = std::accumulate(ilqr_costs_2.begin(), ilqr_costs_2.end(), 0.0);
    IS_ALMOST_EQUAL(ilqr_total_cost_2, ilqr_total_cost, TIGHTER_TOL);
    IS_TRUE(std::equal(ilqr_costs.begin(), ilqr_costs.end(), 
                ilqr_costs_2.begin(), [](const double &a, const double &b) { 
                return std::abs(a-b) <= TOL;
            }));
    IS_TRUE(std::equal(ilqr_states.begin(), ilqr_states.end(), ilqr_states_2.begin(), 
            [](const Eigen::VectorXd &a, const Eigen::VectorXd &b) { 
                return math::is_equal(a, b, TOL);
            }));
    IS_TRUE(std::equal(ilqr_controls.begin(), ilqr_controls.end(), ilqr_controls_2.begin(), 
            [](const Eigen::VectorXd &a, const Eigen::VectorXd &b) { 
                return math::is_equal(a, b, TOL);
            }));
}

// iLQR even initialized at different states and controls than the true LQR 
// should converge on 1 iteration (perfect Newton step).
void test_converge_to_lqr(const int state_dim, const int control_dim, const int T)
{
    std::srand(2);

    // Define the dynamics.
    const Eigen::MatrixXd A = Eigen::MatrixXd::Random(state_dim, state_dim);
    const Eigen::MatrixXd B = Eigen::MatrixXd::Random(state_dim, control_dim);
    //const Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim);
    //const Eigen::MatrixXd B = Eigen::MatrixXd::Identity(state_dim, control_dim);
    const ilqr::DynamicsFunc linear_dyn = create_linear_dynamics(A, B);

    // Define the cost.
    const Eigen::MatrixXd Q = make_random_psd(state_dim, 1e-11);
    const Eigen::MatrixXd R = make_random_psd(control_dim, 1e-3);
    //const Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    //const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    const ilqr::CostFunc quad_cost = create_quadratic_cost(Q, R);


    // Create a list of initial states for the iLQR. 
    Eigen::VectorXd x0 = Eigen::VectorXd::Random(state_dim);
    
    // Storage for regular LQR.
    std::vector<Eigen::VectorXd> lqr_states, lqr_controls;
    std::vector<double> lqr_costs;

    // Compute the true LQR result.
    lqr::LQR lqr(A, B, Q, R, T);
    lqr.solve();
    lqr.forward_pass(x0, lqr_costs, lqr_states, lqr_controls);

    // Storage for iLQR
    std::vector<Eigen::VectorXd> ilqr_init_states, ilqr_init_controls;
    std::vector<Eigen::VectorXd> ilqr_states, ilqr_controls;
    std::vector<double> ilqr_costs;
    Eigen::VectorXd xt = x0;
    for (int t = 0; t < T; ++t)
    {
        ilqr_init_states.push_back(xt);
        const Eigen::VectorXd ut = Eigen::VectorXd::Random(control_dim);
        ilqr_init_controls.push_back(ut);
        xt = linear_dyn(xt, ut);
    }

    ilqr::iLQRTree ilqr_tree(state_dim, control_dim);
    std::vector<ilqr::TreeNodePtr> ilqr_chain = construct_chain(ilqr_init_states, ilqr_init_controls,
        linear_dyn, quad_cost, ilqr_tree);
    ilqr_tree.bellman_tree_backup();
    ilqr_tree.forward_tree_update(1.0);
    get_forward_pass_info(ilqr_chain, quad_cost, ilqr_states, ilqr_controls, ilqr_costs);

    // Recompute costs to check that the costs being returned are correct.
    double ilqr_cost = 0;
    double lqr_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        const Eigen::VectorXd lqr_x = lqr_states[t];
        const Eigen::VectorXd lqr_u = lqr_controls[t];
        const Eigen::VectorXd ilqr_x = ilqr_states[t];
        const Eigen::VectorXd ilqr_u = ilqr_controls[t];

        
        //PRINT("t=" << t << ": ilqr_x " << ilqr_x.transpose());
        //WARN("   : lqr_x " << lqr_x.transpose());
        //WARN("   : ilqr_x_orig " << ilqr_init_states[t].transpose());
        //WARN("   : ilqr_u " << ilqr_u.transpose());
        //WARN("   : lqr_u " << lqr_u.transpose());

        //if (t < T-1)
        //{
        //    auto xt1_lqr = linear_dyn(lqr_x, lqr_u);
        //    auto xt1_ilqr = ilqr_chain[t]->item()->dynamics_func()(ilqr_x, ilqr_u);
        //    WARN("   : ilqr_xt1 " << xt1_ilqr.transpose());
        //    WARN("   : lqr_xt1 " << xt1_lqr.transpose());
        //}
        

        IS_TRUE(math::is_equal(lqr_x, ilqr_x, WEAKER_TOL));
        IS_TRUE(math::is_equal(lqr_u, ilqr_u, WEAKER_TOL));


        ilqr_cost += quad_cost(ilqr_x, ilqr_u);
        lqr_cost += quad_cost(lqr_x, lqr_u);
        IS_ALMOST_EQUAL(quad_cost(ilqr_x, ilqr_u), ilqr_costs[t], TOL);
        IS_ALMOST_EQUAL(ilqr_costs[t], lqr_costs[t], WEAKER_TOL);
        IS_ALMOST_EQUAL(ilqr_cost, lqr_cost, WEAKER_TOL);
    }

    const double lqr_total_cost = std::accumulate(lqr_costs.begin(), lqr_costs.end(), 0.0);
    const double ilqr_total_cost = std::accumulate(ilqr_costs.begin(), ilqr_costs.end(), 0.0);
    IS_ALMOST_EQUAL(lqr_total_cost, ilqr_total_cost, TOL);
    IS_ALMOST_EQUAL(lqr_total_cost, lqr_cost, TOL);
    IS_ALMOST_EQUAL(ilqr_total_cost, ilqr_cost, TOL);

    // Confirm another backwards and forwards pass does not change the results.
    std::vector<Eigen::VectorXd> ilqr_states_2, ilqr_controls_2;
    std::vector<double> ilqr_costs_2;
    ilqr_tree.bellman_tree_backup();
    ilqr_tree.forward_tree_update(1.0);
    get_forward_pass_info(ilqr_chain, quad_cost, ilqr_states_2, ilqr_controls_2, ilqr_costs_2);
    const double ilqr_total_cost_2 = std::accumulate(ilqr_costs_2.begin(), ilqr_costs_2.end(), 0.0);
    IS_ALMOST_EQUAL(ilqr_total_cost_2, ilqr_total_cost, TIGHTER_TOL);
    IS_TRUE(std::equal(ilqr_costs.begin(), ilqr_costs.end(), 
                ilqr_costs_2.begin(), [](const double &a, const double &b) { 
                return std::abs(a-b) <= TOL;
            }));
    IS_TRUE(std::equal(ilqr_states.begin(), ilqr_states.end(), ilqr_states_2.begin(), 
            [](const Eigen::VectorXd &a, const Eigen::VectorXd &b) { 
                return math::is_equal(a, b, WEAKER_TOL);
            }));

    IS_TRUE(std::equal(ilqr_controls.begin(), ilqr_controls.end(), ilqr_controls_2.begin(), 
            [](const Eigen::VectorXd &a, const Eigen::VectorXd &b) { 
                return math::is_equal(a, b, WEAKER_TOL);
            }));
}

int main()
{
    // Should work with square and non-square dimensions. 
    // Should work with many timesteps.
    test_with_lqr_initialization(5, 2, 8);
    test_with_lqr_initialization(5, 5, 2);
    test_with_lqr_initialization(5, 2, 2);
    test_with_lqr_initialization(5, 2, 8);
    test_with_lqr_initialization(5, 2, 150);
    test_with_lqr_initialization(1, 1, 150);
    test_with_lqr_initialization(1, 1, 2);
    // Should not work with only 1 timstep. 
    DOES_THROW(test_with_lqr_initialization(3, 2, 1));

    test_converge_to_lqr(8,2,7);
    test_converge_to_lqr(5,5,8);
    test_converge_to_lqr(3,2,4);
    test_converge_to_lqr(3,2,8);
    test_converge_to_lqr(3,2,50);
    test_converge_to_lqr(1,1,8);
    // Should not work with only 1 timstep. 
    DOES_THROW(test_converge_to_lqr(3, 2, 1));
}

