
#include <experiments/simulators/pendulum.hh>
#include <ilqr/iLQR.hh>
#include <ilqr/ilqr_tree.hh>
#include <utils/debug_utils.hh>

namespace 
{

constexpr int STATE_DIM = 2;
constexpr int CONTROL_DIM = 1;

Eigen::VectorXd goal(STATE_DIM);

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

ilqr::CostFunc create_quadratic_cost(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
{
    const Eigen::VectorXd &g = goal;
    return [&Q, &R, &g](const Eigen::VectorXd &x, const Eigen::VectorXd& u) -> double
    {
        const int state_dim = x.size();
        const int control_dim = u.size();
        IS_EQUAL(Q.cols(), state_dim); IS_EQUAL(Q.rows(), state_dim);
        IS_EQUAL(R.rows(), control_dim); IS_EQUAL(R.cols(), control_dim);
        const Eigen::VectorXd xdiff = goal - x;
        const Eigen::VectorXd cost = 0.5*(xdiff.transpose()*Q*xdiff + u.transpose()*R*u);
        IS_EQUAL(cost.size(), 1);
        const double c = cost(0);
        return c;
    };
}

double compute_total_cost(const ilqr::CostFunc &cost,  
                           std::vector<Eigen::VectorXd> &states, 
                           std::vector<Eigen::VectorXd> &controls)  
{
    IS_EQUAL(states.size(), controls.size());
    const int T = states.size();
    double total_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        total_cost += cost(states[t], controls[t]);
    }
    return total_cost;
}

} // namespace

void control_pendulum_as_chain(const int T, const double dt, const Eigen::VectorXd &x0 = Eigen::VectorXd::Zero(2))
{
    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    // Constants for the pendulum.
    constexpr double LENGTH = 1.0;
    constexpr double DAMPING_COEFF = 0.0;
    IS_EQUAL(x0.size(), STATE_DIM);
    goal << M_PI, 0;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    Q(1,1) = 1e-5;

    const Eigen::MatrixXd R = 1e-5*Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
    const ilqr::CostFunc quad_cost = create_quadratic_cost(Q, R);

    const ilqr::DynamicsFunc dynamics 
        = simulators::pendulum::make_discrete_dynamics_func(dt, LENGTH, DAMPING_COEFF);

    std::vector<Eigen::VectorXd> ilqr_init_states, ilqr_init_controls;
    for (int t = 0; t < T; ++t)
    {
        const Eigen::VectorXd target_state = static_cast<double>(t)/static_cast<double>(T-1) * (goal - x0);
        ilqr_init_states.push_back(target_state);
        const Eigen::VectorXd target_control = Eigen::VectorXd::Zero(CONTROL_DIM);
        ilqr_init_controls.push_back(target_control);
    }

    ilqr::iLQRTree ilqr_tree(STATE_DIM, CONTROL_DIM);
    std::vector<ilqr::TreeNodePtr> ilqr_chain = construct_chain(ilqr_init_states, ilqr_init_controls,
        dynamics, quad_cost, ilqr_tree);

    //ilqr::iLQR ilqr(dynamics, quad_cost, ilqr_init_states, ilqr_init_controls);
    for (int i = 0; i < 5; ++i)
    {
        SUCCESS("i:" << i)
        std::vector<Eigen::VectorXd> ilqr_states, ilqr_controls;
        std::vector<double> ilqr_costs;
        //ilqr.backwards_pass();
        //ilqr.forward_pass(ilqr_costs, ilqr_states, ilqr_controls, true);

        ilqr_tree.bellman_tree_backup();
        ilqr_tree.forward_tree_update(1.0);
        get_forward_pass_info(ilqr_chain, quad_cost, ilqr_states, ilqr_controls, ilqr_costs);

        // Recompute costs to check that the costs being returned are correct.
        //for (int t = 0; t < T; ++t)
        //{
        //    const Eigen::VectorXd ilqr_x = ilqr_states[t];
        //    //const Eigen::VectorXd ilqr_u = ilqr_controls[t];
        //    PRINT(" t=" << t << ": " << ilqr_x.transpose());
        //}
        PRINT(" x(" << T-1 << ")= " << ilqr_states[T-1].transpose());
        WARN(" Total cost: " << compute_total_cost(quad_cost, ilqr_states, ilqr_controls));
    }
}

int main()
{
    control_pendulum_as_chain(20, 0.1);
    return 0;
}


