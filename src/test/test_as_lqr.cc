//
// Tests the iLQR Tree with LQR parameters to confirm it gives the same answer.
//

#include <ilqr/ilqr_tree.hh>
#include <utils/debug_utils.hh>

namespace
{

ilqr::DynamicsFunc create_linear_dynamics(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
    return [&A, &B](const Eigen::VectorXd &x, const Eigen::VectorXd& u) -> Eigen::VectorXd
    {
        const int state_dim = x.size();
        IS_EQUAL(A.cols(), state_dim);
        IS_EQUAL(A.rows(), state_dim);
        IS_EQUAL(B.rows(), state_dim);
        IS_EQUAL(B.cols(), u.size());
        Eigen::VectorXd x_next = A*x + B*u;
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
        Eigen::VectorXd cost = (x.transpose()*Q*x + u.transpose()*R*u);
        IS_EQUAL(cost.size(), 1);
        return cost(0);
    };
}

}

// Simplest LQR formulation with a static linear dynamics and quadratic cost.
void simple_lqr()
{
    // Time horizon.
    const int T = 10;
    // State and control dimensions.
    const int state_dim = 3;
    const int control_dim = 2;

    // Define the dynamics.
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim);
    A(1, 1) = -0.5;
    A(2, 2) = 0.25;
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(state_dim, control_dim);
    B(0, 1) = 0.25;
    B(1, 1) = -0.3;
    B(2, 0) = 0.5;
    B(2, 1) = -0.1;
    const ilqr::DynamicsFunc linear_dyn = create_linear_dynamics(A, B);

    // Define the cost.
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Q(0, 0) = 5.0;
    Q(2, 2) = 10.0;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    R(0, 0) = 0.1;
    R(1, 1) = 1.0;
    const ilqr::CostFunc quad_cost = create_quadratic_cost(Q, R);

    // Create a list of initial states for the iLQR. 
    Eigen::VectorXd x0(state_dim);
    x0 << 3, 2, 1;
    std::vector<Eigen::VectorXd> xstars(T);
    std::vector<Eigen::VectorXd> ustars(T);
    Eigen::VectorXd xt = x0;
    for (int t = 0; t < T; ++t)
    {
        Eigen::VectorXd ut = Eigen::VectorXd::Random(control_dim);
        xstars[t] = xt;
        ustars[t] = ut;
        //xt = A*xt + B*ut;
        xt = linear_dyn(xt, ut);
    }

    // Add them to the tree.
    ilqr::iLQRTree ilqr_tree(state_dim, control_dim); 
    ilqr::TreeNodePtr last_tree_node = nullptr;
    for (int t = 0; t < T; ++t)
    {
        // Everything has probability 1.0 since we are making a chain.
        auto plan_node = ilqr_tree.make_plan_node(xstars[t], ustars[t], linear_dyn, quad_cost, 1.0);
        if (t == 0)
        {
            last_tree_node = ilqr_tree.add_root(plan_node);
        }
        else
        {
            auto new_nodes = ilqr_tree.add_nodes({plan_node}, last_tree_node);
            IS_EQUAL(new_nodes.size(), 1);
            last_tree_node = new_nodes[0];
        }
    }

}

int main()
{
    simple_lqr();

    return 0;
}
