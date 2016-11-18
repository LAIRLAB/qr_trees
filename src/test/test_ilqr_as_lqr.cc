//
// Tests the iLQR Tree with LQR parameters to confirm it gives the same answer.
//

#include <ilqr/iLQR.hh>
#include <lqr/LQR.hh>
#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

#include <numeric>

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
        Eigen::VectorXd cost = 0.5*(x.transpose()*Q*x + u.transpose()*R*u);
        IS_EQUAL(cost.size(), 1);
        return cost(0);
    };
}

Eigen::MatrixXd make_random_psd(const int dim, const double min_eig_val)
{
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Random(dim, dim);
    Eigen::MatrixXd tmp2 = (tmp + tmp.transpose())/2.0;
    return math::project_to_psd(tmp2, min_eig_val);
}

}

// Simplest LQR formulation with a static linear dynamics and quadratic cost.
void test_with_lqr_initialization()
{
    std::srand(1);

    // Time horizon.
    constexpr int T = 6;
    // State and control dimensions.
    constexpr int state_dim = 3;
    constexpr int control_dim = 3;

    // Define the dynamics.
    const Eigen::MatrixXd A = Eigen::MatrixXd::Random(state_dim, state_dim);
    //const Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(state_dim, control_dim);
    //B(0, 1) = 1;
    const ilqr::DynamicsFunc linear_dyn = create_linear_dynamics(A, B);

    // Define the cost.
    const Eigen::MatrixXd Q = make_random_psd(state_dim, 1e-11);
    const Eigen::MatrixXd R = make_random_psd(control_dim, 1e-7);
    const ilqr::CostFunc quad_cost = create_quadratic_cost(Q, R);


    // Create a list of initial states for the iLQR. 
    Eigen::VectorXd x0(state_dim);
    x0 << 3, 2, 1;
    
    // Storage for regular LQR.
    std::vector<Eigen::VectorXd> lqr_states;
    std::vector<Eigen::VectorXd> lqr_controls;
    std::vector<double> lqr_costs;
    std::vector<Eigen::MatrixXd> Vs_lqr;

    // Compute the true LQR result.
    lqr::LQR lqr(A, B, Q, R, T);
    lqr.solve(Vs_lqr);
    lqr.forward_pass(x0, lqr_costs, lqr_states, lqr_controls);

    // Storage for iLQR
    std::vector<Eigen::VectorXd> ilqr_states;
    std::vector<Eigen::VectorXd> ilqr_controls;
    std::vector<double> ilqr_costs;
    std::vector<Eigen::MatrixXd> Vs_ilqr;
    std::vector<Eigen::MatrixXd> Gs_ilqr;

    ilqr::iLQR ilqr(linear_dyn, quad_cost, lqr_states, lqr_controls);
    ilqr.backwards_pass(Vs_ilqr, Gs_ilqr);
    ilqr.forward_pass(ilqr_costs, ilqr_states, ilqr_controls);

    for (int t = 0; t < T; ++t)
    {
        const auto lqr_x = lqr_states[t].transpose();
        const auto lqr_u = lqr_controls[t].transpose();
        const auto ilqr_x = ilqr_states[t].transpose();
        const auto ilqr_u = ilqr_controls[t].transpose();
        WARN(",t=" << t << ", xlqr: " << lqr_x);
        WARN("  " << t << ", xilqr: " << ilqr_x);
        WARN(",t=" << t << ", ulqr: " << lqr_u);
        WARN("  " << t << ", uilqr: " << ilqr_u);
    }
    WARN("\n")

    const double lqr_total_cost = std::accumulate(lqr_costs.begin(), lqr_costs.end(), 0.0);
    const double ilqr_total_cost = std::accumulate(ilqr_costs.begin(), ilqr_costs.end(), 0.0);
    PRINT("Total cost LQR:  " << lqr_total_cost);
    PRINT("Total cost iLQR: " << ilqr_total_cost);
}

int main()
{
    test_with_lqr_initialization();

    return 0;
}
