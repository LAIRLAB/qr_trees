//
// Tests the iLQR Taylor expansions of the dynamics and the cost function.
//

#include <ilqr/ilqr_helpers.hh>
#include <utils/math_utils.hh>
#include <utils/debug_utils.hh>

#include <Eigen/Dense>

void test_nonlinear_dynamics_linearization(const int state_dim, const int control_dim)
{
    std::srand(1);
    const Eigen::MatrixXd A_true = Eigen::MatrixXd::Random(state_dim, state_dim);
    const Eigen::MatrixXd B_true = Eigen::MatrixXd::Random(state_dim, control_dim);
    auto dyn = [&A_true, &B_true](const Eigen::VectorXd &x, const Eigen::VectorXd& u) -> Eigen::VectorXd
    {
        const int state_dim = x.size();
        Eigen::VectorXd x_next = A_true * x.array().cos().matrix() + B_true * u.array().sin().matrix();
        IS_EQUAL(x_next.size(), state_dim);
        return x_next;
    };
    ilqr::TaylorExpansion expansion;
    expansion.x = Eigen::VectorXd::Random(state_dim);
    expansion.u = Eigen::VectorXd::Random(control_dim);
    ilqr::update_dynamics(dyn, expansion);

    // Next linearization point under true dynamics.
    const Eigen::VectorXd xt1_star = dyn(expansion.x, expansion.u);

    Eigen::MatrixXd rep_x = (expansion.x.replicate(1, state_dim)).transpose();
    Eigen::MatrixXd rep_u = (expansion.u.replicate(1, state_dim)).transpose();
    Eigen::MatrixXd J_x = -1.*(A_true.array()*rep_x.array().sin()); 
    Eigen::MatrixXd J_u = B_true.array()*rep_u.array().cos(); 
    IS_TRUE(math::is_equal(expansion.dynamics.A, J_x, 1e-3)); // high tolerance since finite differencing is not that great.
    IS_TRUE(math::is_equal(expansion.dynamics.B, J_u, 1e-3));

    const Eigen::VectorXd xt_test = expansion.x + 0.01*Eigen::VectorXd::Random(state_dim);
    const Eigen::VectorXd ut_test = expansion.u + 0.01*Eigen::VectorXd::Random(control_dim);

    Eigen::VectorXd xt_diff = xt_test- expansion.x;
    const Eigen::VectorXd ut_diff = ut_test - expansion.u;

    // Use the linearization to predict the next x_{t+1}. 
    const Eigen::VectorXd xt1_diff = expansion.dynamics.A*xt_diff + expansion.dynamics.B*ut_diff;
    IS_LESS_EQUAL(((xt1_diff + xt1_star) - dyn(xt_test, ut_test)).norm(), 1e-3);
}

void test_linear_dynamics_linearization(const int state_dim, const int control_dim)
{
    IS_GREATER(state_dim, 0);
    IS_GREATER(control_dim, 0);

    std::srand(2);

    // Test with linear dynamics first. The taylor expansion should produce
    // nearly exactly the same result.
    const Eigen::MatrixXd A_true = Eigen::MatrixXd::Random(state_dim, state_dim);
    const Eigen::MatrixXd B_true = Eigen::MatrixXd::Random(state_dim, control_dim);
    auto linear_dyn = [&A_true, &B_true](const Eigen::VectorXd &x, const Eigen::VectorXd& u) -> Eigen::VectorXd
    {
        const int state_dim = x.size();
        Eigen::VectorXd x_next = A_true*x + B_true*u;
        IS_EQUAL(x_next.size(), state_dim);
        return x_next;
    };
    
    ilqr::TaylorExpansion expansion;
    expansion.x = Eigen::VectorXd::Random(state_dim);
    expansion.u = Eigen::VectorXd::Random(control_dim);
    ilqr::update_dynamics(linear_dyn, expansion);

    IS_TRUE(math::is_equal(expansion.dynamics.A, A_true, 1e-10)); 
    IS_TRUE(math::is_equal(expansion.dynamics.B, B_true, 1e-10));

    // Next linearization point under true dynamics.
    const Eigen::VectorXd xt1_star = linear_dyn(expansion.x, expansion.u);

    const Eigen::VectorXd xt_test= Eigen::VectorXd::Random(state_dim);
    const Eigen::VectorXd ut_test = Eigen::VectorXd::Random(control_dim);

    Eigen::VectorXd xt_diff = xt_test- expansion.x;
    const Eigen::VectorXd ut_diff = ut_test - expansion.u;

    // Use the linearization to predict the next x_{t+1}. 
    const Eigen::VectorXd xt1_diff = expansion.dynamics.A*xt_diff + expansion.dynamics.B*ut_diff;

    IS_TRUE(math::is_equal(xt1_diff + xt1_star, linear_dyn(xt_test, ut_test), 1e-7));
}

void test_quadratic_cost_taylor_expansion(const int state_dim, const int control_dim)
{
    IS_GREATER(state_dim, 0);
    IS_GREATER(control_dim, 0);

    std::srand(3);

    // Test with linear dynamics first. The taylor expansion should produce
    // nearly exactly the same result.
    Eigen::MatrixXd Q_true = Eigen::MatrixXd::Random(state_dim, state_dim);
    Eigen::MatrixXd R_true = Eigen::MatrixXd::Random(control_dim, control_dim);
    Q_true = math::project_to_psd((Q_true + Q_true.transpose()), 1e-5);
    R_true = math::project_to_psd((R_true + R_true.transpose()), 1e-5);
    auto cost_func = [&Q_true, &R_true](const Eigen::VectorXd &x, const Eigen::VectorXd& u) -> double
    {
        Eigen::VectorXd cost = 0.5*(x.transpose()*Q_true*x + u.transpose()*R_true*u);
        IS_EQUAL(cost.size(), 1);
        return cost(0);
    };
    // Compute the Taylor expansion at a nominal point.
    ilqr::TaylorExpansion expansion;
    expansion.x = Eigen::VectorXd::Random(state_dim);
    expansion.u = Eigen::VectorXd::Random(control_dim);
    ilqr::update_cost(cost_func, expansion);

    IS_TRUE(math::is_equal(expansion.cost.Q, Q_true, 1e-4));
    IS_TRUE(math::is_equal(expansion.cost.R, R_true, 1e-4));
    IS_TRUE(math::is_equal(expansion.cost.P, Eigen::MatrixXd::Zero(state_dim, control_dim), 1e-4)); 
    IS_TRUE(math::is_equal(expansion.cost.g_x,  Q_true*expansion.x, 1e-10));
    IS_TRUE(math::is_equal(expansion.cost.g_u,  R_true*expansion.u, 1e-3)); // high tolerance since numerical error from finite differencing.

    // Next linearization point under true dynamics.
    const Eigen::VectorXd xt_test= Eigen::VectorXd::Random(state_dim);
    const Eigen::VectorXd ut_test = Eigen::VectorXd::Random(control_dim);

    Eigen::VectorXd xt_diff = Eigen::VectorXd::Ones(state_dim);
    xt_diff = xt_test- expansion.x;
    const Eigen::VectorXd ut_diff = ut_test - expansion.u;
    Eigen::VectorXd ct_pred = 0.5*(xt_diff.transpose() * expansion.cost.Q * xt_diff)
            + (xt_diff.transpose()*expansion.cost.P*ut_diff)
            + 0.5*(ut_diff.transpose() * expansion.cost.R * ut_diff)
            + (expansion.cost.g_u.transpose() * ut_diff)
            + (expansion.cost.g_x.transpose() * xt_diff);
    ct_pred.array() += expansion.cost.c;
    IS_EQUAL(ct_pred.size(), 1); 
    IS_ALMOST_EQUAL(ct_pred(0,0), cost_func(xt_test, ut_test), 1e-5);
}

void test_exp_cost_taylor_expansion(const int state_dim, const int control_dim)
{
    IS_GREATER(state_dim, 0);
    IS_GREATER(control_dim, 0);

    std::srand(3);

    // Test with linear dynamics first. The taylor expansion should produce
    // nearly exactly the same result.
    Eigen::MatrixXd wx = Eigen::VectorXd::Random(state_dim);
    Eigen::MatrixXd wu = Eigen::VectorXd::Random(control_dim);
    auto cost_func = [&wx, &wu](const Eigen::VectorXd &x, const Eigen::VectorXd& u) -> double
    {
        const double norm_sq = x.squaredNorm();
        const Eigen::VectorXd term = wx.transpose() * x + wu.transpose() * u * 0.5*norm_sq;
        IS_EQUAL(term.size(), 1);
        const double cost = std::exp(-1.*term(0));
        return cost;
    };

    // Compute the Taylor expansion at a nominal point.
    ilqr::TaylorExpansion expansion;
    expansion.x = Eigen::VectorXd::Random(state_dim);
    expansion.u = Eigen::VectorXd::Random(control_dim);
    ilqr::update_cost(cost_func, expansion);

    double ct_xu = cost_func(expansion.x, expansion.u);
    const Eigen::VectorXd g_x_true = ct_xu * (-1.*wx - (wu.transpose()*expansion.u)[0]*expansion.x);
    const Eigen::VectorXd g_u_true = ct_xu * (-0.5*wu*expansion.x.squaredNorm());
    const Eigen::MatrixXd Hxx_true = ct_xu * -1.0*(wu.transpose()*expansion.u)[0]*Eigen::MatrixXd::Identity(state_dim, state_dim).array() 
        + (1.0/ct_xu) * (g_x_true * g_x_true.transpose()).array();
    const Eigen::MatrixXd Huu_true =  1.0/ct_xu * (g_u_true * g_u_true.transpose()).array();
    const Eigen::MatrixXd Hux_true = ct_xu * (-1.0*wu*expansion.x.transpose()).array() + (1.0/ct_xu) * (g_u_true * g_x_true.transpose()).array(); 

    // Since we may need to project to PSD cone, construct and project it.
    Eigen::MatrixXd Q_true = math::project_to_psd(Hxx_true, 1e-11);
    math::check_psd(Q_true, 1e-12);

    Eigen::MatrixXd R_true = math::project_to_psd(Huu_true, 1e-8);
    math::check_psd(R_true, 1e-9);

    IS_TRUE(math::is_equal(expansion.cost.Q, Q_true, 1e-4));
    IS_TRUE(math::is_equal(expansion.cost.R, R_true, 1e-4));
    IS_TRUE(math::is_equal(expansion.cost.P, Hux_true.transpose(), 1e-4)); 
    IS_TRUE(math::is_equal(expansion.cost.g_u,  g_u_true, 1e-3)); // high tolerance since numerical error from finite differencing.
    IS_TRUE(math::is_equal(expansion.cost.g_x,  g_x_true, 1e-3)); // high tolerance since numerical error from finite differencing.

    constexpr double deviation = 1e-2;
    const Eigen::VectorXd xt_test = expansion.x.array() + deviation;
    const Eigen::VectorXd ut_test = expansion.u.array() + deviation;

    Eigen::VectorXd xt_diff = xt_test- expansion.x;
    const Eigen::VectorXd ut_diff = ut_test - expansion.u;
    Eigen::VectorXd ct_pred = 0.5*(xt_diff.transpose() * expansion.cost.Q * xt_diff)
            + (xt_diff.transpose()*expansion.cost.P*ut_diff)
            + 0.5*(ut_diff.transpose() * expansion.cost.R * ut_diff)
            + (expansion.cost.g_u.transpose() * ut_diff)
            + (expansion.cost.g_x.transpose() * xt_diff);
    ct_pred.array() += expansion.cost.c;
    IS_EQUAL(ct_pred.size(), 1); 
    IS_ALMOST_EQUAL(ct_pred(0,0), cost_func(xt_test, ut_test), 5e-2);
}


int main()
{
    // Test linear dynamics linearization. 
    test_linear_dynamics_linearization(12, 12);
    test_linear_dynamics_linearization(12, 5);
    test_linear_dynamics_linearization(5, 12);
    test_linear_dynamics_linearization(1, 5);
    test_linear_dynamics_linearization(5, 1);
    test_linear_dynamics_linearization(1, 1);


    // Test nonlinear (cos & sin) dynamics linearization. 
    test_nonlinear_dynamics_linearization(12, 12);
    test_nonlinear_dynamics_linearization(12, 5);
    test_nonlinear_dynamics_linearization(5, 12);
    test_nonlinear_dynamics_linearization(5, 3);
    test_nonlinear_dynamics_linearization(1, 1);
    test_nonlinear_dynamics_linearization(3, 3);


    // Test quadratic cost function 2nd order taylor expansions. 
    test_quadratic_cost_taylor_expansion(12, 12);
    test_quadratic_cost_taylor_expansion(12, 5);
    test_quadratic_cost_taylor_expansion(5, 12);
    test_quadratic_cost_taylor_expansion(1, 5);
    test_quadratic_cost_taylor_expansion(5, 1);
    test_quadratic_cost_taylor_expansion(1, 1);

    // Test exponential cost function 2nd order taylor expansions. 
    // Larger dimensions 3 makes the tolerances higher in the test.
    test_exp_cost_taylor_expansion(5,5);
    test_exp_cost_taylor_expansion(3,5);
    test_exp_cost_taylor_expansion(5,3);
    test_exp_cost_taylor_expansion(3,1);
    test_exp_cost_taylor_expansion(1,3);
    test_exp_cost_taylor_expansion(1,1);

    return 0;
}
