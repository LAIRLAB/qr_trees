
#include <utils/helpers.hh>

#include <utils/debug_utils.hh>

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

ilqr::CostFunc create_quadratic_cost(const Eigen::MatrixXd &Q, 
        const Eigen::MatrixXd &R, const Eigen::VectorXd &x_goal)
{
    return [&Q, &R, &x_goal](const Eigen::VectorXd &x, const Eigen::VectorXd& u) 
    {
        const int state_dim = x.size();
        const int control_dim = u.size();
        IS_EQUAL(Q.cols(), state_dim);
        IS_EQUAL(Q.rows(), state_dim);
        IS_EQUAL(R.rows(), control_dim);
        IS_EQUAL(R.cols(), control_dim);
        const Eigen::VectorXd x_diff = x - x_goal;
        const Eigen::VectorXd cost = 0.5*(x_diff.transpose()*Q*x_diff 
                + u.transpose()*R*u);
        IS_EQUAL(cost.size(), 1);
        const double c = cost(0);
        return c;
    };
}

ilqr::CostFunc create_quadratic_cost(const Eigen::MatrixXd &Q, 
        const Eigen::MatrixXd &R)
{
    return create_quadratic_cost(Q, R, Eigen::VectorXd::Zero(Q.rows()));
}


double compute_total_cost(const ilqr::CostFunc &cost,  
                          const std::vector<Eigen::VectorXd> &states, 
                          const std::vector<Eigen::VectorXd> &controls)  
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

std::vector<Eigen::VectorXd> linearly_interpolate(const int t0, 
        const Eigen::VectorXd& x_t0, const int T, const Eigen::VectorXd& x_T)
{
    std::vector<Eigen::VectorXd> xstars;
    Eigen::VectorXd xt = x_t0;
    for (int t_target = t0; t_target < T; ++t_target)
    {
        const double progress = static_cast<double>(t_target)/static_cast<double>(T-1);
        const Eigen::VectorXd target_state = progress * (x_T - xt) + xt;
        xstars.push_back(target_state);
    }
    return xstars;
}
