
#include <ilqr/ilqr_taylor_expansions.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace ilqr
{

QuadraticValue::QuadraticValue(const int state_dim) :
    V_(Eigen::MatrixXd::Zero(state_dim, state_dim)), 
    G_(Eigen::MatrixXd::Zero(1, state_dim)),
    W_(0)
{
}

ilqr::Dynamics linearize_dynamics(const DynamicsFunc &dynamics_func, 
                        const Eigen::VectorXd &x, 
                        const Eigen::VectorXd &u
                       )
{

    const int state_dim = x.size();
    const int control_dim = u.size();
    IS_GREATER(state_dim, 0);
    IS_GREATER(control_dim, 0);

    auto dynamics_wrapper = [state_dim, control_dim, &dynamics_func](const Eigen::VectorXd &pt) -> Eigen::VectorXd
    { 
        return Eigen::VectorXd(dynamics_func(pt.topRows(state_dim), pt.bottomRows(control_dim)));
    };

    Eigen::VectorXd xu(state_dim + control_dim);
    xu.topRows(state_dim) = x;
    xu.bottomRows(control_dim) = u;
    const auto J = math::jacobian(dynamics_wrapper, xu);

    ilqr::Dynamics dynamics;
    Eigen::MatrixXd &A = dynamics.A;
    Eigen::MatrixXd &B = dynamics.B;
    
    A =  J.leftCols(state_dim);
    B = J.rightCols(control_dim);

    return dynamics;

}

ilqr::Cost quadraticize_cost(const CostFunc &cost_func, 
                        const Eigen::VectorXd &x, 
                        const Eigen::VectorXd &u)
{
    const int state_dim = x.size();
    const int control_dim = u.size();
    IS_GREATER(state_dim, 0);
    IS_GREATER(control_dim, 0);
    auto cost_wrapper = [state_dim, control_dim, &cost_func](const Eigen::VectorXd &pt) -> double
    { 
        return cost_func(pt.topRows(state_dim), pt.bottomRows(control_dim));
    };

    ilqr::Cost cost;
    Eigen::MatrixXd &Q = cost.Q;
    Eigen::MatrixXd &R = cost.R;
    Eigen::MatrixXd &P = cost.P;
    Eigen::VectorXd &g_x = cost.g_x;
    Eigen::VectorXd &g_u = cost.g_u;
    double &c= cost.c;
    
    Eigen::VectorXd xu(state_dim + control_dim);
    xu.topRows(state_dim) = x;
    xu.bottomRows(control_dim) = u;

    constexpr double ZERO_THRESH = 1e-7;

    Eigen::VectorXd g = math::gradient(cost_wrapper, xu);
    g = g.array() * (g.array().abs() > ZERO_THRESH).cast<double>();
    IS_EQUAL(g.size(), state_dim + control_dim);
    g_x = g.topRows(state_dim);
    g_u = g.bottomRows(control_dim);


    // Zero out components that are less than this threshold. We do this since
    // finite differencing has numerical issues.
    Eigen::MatrixXd H = math::hessian(cost_wrapper, xu);
    //Eigen::MatrixXd H = g * g.transpose();
    H = H.array() * (H.array().abs() > ZERO_THRESH).cast<double>();
    Q = H.topLeftCorner(state_dim, state_dim);
    P = H.topRightCorner(state_dim, control_dim);
    R = H.bottomRightCorner(control_dim, control_dim);

    c = cost_wrapper(xu);
    
    Q = (Q + Q.transpose())/2.0;
    try
    {
        IS_TRUE(math::is_symmetric(Q));
    }
    catch (...)
    {
        WARN("H\n" << g);

    }
    Q = math::project_to_psd(Q, 1e-11);
    math::check_psd(Q, 1e-12);

    // Control terms.
    R = math::project_to_psd(R, 1e-8);
    //math::check_psd(R, 1e-9);

    return cost;
}

} // namespace ilqr
