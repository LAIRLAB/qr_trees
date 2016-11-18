
#include <ilqr/ilqr_helpers.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace ilqr
{

void compute_backup(const TaylorExpansion &expansion, 
        const Eigen::MatrixXd &Vt1,
        const Eigen::MatrixXd &Gt1,
        Eigen::MatrixXd &Kt,
        Eigen::MatrixXd &Vt,
        Eigen::MatrixXd &Gt
        )
{
    // [dim(x) + 1] x [dim(x) + 1]
    const Eigen::MatrixXd &A = expansion.dynamics.A; 
    // [dim(x) + 1] x [dim(u)]
    const Eigen::MatrixXd &B = expansion.dynamics.B;

    // [dim(x) + 1] x [dim(u)]
    const Eigen::MatrixXd &P = expansion.cost.P;
    // [dim(u)] x [dim(u)]
    const Eigen::MatrixXd &R = expansion.cost.R;
    // [dim(u)] x [1]
    const Eigen::MatrixXd &g_u = expansion.cost.b_u;

    const int state_dim = A.rows()-1;
    IS_GREATER(state_dim, 0);
    IS_EQUAL(state_dim, expansion.x.size());
    const int control_dim = B.cols();
    IS_GREATER(control_dim, 0);
    IS_EQUAL(control_dim, expansion.u.size());

    Eigen::MatrixXd linear_term 
        = Eigen::MatrixXd::Zero(control_dim, state_dim+1);
    linear_term.rightCols(1) = g_u + B.transpose()*Gt1.transpose();

    Kt = (R + B.transpose()*Vt1*B).inverse()
                                    * (P.transpose() 
                                       + B.transpose()*Vt1*A 
                                       + linear_term);
    Kt *= -1.0;
    IS_EQUAL(Kt.rows(), control_dim);
    IS_EQUAL(Kt.cols(), state_dim + 1);

    const Eigen::MatrixXd &Q = expansion.cost.Q;        
    const Eigen::MatrixXd tmp = (A + B*Kt);
    Vt = Q + Kt.transpose()*R*Kt 
        + tmp.transpose()*Vt1*tmp + 2.0*P*Kt;

    Gt = g_u.transpose()*Kt + Gt1*tmp; 
}

void update_dynamics(const DynamicsFunc &dynamics_func, TaylorExpansion &expansion)
{
    const int state_dim = expansion.x.size();
    const int control_dim = expansion.u.size();
    IS_GREATER(state_dim, 0);
    IS_GREATER(control_dim, 0);

    auto dynamics_wrapper = [state_dim, control_dim, &dynamics_func](const Eigen::VectorXd &pt) -> Eigen::VectorXd
    { 
        return Eigen::VectorXd(dynamics_func(pt.topRows(state_dim), pt.bottomRows(control_dim)));
    };

    Eigen::MatrixXd &A = expansion.dynamics.A;
    A = Eigen::MatrixXd::Zero(state_dim + 1, state_dim + 1);

    Eigen::MatrixXd &B = expansion.dynamics.B;
    B = Eigen::MatrixXd::Zero(state_dim + 1, control_dim);
    
    Eigen::VectorXd xu(state_dim + control_dim);
    xu.topRows(state_dim) = expansion.x;
    xu.bottomRows(control_dim) = expansion.u;
    const auto J = math::jacobian(dynamics_wrapper, xu);

    const Eigen::VectorXd xt1 = dynamics_wrapper(xu);
    IS_EQUAL(xt1, dynamics_func(expansion.x, expansion.u));

    A.topLeftCorner(state_dim, state_dim) =  J.leftCols(state_dim);
    A.bottomRightCorner(1,1) = Eigen::MatrixXd::Constant(1,1,1.0);
    IS_EQUAL(A.bottomRightCorner(1,1), Eigen::MatrixXd::Constant(1,1,1.0))

    B.topRows(state_dim) = J.rightCols(control_dim);
    IS_EQUAL(B.bottomRows(1), Eigen::VectorXd::Zero(control_dim).transpose());

}

void update_cost(const CostFunc &cost_func, TaylorExpansion &expansion)
{
    const int state_dim = expansion.x.size();
    const int control_dim = expansion.u.size();
    IS_GREATER(state_dim, 0);
    IS_GREATER(control_dim, 0);
    auto cost_wrapper = [state_dim, control_dim, &cost_func](const Eigen::VectorXd &pt) -> double
    { 
        return cost_func(pt.topRows(state_dim), pt.bottomRows(control_dim));
    };

    Eigen::MatrixXd &Q = expansion.cost.Q;
    Eigen::MatrixXd &R = expansion.cost.R;
    Eigen::MatrixXd &P = expansion.cost.P;
    
    Q.resize(state_dim + 1, state_dim + 1);
    Q.setZero();
    P.resize(state_dim+1, control_dim); 
    P.setZero(); 

    Eigen::VectorXd xu(state_dim + control_dim);
    xu.topRows(state_dim) = expansion.x;
    xu.bottomRows(control_dim) = expansion.u;

    // Zero out components that are less than this threshold. We do this since
    // finite differencing has numerical issues.
    constexpr double ZERO_THRESH = 1e-7;
    auto H = math::hessian(cost_wrapper, xu);
    H = H.array() * (H.array().abs() > ZERO_THRESH).cast<double>();
    auto g = math::gradient(cost_wrapper, xu);
    g = g.array() * (g.array().abs() > ZERO_THRESH).cast<double>();
    IS_EQUAL(g.size(), state_dim + control_dim);
    const auto g_x = g.topRows(state_dim);
    const auto g_u = g.bottomRows(control_dim);

    const double c = cost_wrapper(xu);
    IS_EQUAL(c, cost_func(expansion.x, expansion.u));
    
    Q.topLeftCorner(state_dim, state_dim) 
        = H.topLeftCorner(state_dim, state_dim);
    Q.topRightCorner(state_dim, 1) = g_x;
    Q.bottomLeftCorner(1, state_dim) = g_x.transpose();
    Q.bottomRightCorner(1, 1) = Eigen::VectorXd::Constant(1,1,2.0*c);
    IS_EQUAL(Q(state_dim, state_dim), 2.0*c);

    Q = math::project_to_psd(Q, 1e-11);
    math::check_psd(Q, 1e-12);

    // Cross terms.
    P = Eigen::MatrixXd::Zero(state_dim+1, control_dim); 
    P.topRows(state_dim) = H.topRightCorner(state_dim, control_dim);

    // Control terms.
    R = H.bottomRightCorner(control_dim, control_dim);
    R = math::project_to_psd(R, 1e-8);
    math::check_psd(R, 1e-9);
    expansion.cost.b_u = g_u;
}

} // namespace ilqr
