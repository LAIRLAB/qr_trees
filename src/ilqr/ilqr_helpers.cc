
#include <ilqr/ilqr_helpers.hh>

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

void compute_backup(const TaylorExpansion &expansion, const QuadraticValue &Jt1,
        Eigen::MatrixXd &Kt, Eigen::VectorXd &kt, QuadraticValue &Jt)
{
    const Eigen::MatrixXd &Vt1 = Jt1.V();
    const Eigen::MatrixXd &Gt1 = Jt1.G();
    const double Wt1 = Jt1.W();
    Eigen::MatrixXd &Vt = Jt.V();
    Eigen::MatrixXd &Gt = Jt.G();
    double &Wt = Jt.W();

    // [dim(x)] x [dim(x)]
    const Eigen::MatrixXd &A = expansion.dynamics.A; 
    // [dim(x)] x [dim(u)]
    const Eigen::MatrixXd &B = expansion.dynamics.B;

    // [dim(x)] x [dim(x)]
    const Eigen::MatrixXd &Q = expansion.cost.Q;        
    // [dim(x)] x [dim(u)]
    const Eigen::MatrixXd &P = expansion.cost.P;
    // [dim(u)] x [dim(u)]
    const Eigen::MatrixXd &R = expansion.cost.R;
    // [dim(u)] x [1]
    const Eigen::VectorXd &g_u = expansion.cost.g_u;
    const Eigen::VectorXd &g_x = expansion.cost.g_x;
    const double &c= expansion.cost.c;

    const int state_dim = A.rows();
    IS_GREATER(state_dim, 0);
    IS_EQUAL(state_dim, expansion.x.size());
    const int control_dim = B.cols();
    IS_GREATER(control_dim, 0);
    IS_EQUAL(control_dim, expansion.u.size());

    const Eigen::MatrixXd inv_term = -1.0*(R + B.transpose()*Vt1*B).inverse();
    Kt = inv_term * (P.transpose() + B.transpose()*Vt1*A); 
    kt = inv_term * (g_u + B.transpose()*Gt1.transpose());

    IS_EQUAL(Kt.rows(), control_dim);
    IS_EQUAL(Kt.cols(), state_dim);
    IS_EQUAL(kt.size(), control_dim);

    const Eigen::MatrixXd tmp = (A + B*Kt);
    Vt = Q + 2.0*(P*Kt) + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp;

    Gt = kt.transpose()*P.transpose() + kt.transpose()*R*Kt + g_x.transpose() 
        + g_u.transpose()*Kt + kt.transpose()*B.transpose()*Vt1*tmp + Gt1*tmp;

    const Eigen::VectorXd Wt_mat = 0.5*(kt.transpose()*R*kt) 
        + g_u.transpose()*kt + Gt1*B*kt 
        + 0.5*(kt.transpose()*B.transpose()*Vt1*B*kt);
    Wt = Wt_mat(0) + c + Wt1;
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

    Eigen::VectorXd xu(state_dim + control_dim);
    xu.topRows(state_dim) = expansion.x;
    xu.bottomRows(control_dim) = expansion.u;
    const auto J = math::jacobian(dynamics_wrapper, xu);

    Eigen::MatrixXd &A = expansion.dynamics.A;
    Eigen::MatrixXd &B = expansion.dynamics.B;
    
    A =  J.leftCols(state_dim);
    B = J.rightCols(control_dim);

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
    Eigen::VectorXd &g_x = expansion.cost.g_x;
    Eigen::VectorXd &g_u = expansion.cost.g_u;
    double &c= expansion.cost.c;
    
    Eigen::VectorXd xu(state_dim + control_dim);
    xu.topRows(state_dim) = expansion.x;
    xu.bottomRows(control_dim) = expansion.u;

    // Zero out components that are less than this threshold. We do this since
    // finite differencing has numerical issues.
    constexpr double ZERO_THRESH = 1e-7;
    auto H = math::hessian(cost_wrapper, xu);
    H = H.array() * (H.array().abs() > ZERO_THRESH).cast<double>();
    Q = H.topLeftCorner(state_dim, state_dim);
    P = H.topRightCorner(state_dim, control_dim);
    R = H.bottomRightCorner(control_dim, control_dim);

    auto g = math::gradient(cost_wrapper, xu);
    g = g.array() * (g.array().abs() > ZERO_THRESH).cast<double>();
    IS_EQUAL(g.size(), state_dim + control_dim);
    g_x = g.topRows(state_dim);
    g_u = g.bottomRows(control_dim);

    c = cost_wrapper(xu);
    
    Q = math::project_to_psd(Q, 1e-11);
    math::check_psd(Q, 1e-12);

    // Control terms.
    R = math::project_to_psd(R, 1e-8);
    math::check_psd(R, 1e-9);
}

} // namespace ilqr
