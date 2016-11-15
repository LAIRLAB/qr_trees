// 
// Implements iLQR for linear dynamics and cost.
//

#include <ilqr/iLQR.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

#include <numeric>

namespace
{
}

namespace ilqr 
{

iLQR::iLQR(const DynamicsFunc &dynamics, 
           const CostFunc &cost, 
           const std::vector<Eigen::VectorXd> &Xs, 
           const std::vector<Eigen::VectorXd> &Us)
    : true_dynamics_(dynamics),
      true_cost_(cost)
{
    // At least two points in trajectory.
    IS_GREATER(Xs.size(), 1);
    IS_EQUAL(Us.size(), Xs.size());
    T_ = Xs.size();

    expansions_.resize(T_);
    for (int i = 0; i < T_; ++i)
    {
        expansions_[i].x = Xs[i];
        expansions_[i].u = Us[i];
        update_dynamics(expansions_[i]);
        update_cost(expansions_[i]);
    }

    state_dim_ = Xs[0].size();
    control_dim_ = Us[0].size();

    // Create local variables for allowing lambda captures.
    const int state_dim = state_dim_;
    const int control_dim = control_dim_;
    dynamics_wrapper_ = [state_dim, control_dim, dynamics](const Eigen::VectorXd &pt) 
    { 
        return dynamics(pt.topRows(state_dim), pt.bottomRows(control_dim));
    };

    cost_wrapper_ = [state_dim, control_dim, cost](const Eigen::VectorXd &pt) 
    { 
        return cost(pt.topRows(state_dim), pt.bottomRows(control_dim));
    };
}


void iLQR::update_dynamics(TaylorExpansion &expansion)
{
    Eigen::VectorXd xu(state_dim_ + control_dim_);
    xu.topRows(state_dim_) = expansion.x;
    xu.bottomRows(control_dim_) = expansion.u;
    const auto J = math::jacobian(dynamics_wrapper_, xu);

    const auto xt1 = dynamics_wrapper_(xu);
    IS_EQUAL(xt1, true_dynamics_(expansion.x, expansion.u));

    Eigen::MatrixXd &A = expansion.dynamics.A;

    A = Eigen::MatrixXd::Zero(state_dim_ + 1, state_dim_ + 1);
    A.topLeftCorner(state_dim_, state_dim_) =  J.leftCols(state_dim_);
    Eigen::VectorXd right_column = Eigen::VectorXd::Ones(state_dim_+1);
    right_column.topRows(state_dim_) = xt1;
    A.rightCols(1) = xt1;

    Eigen::MatrixXd &B = expansion.dynamics.B;
    B = Eigen::MatrixXd::Zero(state_dim_ + 1, control_dim_);
    B.topRows(state_dim_) = J.rightCols(control_dim_);
}

void iLQR::update_cost(TaylorExpansion &expansion)
{
    auto &Q = expansion.cost.Q;
    auto &R = expansion.cost.R;
    auto &P = expansion.cost.P;

    Eigen::VectorXd xu(state_dim_ + control_dim_);
    xu.topRows(state_dim_) = expansion.x;
    xu.bottomRows(control_dim_) = expansion.u;
    const auto H = math::hessian(cost_wrapper_, xu);
    const auto g = math::gradient(cost_wrapper_, xu);
    const auto g_x = g.leftCols(state_dim_);
    const auto g_u = g.rightCols(control_dim_);

    const double c = cost_wrapper_(xu);
    IS_EQUAL(c, true_cost_(expansion.x, expansion.u));
    
    Q = Eigen::MatrixXd::Zero(state_dim_ + 1, state_dim_ + 1);
    Q.topLeftCorner(state_dim_, state_dim_) 
        = H.topLeftCorner(state_dim_, state_dim_);
    Q.topRightCorner(state_dim_, 1) = g_x;
    Q.bottomLeftCorner(1, state_dim_) = g_x;
    Q.bottomRightCorner(1, 1) = Eigen::VectorXd::Constant(1,1,2.0*c);
    IS_EQUAL(Q(state_dim_, state_dim_), 2.0*c);

    // Cross terms.
    P = Eigen::MatrixXd::Zero(state_dim_+1, control_dim_); 
    P.topRows(state_dim_) = H.topLeftCorner(state_dim_, control_dim_);

    // Control terms.
    R = H.bottomRightCorner(control_dim_, control_dim_);
    expansion.cost.b_u = g_u;
}

void iLQR::solve()
{
    backwards_pass();
    std::vector<double> costs = forward_pass();
    const double total_cost = std::accumulate(costs.begin(), costs.end(), 0.0);
    PRINT("Total cost: " << total_cost);
}

void iLQR::backwards_pass()
{
    Ks_.clear(); Ks_.resize(T_);
    // [dim(x) + 1] x [dim(x) + 1]
    Eigen::MatrixXd Vt1 = Eigen::MatrixXd::Zero(state_dim_+1, state_dim_+1);
    // [1] * [dim(x) + 1] 
    Eigen::MatrixXd Gt1 = Eigen::MatrixXd::Zero(1, state_dim_+1);

    for (int t = T_-1; t >= 0; t--)
    {
        TaylorExpansion expansion = expansions_[t];
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

        Eigen::MatrixXd linear_term 
            = Eigen::MatrixXd::Zero(control_dim_, state_dim_+1);
        linear_term.rightCols(1) = g_u + B.transpose()*Gt1.transpose();

        Eigen::MatrixXd Kt = (R + B.transpose()*Vt1*B).inverse()
                                        * (P.transpose() 
                                           + B.transpose()*Vt1*A 
                                           + linear_term);
        Kt *= -1.0;

        IS_EQUAL(Kt.rows(), control_dim_);
        IS_EQUAL(Kt.cols(), state_dim_ + 1);
        Ks_[t] = Kt;

        const Eigen::MatrixXd &Q = expansion.cost.Q;
        const auto tmp = (A + B*Kt);
        Vt1 = Q + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp + 2.0*P*Kt;
        Gt1 = g_u.transpose()*Kt + Gt1*tmp;
    }
}

std::vector<double> iLQR::forward_pass() 
{
    TaylorExpansion expansion = expansions_[0];
    Eigen::VectorXd xt = expansion.x;
    Eigen::VectorXd ut = Eigen::VectorXd::Zero(control_dim_);
    std::vector<double> costs(T_);
    for (int t = 0; t < T_; ++t)
    {
        expansion = expansions_[t];
        const Eigen::MatrixXd &Kt = Ks_[t];
        Eigen::VectorXd zt = Eigen::VectorXd::Ones(state_dim_+1);
        zt.topRows(state_dim_) = (xt - expansion.x);

        Eigen::VectorXd vt = Kt * zt;
        ut = vt + expansion.u;

        const double cost = true_cost_(xt, ut);
        costs[t] = cost;

        // Roll forward the dynamics.
        xt = true_dynamics_(xt, ut);
        IS_EQUAL(xt.rows(), state_dim_);
    }

    return costs;
}

} // namespace lqr
