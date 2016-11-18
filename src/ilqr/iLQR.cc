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
iLQR::iLQR(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
        const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, 
        const int T, 
        const std::vector<Eigen::VectorXd> &Xs, 
        const std::vector<Eigen::VectorXd> &Us
        )
    : T_(T), A_(A), B_(B), Q_(Q), R_(R)
{
    has_true_lqr_ = true;

    state_dim_ = A.rows();
    control_dim_ = B.cols();

    true_dynamics_ = [A,B](const Eigen::VectorXd& x, const Eigen::VectorXd& u) { 
        Eigen::VectorXd xt1 = A*x + B*u; 
        return xt1;
    };
    true_cost_ = [Q,R](const Eigen::VectorXd& x, const Eigen::VectorXd& u) { 
        return (x.transpose()*Q*x + u.transpose()*R*u)[0]; 
    };

    // Setup the expansion points.
    expansions_.resize(T_);
    for (int i = 0; i < T_; ++i)
    {
        expansions_[i].x = Xs[i];
        expansions_[i].u = Us[i];
        update_dynamics(true_dynamics_, expansions_[i]);
        update_cost(true_cost_, expansions_[i]);
    }
}

iLQR::iLQR(const DynamicsFunc &dynamics, 
           const CostFunc &cost, 
           const std::vector<Eigen::VectorXd> &Xs, 
           const std::vector<Eigen::VectorXd> &Us
           )
    : true_dynamics_(dynamics),
      true_cost_(cost)
{
    has_true_lqr_ = false;
    // At least two points in trajectory.
    IS_GREATER(Xs.size(), 1);
    IS_EQUAL(Us.size(), Xs.size());
    T_ = Xs.size();

    state_dim_ = Xs[0].size();
    control_dim_ = Us[0].size();

    // Setup the expansion points.
    expansions_.resize(T_);
    for (int i = 0; i < T_; ++i)
    {
        expansions_[i].x = Xs[i];
        expansions_[i].u = Us[i];
        update_dynamics(true_dynamics_, expansions_[i]);
        update_cost(true_cost_, expansions_[i]);
    }
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

void iLQR::solve()
{
    backwards_pass();
    std::vector<double> costs = forward_pass();
    const double total_cost = std::accumulate(costs.begin(), costs.end(), 0.0);
    PRINT("Total cost: " << total_cost);
}

std::vector<Eigen::VectorXd> iLQR::states()
{
    std::vector<Eigen::VectorXd> states; 
    states.reserve(T_);
    std::transform(expansions_.begin(), expansions_.end(), 
            std::back_inserter(states), 
            [](const TaylorExpansion &expansion) { return expansion.x; });
    return states;
}

std::vector<Eigen::VectorXd> iLQR::controls()
{
    std::vector<Eigen::VectorXd> controls;
    controls.reserve(T_);
    std::transform(expansions_.begin(), expansions_.end(), 
            std::back_inserter(controls), 
            [](const TaylorExpansion &expansion) { return expansion.u; });
    return controls;
}

void iLQR::backwards_pass()
{
    Ks_.clear(); Ks_.resize(T_);
    // [dim(x) + 1] x [dim(x) + 1]
    Eigen::MatrixXd Vt1 = Eigen::MatrixXd::Zero(state_dim_+1, state_dim_+1);
    //Eigen::MatrixXd Vt1 = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
    // [1] * [dim(x) + 1] 
    Eigen::MatrixXd Gt1 = Eigen::MatrixXd::Zero(1, state_dim_+1);
    //Eigen::MatrixXd Gt1 = Eigen::MatrixXd::Zero(1, state_dim_);

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
        //Eigen::MatrixXd Kt = (R + B.transpose()*Vt1*B).inverse()
        //                                * (B.transpose()*Vt1*A);
        //Kt *= -1.0;

        IS_EQUAL(Kt.rows(), control_dim_);
        //IS_EQUAL(Kt.cols(), state_dim_ + 1);
        Ks_[t] = Kt;

        const Eigen::MatrixXd &Q = expansion.cost.Q;        
        const Eigen::MatrixXd tmp = (A + B*Kt);
        Eigen::MatrixXd Vt1next = Q + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp + 2.0*P*Kt;
        //Eigen::MatrixXd Vt1next = Q + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp;
        Vt1 = Vt1next;
        //Gt1 = g_u.transpose()*Kt + Gt1*tmp;
    }
}

std::vector<double> iLQR::forward_pass() 
{
    IS_TRUE(true_dynamics_);
    IS_TRUE(true_cost_);

    Eigen::VectorXd xt = expansions_[0].x;
    Eigen::VectorXd ut = Eigen::VectorXd::Zero(control_dim_);
    std::vector<double> costs(T_);
    for (int t = 0; t < T_; ++t)
    {
        TaylorExpansion &expansion = expansions_[t];
        const Eigen::MatrixXd &Kt = Ks_[t];
        Eigen::VectorXd zt = Eigen::VectorXd::Ones(state_dim_+1);
        zt.topRows(state_dim_) = (xt - expansion.x);
        if (has_true_lqr_) 
        {
            zt.topRows(state_dim_) = xt;
        }

        Eigen::VectorXd vt = Kt * zt;
        ut = vt + expansion.u;
        if (has_true_lqr_)
        {
            ut = vt;
        }

        const double cost = true_cost_(xt, ut);
        costs[t] = cost;

        expansion.x = xt;
        expansion.u = ut;

        // Roll forward the dynamics.
        const Eigen::VectorXd xt1 = true_dynamics_(xt, ut);
        IS_EQUAL(xt1.rows(), state_dim_);
        xt = xt1;
        IS_EQUAL(xt.rows(), state_dim_);
    }

    for (auto &expansion : expansions_)
    {
        update_dynamics(true_dynamics_, expansion);
        update_cost(true_cost_, expansion);
    }

    return costs;
}

} // namespace lqr
