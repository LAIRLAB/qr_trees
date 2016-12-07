// 
// Implements iLQR for linear dynamics and cost.
//

#include <ilqr/iLQR.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace
{

const Eigen::VectorXd& X(const ilqr::iLQR::TaylorExpansion &expansion)
{
    return std::get<0>(expansion);
}
Eigen::VectorXd& X(ilqr::iLQR::TaylorExpansion &expansion)
{
    return std::get<0>(expansion);
}

const Eigen::VectorXd& U(const ilqr::iLQR::TaylorExpansion &expansion)
{
    return std::get<1>(expansion);
}
Eigen::VectorXd& U(ilqr::iLQR::TaylorExpansion &expansion)
{
    return std::get<1>(expansion);
}

void update_dynamics(const ilqr::DynamicsFunc &dynamics_func, ilqr::iLQR::TaylorExpansion &expansion)
{
    std::get<2>(expansion) = ilqr::linearize_dynamics(dynamics_func, X(expansion), U(expansion));
}

void update_cost(const ilqr::CostFunc &cost_func, ilqr::iLQR::TaylorExpansion &expansion)
{
    std::get<3>(expansion) 
        = ilqr::quadraticize_cost(cost_func, X(expansion), U(expansion));
}

// Compute a standard iLQR backup.
void compute_backup(const ilqr::iLQR::TaylorExpansion &expansion, const ilqr::QuadraticValue &Jt1,
        Eigen::MatrixXd &Kt, Eigen::VectorXd &kt, ilqr::QuadraticValue &Jt)
{
    const Eigen::MatrixXd &Vt1 = Jt1.V();
    const Eigen::MatrixXd &Gt1 = Jt1.G();
    const double Wt1 = Jt1.W();
    Eigen::MatrixXd &Vt = Jt.V();
    Eigen::MatrixXd &Gt = Jt.G();
    double &Wt = Jt.W();

    const ilqr::Dynamics &dynamics = std::get<2>(expansion);
    // [dim(x)] x [dim(x)]
    const Eigen::MatrixXd &A = dynamics.A; 
    // [dim(x)] x [dim(u)]
    const Eigen::MatrixXd &B = dynamics.B;

    const ilqr::Cost &cost = std::get<3>(expansion);
    // [dim(x)] x [dim(x)]
    const Eigen::MatrixXd &Q = cost.Q;        
    // [dim(x)] x [dim(u)]
    const Eigen::MatrixXd &P = cost.P;
    // [dim(u)] x [dim(u)]
    const Eigen::MatrixXd &R = cost.R;
    // [dim(u)] x [1]
    const Eigen::VectorXd &g_u = cost.g_u;
    const Eigen::VectorXd &g_x = cost.g_x;
    const double &c= cost.c;

    const int state_dim = A.rows();
    IS_GREATER(state_dim, 0);
    IS_EQUAL(state_dim, X(expansion).size());
    const int control_dim = B.cols();
    IS_GREATER(control_dim, 0);
    IS_EQUAL(control_dim, U(expansion).size());

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

}

namespace ilqr 
{

iLQR::iLQR(const DynamicsFunc &dynamics, 
           const CostFunc &cost, 
           const std::vector<Eigen::VectorXd> &Xs, 
           const std::vector<Eigen::VectorXd> &Us
           )
    : true_dynamics_(dynamics),
      true_cost_(cost)
{
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
        X(expansions_[i]) = Xs[i];
        U(expansions_[i]) = Us[i];
        update_dynamics(true_dynamics_, expansions_[i]);
        update_cost(true_cost_, expansions_[i]);
    }
}


std::vector<Eigen::VectorXd> iLQR::states()
{
    std::vector<Eigen::VectorXd> states; 
    states.reserve(T_);
    std::transform(expansions_.begin(), expansions_.end(), 
            std::back_inserter(states), 
            [](const TaylorExpansion &expansion) { return std::get<0>(expansion); });
    return states;
}

std::vector<Eigen::VectorXd> iLQR::controls()
{
    std::vector<Eigen::VectorXd> controls;
    controls.reserve(T_);
    std::transform(expansions_.begin(), expansions_.end(), 
            std::back_inserter(controls), 
            [](const TaylorExpansion &expansion) { return std::get<1>(expansion); });
    return controls;
}

void iLQR::backwards_pass()
{
    Ks_.clear(); Ks_.resize(T_);
    ks_.clear(); ks_.resize(T_);
    ilqr::QuadraticValue Jt1(state_dim_);
    ilqr::QuadraticValue Jt(state_dim_);
    for (int t = T_-1; t >= 0; t--)
    {
        compute_backup(expansions_[t], Jt1, 
                Ks_[t], ks_[t], Jt);
        Jt1 = Jt;
    }
}

void iLQR::forward_pass(std::vector<double> &costs, 
            std::vector<Eigen::VectorXd> &states,
            std::vector<Eigen::VectorXd> &controls,
            bool update_linearizations)
{
    IS_TRUE(true_dynamics_);
    IS_TRUE(true_cost_);

    costs.clear(); states.clear(); controls.clear();
    costs.reserve(T_); states.reserve(T_); controls.reserve(T_);

    Eigen::VectorXd xt = X(expansions_[0]);
    for (int t = 0; t < T_; ++t)
    {
        TaylorExpansion &expansion = expansions_[t];
        const Eigen::MatrixXd &Kt = Ks_[t];
        const Eigen::MatrixXd &kt = ks_[t];
        Eigen::VectorXd zt = (xt - X(expansion));

        const Eigen::VectorXd vt = Kt * zt + kt;
        const Eigen::VectorXd ut = vt + U(expansion);

        const double cost = true_cost_(xt, ut);

        costs.push_back(cost);
        states.push_back(xt);
        controls.push_back(ut);

        // Roll forward the dynamics.
        const Eigen::VectorXd xt1 = true_dynamics_(xt, ut);
        IS_EQUAL(xt1.rows(), state_dim_);
        xt = xt1;
        IS_EQUAL(xt.rows(), state_dim_);
    }

    if (update_linearizations)
    {
        for (int t = 0; t < T_; ++t)
        {
            TaylorExpansion &expansion = expansions_[t];
            X(expansion) = states[t];
            U(expansion) = controls[t];

            update_dynamics(true_dynamics_, expansion);
            update_cost(true_cost_, expansion);
        }
    }
}

} // namespace lqr
