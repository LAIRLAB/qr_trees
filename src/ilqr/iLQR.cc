// 
// Implements iLQR for linear dynamics and cost.
//

#include <ilqr/iLQR.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

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
        expansions_[i].x = Xs[i];
        expansions_[i].u = Us[i];
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

void iLQR::backwards_pass(std::vector<Eigen::MatrixXd> &Vs, std::vector<Eigen::MatrixXd> &Gs)
{
    Ks_.clear(); Ks_.resize(T_);
    Vs.clear(); Vs.resize(T_);
    Gs.clear(); Gs.resize(T_);

    // [dim(x) + 1] x [dim(x) + 1]
    Eigen::MatrixXd Vt1 = Eigen::MatrixXd::Zero(state_dim_+1, state_dim_+1);
    // [1] * [dim(x) + 1] 
    Eigen::MatrixXd Gt1 = Eigen::MatrixXd::Zero(1, state_dim_+1);

    for (int t = T_-1; t >= 0; t--)
    {
        ilqr::compute_backup(expansions_[t], Vt1, Gt1, Ks_[t], Vs[t], Gs[t]);
        Gt1 = Gs[t];
        Vt1 = Vs[t];
    }
}

void iLQR::forward_pass(std::vector<double> &costs, 
            std::vector<Eigen::VectorXd> &states,
            std::vector<Eigen::VectorXd> &controls)
{
    IS_TRUE(true_dynamics_);
    IS_TRUE(true_cost_);

    Eigen::VectorXd xt = expansions_[0].x;
    Eigen::VectorXd ut = Eigen::VectorXd::Zero(control_dim_);
    costs.reserve(T_);
    for (int t = 0; t < T_; ++t)
    {
        TaylorExpansion &expansion = expansions_[t];
        const Eigen::MatrixXd &Kt = Ks_[t];
        Eigen::VectorXd zt = Eigen::VectorXd::Ones(state_dim_+1);
        zt.topRows(state_dim_) = (xt - expansion.x);

        Eigen::VectorXd vt = Kt * zt;
        ut = vt + expansion.u;

        const double cost = true_cost_(xt, ut);

        expansion.x = xt;
        expansion.u = ut;

        costs.push_back(cost);
        states.push_back(xt);
        controls.push_back(ut);

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
}

} // namespace lqr
