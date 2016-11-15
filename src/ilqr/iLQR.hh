// 
// Implements iLQR (on a traditional chain) for nonlinear dynamics and cost.
//

#pragma once

#include <ilqr/types.hh>

#include <Eigen/Dense>

#include <vector>

namespace ilqr
{

struct StateCost
{
    StateCost(Eigen::VectorXd state, Eigen::VectorXd control, double cost) : x(state), u(control), c(cost) {}
    Eigen::VectorXd x;
    Eigen::VectorXd u;
    double c;
};

// Taylor series expansion points of dynamics and cost functions.
struct TaylorExpansion
{
    Eigen::VectorXd x;
    Eigen::VectorXd u;
    ilqr::Dynamics dynamics;
    ilqr::Cost cost;
};

class iLQR
{
public:

    iLQR(const DynamicsFunc &dynamics, const CostFunc &cost, 
            const std::vector<Eigen::VectorXd> &Xs, 
            const std::vector<Eigen::VectorXd> &Us);

    // Update the dynamics inside the expansion point in-place.
    void update_dynamics(TaylorExpansion &expansion);
    void update_cost(TaylorExpansion &expansion);

    void solve();

    void backwards_pass();

    std::vector<double> forward_pass();

private:
    int state_dim_ = -1;
    int control_dim_  = -1;
    int T_ = -1; // time horizon

    const DynamicsFunc true_dynamics_; 
    const CostFunc true_cost_; 
    using DynamicsWrapper = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
    using CostWrapper = std::function<double(const Eigen::VectorXd&)>;
    DynamicsWrapper dynamics_wrapper_;
    CostWrapper cost_wrapper_;

    // Taylor series expansion points and expanded dynamics, cost.
    std::vector<TaylorExpansion> expansions_;

    // Feedback control gains.
    std::vector<Eigen::MatrixXd> Ks_;
    std::vector<Eigen::MatrixXd> ks_;
};

} // namespace lqr

