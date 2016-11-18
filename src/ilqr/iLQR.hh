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

// Update the dynamics inside the expansion point in-place.
void update_dynamics(const DynamicsFunc &dynamics_func, TaylorExpansion &expansion);
void update_cost(const CostFunc &cost_func, TaylorExpansion &expansion);

class iLQR
{
public:

    iLQR(const DynamicsFunc &dynamics, const CostFunc &cost, 
            const std::vector<Eigen::VectorXd> &Xs, 
            const std::vector<Eigen::VectorXd> &Us);

    void backwards_pass(std::vector<Eigen::MatrixXd> &Vs,
                        std::vector<Eigen::MatrixXd> &Gs);

    void forward_pass(std::vector<double> &costs, 
            std::vector<Eigen::VectorXd> &states,
            std::vector<Eigen::VectorXd> &controls);

    std::vector<Eigen::VectorXd> states();
    std::vector<Eigen::VectorXd> controls();

private:
    int state_dim_ = -1;
    int control_dim_  = -1;
    int T_ = -1; // time horizon

    DynamicsFunc true_dynamics_; 
    CostFunc true_cost_; 

    // Taylor series expansion points and expanded dynamics, cost.
    std::vector<TaylorExpansion> expansions_;

    // Feedback control gains.
    std::vector<Eigen::MatrixXd> Ks_;
};

} // namespace lqr

