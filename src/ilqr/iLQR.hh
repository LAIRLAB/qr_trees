// 
// Implements iLQR (on a traditional chain) for nonlinear dynamics and cost.
//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// December 2016
//

#pragma once

#include <ilqr/ilqr_taylor_expansions.hh>

#include <Eigen/Dense>

#include <tuple>
#include <vector>

namespace ilqr
{

class iLQR
{
public:
    // Stores linearization points x, u and the Taylor expansions of the dynamics and cost.
    using TaylorExpansion = std::tuple<Eigen::VectorXd, Eigen::VectorXd, ilqr::Dynamics, ilqr::Cost>;

    iLQR(const DynamicsFunc &dynamics, const CostFunc &cost, 
            const std::vector<Eigen::VectorXd> &Xs, 
            const std::vector<Eigen::VectorXd> &Us);

    void backwards_pass();

    void forward_pass(std::vector<double> &costs, 
            std::vector<Eigen::VectorXd> &states,
            std::vector<Eigen::VectorXd> &controls, 
            bool update_linearizations
            );

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
    std::vector<Eigen::VectorXd> ks_;
};

} // namespace lqr

