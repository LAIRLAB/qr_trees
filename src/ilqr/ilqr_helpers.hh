
#pragma once

#include <ilqr/types.hh>

#include <Eigen/Dense>

namespace ilqr
{

// Taylor series expansion points of dynamics and cost functions.
struct TaylorExpansion
{
    Eigen::VectorXd x;
    Eigen::VectorXd u;
    ilqr::Dynamics dynamics;
    ilqr::Cost cost;
};

void compute_backup(const TaylorExpansion &expansion, 
        const Eigen::MatrixXd &Vt1, const Eigen::MatrixXd &Gt1, const double Wt1,
        Eigen::MatrixXd &Kt, Eigen::VectorXd &kt,
        Eigen::MatrixXd &Vt, Eigen::MatrixXd &Gt, double &Wt);

void update_dynamics(const DynamicsFunc &dynamics_func, TaylorExpansion &expansion);

void update_cost(const CostFunc &cost_func, TaylorExpansion &expansion);

} // namespace ilqr
