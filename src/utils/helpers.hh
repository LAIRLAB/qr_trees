// 
// Helper functions for tests.
//

#pragma once

#include <ilqr/ilqr_taylor_expansions.hh>

#include <Eigen/Dense>

#include <vector>

ilqr::DynamicsFunc create_linear_dynamics(const Eigen::MatrixXd &A, 
        const Eigen::MatrixXd &B);

// Creates cost function of form 0.5*[(x-x_goal)^T Q (x-x_goal) + u^T R u]
ilqr::CostFunc create_quadratic_cost(const Eigen::MatrixXd &Q, 
        const Eigen::MatrixXd &R, 
        const Eigen::VectorXd &x_goal);


// Creates cost function of form 0.5*[x^T Q x + u^T R u]
ilqr::CostFunc create_quadratic_cost(const Eigen::MatrixXd &Q, 
        const Eigen::MatrixXd &R);


// Computes \sum_i cost(x_i, u_i)
double compute_total_cost(const ilqr::CostFunc &cost,  
                          const std::vector<Eigen::VectorXd> &states, 
                          const std::vector<Eigen::VectorXd> &controls);

// Returns a randomly generated [dim x dim] PSD matrix with specified 
// minimum eigen value.
Eigen::MatrixXd make_random_psd(const int dim, const double min_eig_val);


// Linearly interpolate from x_t0 -> x_T corresponding to time steps t0->T.
std::vector<Eigen::VectorXd> linearly_interpolate(const int t0, 
        const Eigen::VectorXd& x_t0, const int T, const Eigen::VectorXd& x_T);

template <typename T>
inline size_t argmax(std::vector<T> v)
{
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}
