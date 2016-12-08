// 
// Helper functions for tests.
//

#pragma once

#include <ilqr/ilqr_taylor_expansions.hh>

#include <Eigen/Dense>

ilqr::DynamicsFunc create_linear_dynamics(const Eigen::MatrixXd &A, 
        const Eigen::MatrixXd &B);

// Creates cost function of form 0.5*[(x-x_goal)^T Q (x-x_goal) + u^T R u]
ilqr::CostFunc create_quadratic_cost(const Eigen::MatrixXd &Q, 
        const Eigen::MatrixXd &R, 
        const Eigen::VectorXd &x_goal);


// Creates cost function of form 0.5*[x^T Q x + u^T R u]
ilqr::CostFunc create_quadratic_cost(const Eigen::MatrixXd &Q, 
        const Eigen::MatrixXd &R);

// Returns a randomly generated [dim x dim] PSD matrix with specified 
// minimum eigen value.
Eigen::MatrixXd make_random_psd(const int dim, const double min_eig_val);
