#pragma once

#include <Eigen/Dense>

namespace math
{

// Computes the Gradient using finite differencing.
Eigen::VectorXd gradient(
        const std::function<double(const Eigen::VectorXd&)> &func, 
        const Eigen::VectorXd &pt, 
        const double delta = 1e-3);

// Computes the Jacobian using finite differencing.
Eigen::MatrixXd jacobian(
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> &func, 
        const Eigen::VectorXd &pt, 
        const double delta = 1e-3);

// Computes the Hessian using finite differencing.
Eigen::MatrixXd hessian(
        const std::function<double(const Eigen::VectorXd&)> &func, 
        const Eigen::VectorXd &pt, 
        const double delta = 1e-3);

} // namespace math

