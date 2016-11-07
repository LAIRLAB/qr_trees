#pragma once

#include <Eigen/Dense>

namespace math
{

// Returns true if the matrices are element-wise equal within tolerance.
bool is_equal(const Eigen::MatrixXd &mat1, const Eigen::MatrixXd &mat2, const double tol=1e-12);

// Returns true if a matrix is equal to its transpose. Returns false otherwise. 
// Will likely crash if input is non-square (undefined behavior for non-square).
bool is_symmetric(const Eigen::MatrixXd &mat);

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

// Projects a symmetric matrix onto the PSD (positive semi-definite) cone by bumping up any 
// eigenvalues <= 0 to min_eigval.  If min_eigval is 0 this projects onto the boundary of the PSD cone.
// For numerical stability, a value higher than 0 is recommended.
Eigen::MatrixXd project_to_psd(const Eigen::MatrixXd &symmetric_mat, const double min_eigval = 0);

// Throws exception if the matrix mat is not positive-semi definite with all eigenvalues
// greater than the specified min_eigval.
void check_psd(const Eigen::MatrixXd &square_mat, const double min_eigval = 0);

} // namespace math

