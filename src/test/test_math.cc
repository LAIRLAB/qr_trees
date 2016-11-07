//
// Test functions for math utilities.
//

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

#include <cmath>

namespace
{
constexpr double TOL = 1e-7;

bool is_orthonormal(const Eigen::MatrixXd &mat)
{
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(mat.rows(), mat.cols());
    return math::is_equal(mat.transpose()*mat, identity, TOL) 
        && math::is_equal(mat*mat.transpose(), identity, TOL);
}


Eigen::VectorXd simple_poly(const Eigen::VectorXd &x)
{
    IS_EQUAL(x.size(), 2);
    Eigen::VectorXd y(2);
    y(0) = x(0)*x(0);
    y(1) = x(0) + x(1);
    return y;
}

double quadratic_cost(const Eigen::VectorXd &x)
{
    IS_EQUAL(x.size(), 2);
    Eigen::MatrixXd Q(2,2);
    Q << 5, 6, 6, 8;
    double c = x.transpose() * Q * x;
    return c;
}

} // namespace

void test_jacobian()
{
    Eigen::VectorXd x(2);
    x << 1., 2.;
    auto result = math::jacobian(simple_poly, x);
    WARN("Jacobian:\n" << result)
}

void test_hessian()
{
    Eigen::VectorXd x(2);
    x << 1., 2.;
    auto hess = math::hessian(quadratic_cost, x);
    WARN("Hessian:\n" << hess)
}

void test_gradient()
{
    Eigen::VectorXd x(2);
    x << 1., 2.;
    auto grad = math::gradient(quadratic_cost, x);
    WARN("Grad:\n" << grad)
}

void test_check_psd()
{
    // Input eigen vectors and eigenvalues.
    Eigen::MatrixXd P(2,2);
    Eigen::VectorXd d(2);
    // Generate a rotation matrix since it is orthonormal.
    P << cos(0.1), -sin(0.1), sin(0.1), cos(0.1);
    d << 5, 1;
    IS_TRUE(is_orthonormal(P));
    Eigen::MatrixXd input = P*d.asDiagonal()*P.transpose();
    DOES_NOT_THROW(math::check_psd(input));

    d << 5, -1;
    input = P*d.asDiagonal()*P.transpose();
    DOES_THROW(math::check_psd(input));

    // If we give a lower threshold (it should pass, even though it is unreasonable).
    double min_eigval = -1;
    DOES_NOT_THROW(math::check_psd(input, min_eigval));

    // Fail and pass checks with a more reasonable threshold.
    min_eigval = 0.5;
    DOES_THROW(math::check_psd(input, min_eigval));
    d << 5, 1.0;
    input = P*d.asDiagonal()*P.transpose();
    DOES_NOT_THROW(math::check_psd(input, min_eigval));
    DOES_NOT_THROW(math::check_psd(input));

    // Make the matrix non-symmetric and we should not pass.
    input(0,1) = input(1,0) + 0.05;
    DOES_THROW(math::check_psd(input));

    // Non-square matrices should also throw.
    input = Eigen::MatrixXd(3,2);
    DOES_THROW(math::check_psd(input));
}

void test_project_psd()
{
    Eigen::MatrixXd P(2,2);
    Eigen::VectorXd d(2);
    // Generate a rotation matrix since it is orthonormal.
    P << cos(0.1), -sin(0.1), sin(0.1), cos(0.1);
    d << 5, 1;
    IS_TRUE(is_orthonormal(P));
    // Confirm input is PSD.
    Eigen::MatrixXd input = P*d.asDiagonal()*P.transpose();
    DOES_NOT_THROW(math::check_psd(input));

    // Output should be same as input if the input is already PSD.
    Eigen::MatrixXd output = math::project_to_psd(input);
    IS_TRUE(math::is_equal(input, output));

    // Make an eigenvalue negative.
    Eigen::VectorXd d2(2); // another set of eigenvalues.
    d2 << 5, -1;
    input = P*d2.asDiagonal()*P.transpose();
    output = math::project_to_psd(input);
    d << 5, 0;
    Eigen::MatrixXd answer = P*d.asDiagonal()*P.transpose();
    IS_TRUE(math::is_equal(output, answer));

    // If we want to make it PD with min eigenvalue with 0.5.
    output = math::project_to_psd(input, 0.5);
    d << 5, 0.5;
    answer = P*d.asDiagonal()*P.transpose();
    IS_TRUE(math::is_equal(output, answer));
}


int main()
{
    test_check_psd();

    test_project_psd();

    // These currently just print stuff but do not actually check.
    //test_gradient();
    //test_jacobian();
    //test_hessian();


    return 0;
}
