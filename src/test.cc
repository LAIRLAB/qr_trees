#include <iostream>
#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

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

int main()
{
    Eigen::VectorXd x(2);
    x << 1., 2.;
    //auto result = math::jacobian(simple_poly, x);
    //WARN("Jacobian:\n" << result)
    auto hess = math::hessian(quadratic_cost, x);
    WARN("Hessian:\n" << hess)
    auto grad = math::gradient(quadratic_cost, x);
    WARN("Grad:\n" << grad)
    return 0;
}
