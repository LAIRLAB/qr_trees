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

int main()
{
    Eigen::VectorXd x(2);
    x << 1., 2.;
    auto result = math::jacobian(simple_poly, x);
    WARN("Jacobian:\n" << result)
    return 0;
}
