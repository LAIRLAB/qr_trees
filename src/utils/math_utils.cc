#include <utils/math_utils.hh>

#include <utils/debug_utils.hh>

namespace math 
{

Eigen::MatrixXd jacobian(
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> &func, 
        const Eigen::VectorXd &pt, const double delta)
{
    IS_GREATER(delta, 0)

    const int dim = pt.size();
    IS_GREATER(dim, 0)

    Eigen::MatrixXd jacobian(dim, dim);

    for (int i = 0; i < dim; ++i)
    {
        Eigen::VectorXd pt_positive = pt;
        pt_positive[i] += delta;

        Eigen::VectorXd pt_negative = pt;
        pt_negative[i] -= delta;

        jacobian.col(i) = (func(pt_positive) - func(pt_negative)) / (2.0 * delta);
    }

    return jacobian;
}

Eigen::MatrixXd hessian(
        const std::function<double(const Eigen::VectorXd&)> &func, 
        const Eigen::VectorXd &pt, const double delta)
{
}

} // namespace math 
