#include <utils/math_utils.hh>

#include <utils/debug_utils.hh>

namespace math 
{

Eigen::VectorXd gradient(
        const std::function<double(const Eigen::VectorXd&)> &func, 
        const Eigen::VectorXd &pt, const double delta)
{
    IS_GREATER(delta, 0);

    const int dim = pt.size();
    IS_GREATER(dim, 0);

    Eigen::VectorXd gradient(dim);

    for (int i = 0; i < dim; ++i)
    {
        Eigen::VectorXd pt_positive = pt;
        pt_positive[i] += delta;

        Eigen::VectorXd pt_negative = pt;
        pt_negative[i] -= delta;

        gradient(i) = (func(pt_positive) - func(pt_negative)) / (2.0 * delta);
    }

    return gradient;
}
Eigen::MatrixXd jacobian(
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> &func, 
        const Eigen::VectorXd &pt, const double delta)
{
    IS_GREATER(delta, 0);

    const int dim = pt.size();
    IS_GREATER(dim, 0);

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
    IS_GREATER(delta, 0);
    const int dim = pt.size();
    IS_GREATER(dim, 0);

    // Precompute constants.
    const double two_delta = 2.0*delta;
    const double delta_sq = delta*delta;
    const double ij_eq_div = 12.*delta_sq;
    const double ij_neq_div = 4.*delta_sq;
    
    Eigen::MatrixXd hessian(dim, dim);
    // Central difference approximation based on 
    // https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
    for (int i = 0; i < dim; ++i)
    {
        for (int j = i; j < dim; ++j)
        {
            Eigen::VectorXd p1 = pt;
            Eigen::VectorXd p2 = pt;
            Eigen::VectorXd p3 = pt;
            Eigen::VectorXd p4 = pt;
            double value = 0;
            if (i == j)
            {
                p1[i] += two_delta; 
                p2[i] += delta;
                p3[i] -= delta;
                p4[i] -= two_delta;
                value = (-func(p1) + 16.*(func(p2) + func(p3)) - 30.*func(pt) - func(p4)) 
                    / (ij_eq_div);
            } 
            else
            {
                p1[i] += delta;
                p1[j] += delta;

                p2[i] += delta;
                p2[j] -= delta;

                p3[i] -= delta;
                p3[j] += delta;

                p4[i] -= delta;
                p4[j] -= delta;

                value = (func(p1) - func(p2) - func(p3) + func(p4)) / (ij_neq_div);
            }

            hessian(i,j) = value;
            hessian(j,i) = value;
        }
    }

    return hessian;
}

} // namespace math 
