#include <utils/math_utils.hh>

#include <utils/debug_utils.hh>

#include <Eigen/Eigenvalues>

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

Eigen::MatrixXd project_to_psd(const Eigen::MatrixXd &symmetric_mat, const double min_eigval)
{
    // Cheap check to confirm matrix is square.
    IS_EQUAL(symmetric_mat.rows(), symmetric_mat.cols());

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetric_mat);
    Eigen::MatrixXd evals = es.eigenvalues().asDiagonal();
    const Eigen::MatrixXd evecs = es.eigenvectors();

    // Indices of eigen values that are less than the minimum eigenvalue threshold.
    int num_failed_evals = 0;
    for (int i = 0; i < evals.size(); ++i)
    {
        if (evals(i,i) < min_eigval)
        {
            ++num_failed_evals;
            evals(i,i) = min_eigval;
        }
    }
    // If we didn't have eigenvalues less than the minimum threshold, then return the original
    // matrix.
    if (num_failed_evals == 0)
    {
        return symmetric_mat;
    }

    // We can reconstruct with P\inv D P = P^T D P if P is orthonormal.
    const Eigen::MatrixXd projection = evecs.transpose() * evals * evecs;
    return projection;
}

void check_psd(const Eigen::MatrixXd &symmetric_mat, const double min_eigval)
{
    // Cheap check to confirm matrix is square.
    IS_EQUAL(symmetric_mat.rows(), symmetric_mat.cols());

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetric_mat, Eigen::EigenvaluesOnly);
    const auto evals = es.eigenvalues();

    for (int i = 0; i < evals.size(); ++i)
    {
        // Throw an exception if this is not true.
        IS_GREATER_EQUAL(evals(i), min_eigval);
    }
}

} // namespace math 
