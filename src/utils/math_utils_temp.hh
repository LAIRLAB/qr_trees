#include <utils/debug_utils.hh>

#include <Eigen/Eigenvalues>

namespace math 
{

constexpr double DEFAULT_STEP_SIZE = 0.0009765625;

template<int _rows, int _cols>
using Matrix = Eigen::Matrix<double, _rows, _cols>;

template<int _rows>
using Vector = Eigen::Matrix<double, _rows, 1>;

template<int _rows, int _cols>
bool is_equal(const Matrix<_rows, _cols> &mat1, const Matrix<_rows, _cols> &mat2, const double tol = 1e-8)
{
    return ((mat1 - mat2).array().abs() < tol).all();
}

template<int _rows>
bool is_symmetric(const Matrix<_rows, _rows> &mat, const double tol = 1e-8)
{
    return math::is_equal<_rows,_rows>(mat, mat.transpose(), tol);
}

template<int _rows, typename Callable>
Vector<_rows> gradient(
        const Callable &func, 
        const Vector<_rows> &pt, const double delta = DEFAULT_STEP_SIZE)
{
    IS_GREATER(delta, 0);

    const int dim = pt.size();
    IS_GREATER(dim, 0);

    Vector<_rows> gradient(dim);

    for (int i = 0; i < dim; ++i)
    {
        Vector<_rows> pt_positive = pt;
        pt_positive[i] += delta;

        Vector<_rows> pt_negative = pt;
        pt_negative[i] -= delta;

        gradient(i) = (func(pt_positive) - func(pt_negative)) / (2.0 * delta);
    }

    return gradient;
}

template<int _in_rows, int _out_rows, typename Callable>
Matrix<_out_rows, _in_rows> jacobian(
        const Callable &func, 
        const Vector<_in_rows> &pt, const double delta = DEFAULT_STEP_SIZE)
{
    IS_GREATER(delta, 0);

    Matrix<_out_rows, _in_rows> jacobian; jacobian.setZero();

    for (int i = 0; i < _in_rows; ++i)
    {
        Vector<_in_rows> pt_positive = pt;
        pt_positive[i] += delta;

        Vector<_in_rows> pt_negative = pt;
        pt_negative[i] -= delta;

        jacobian.col(i) = (func(pt_positive) - func(pt_negative)) / (2.0 * delta);
    }

    return jacobian;
}

template<int _rows, typename Callable>
Matrix<_rows, _rows> hessian(
        const Callable &func, 
        const Vector<_rows> &pt, const double delta = DEFAULT_STEP_SIZE)
{
    IS_GREATER(delta, 0);

    // Precompute constants.
    const double two_delta = 2.0*delta;
    const double delta_sq = delta*delta;
    const double ij_eq_div = 12.*delta_sq;
    const double ij_neq_div = 4.*delta_sq;
    
    Matrix<_rows, _rows> hessian; hessian.setZero();
    // Central difference approximation based on 
    // https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = i; j < _rows; ++j)
        {
            Vector<_rows> p1 = pt;
            Vector<_rows> p2 = pt;
            Vector<_rows> p3 = pt;
            Vector<_rows> p4 = pt;
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

template<int _rows>
Matrix<_rows, _rows> project_to_psd(const Matrix<_rows, _rows> &symmetric_mat, const double min_eigval)
{
    // Cheap check to confirm matrix is square.
    IS_EQUAL(symmetric_mat.rows(), symmetric_mat.cols());

    // O(n^2) check to make sure that the matrix is symmetric.
    IS_TRUE(math::is_symmetric(symmetric_mat));

    const Eigen::SelfAdjointEigenSolver<Matrix<_rows, _rows>> es(symmetric_mat);
    Matrix<_rows, _rows> evals = es.eigenvalues().asDiagonal();
    const Matrix<_rows, _rows> evecs = es.eigenvectors();

    // Indices of eigen values that are less than the minimum eigenvalue threshold.
    int num_failed_evals = 0;
    const int num_evals = evals.rows();
    for (int i = 0; i < num_evals; ++i)
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

    // We can reconstruct with P D P\inv = P D P^T if P is orthonormal.
    const Matrix<_rows, _rows> projection = evecs * evals * evecs.transpose() ;
    return projection;
}

template<int _rows>
void check_psd(const Matrix<_rows, _rows> &mat, const double min_eigval)
{
    // Cheap check to confirm matrix is square.
    IS_EQUAL(mat.rows(), mat.cols());

    // O(n^2) check to make sure that the matrix is symmetric.
    IS_TRUE(math::is_symmetric(mat));

    const Eigen::SelfAdjointEigenSolver<Matrix<_rows, _rows>> es(mat, Eigen::EigenvaluesOnly);
    const auto evals = es.eigenvalues();

    for (int i = 0; i < evals.size(); ++i)
    {
        // Throw an exception if this is not true.
        IS_GREATER_EQUAL(evals(i), min_eigval);
    }
}

} // namespace math 
