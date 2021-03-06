//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// December 2016
//

#pragma once

#include <utils/math_utils_temp.hh>

#include <Eigen/Dense>

#include <sstream>
#include <vector>

namespace ilqr
{

template<int _rows, int _cols>
using Matrix = Eigen::Matrix<double, _rows, _cols>;

template<int _rows>
using Vector = Eigen::Matrix<double, _rows, 1>;

// Helper function for debugging a sequence of vectors.
template<int _dim>
std::string time_print(const std::vector<Vector<_dim>> &vectors)
{
    std::ostringstream oss;
    for (size_t t = 0; t < vectors.size(); ++t)
    {
        const Vector<_dim> &vec = vectors[t];
        oss << "t=" << t  << ": " << vec.transpose();
        if (t < vectors.size() -1)
        {
            oss << std::endl;
        }
    }
    return oss.str();
}


template<int _xdim, int _udim, typename DynamicsFunc>
void linearize_dynamics(const DynamicsFunc &dynamics_func, 
                        const Vector<_xdim> &x, 
                        const Vector<_udim> &u,
                        Matrix<_xdim, _xdim> &A,
                        Matrix<_xdim, _udim> &B
                       )
{
    const auto helper = [&dynamics_func](const Vector<_xdim+_udim> &xu) -> Vector<_xdim>
    { 
        Vector<_xdim> x = xu.topRows(_xdim);
        Vector<_udim> u = xu.bottomRows(_udim);
        return Vector<_xdim>(dynamics_func(x,u));
    };

    Vector<_xdim + _udim> xu;
    xu.topRows(_xdim) = x;
    xu.bottomRows(_udim) = u;
    const Matrix<_xdim, _xdim+_udim> J 
        = math::jacobian<_xdim+_udim, _xdim, decltype(helper)>(helper, xu);

    A = J.leftCols(_xdim);
    B = J.rightCols(_udim);
}

template<int _xdim, int _udim, typename CostFunc>
void quadratize_cost(const CostFunc &cost_func, 
                     const int t,
                     const Eigen::VectorXd &x, 
                     const Eigen::VectorXd &u,
                     Matrix<_xdim,_xdim> &Q,
                     Matrix<_udim,_udim> &R,
                     Matrix<_xdim,_udim> &P,
                     Vector<_xdim> &g_x,
                     Vector<_udim> &g_u
                     )
{
    const auto helper = [&cost_func, t](const Vector<_xdim+_udim> &xu) -> double
    { 
        Vector<_xdim> x = xu.topRows(_xdim);
        Vector<_udim> u = xu.bottomRows(_udim);
        return double(cost_func(x,u,t));
    };

    Vector<_xdim + _udim> xu;
    xu.topRows(_xdim) = x;
    xu.bottomRows(_udim) = u;

    constexpr double ZERO_THRESH = 1e-7;

    Vector<_xdim+_udim> g 
        = math::gradient<_xdim+_udim, decltype(helper)>(helper, xu);
    g = g.array() * (g.array().abs() > ZERO_THRESH).template cast<double>();
    g_x = g.topRows(_xdim);
    g_u = g.bottomRows(_udim);


    // Zero out components that are less than this threshold. We do this since
    // finite differencing has numerical issues.
    Matrix<_xdim+_udim,_xdim+_udim> H 
        = math::hessian<_xdim+_udim, decltype(helper)>(helper, xu);
    //Eigen::MatrixXd H = g * g.transpose();
    H = H.array() * (H.array().abs() > ZERO_THRESH).template cast<double>();
    Q = H.topLeftCorner(_xdim, _xdim);
    P = H.topRightCorner(_xdim, _udim);
    R = H.bottomRightCorner(_udim, _udim);

    Q = (Q + Q.transpose())/2.0;
    Q = math::project_to_psd(Q, 1e-11);
    //math::check_psd(Q, 1e-12);

    // Control terms.
    R = math::project_to_psd(R, 1e-8);
    //math::check_psd(R, 1e-9);
}

template<int _xdim, typename CostFunc>
void quadratize_cost(const CostFunc &cost_func, 
                     const Eigen::VectorXd &x, 
                     Matrix<_xdim,_xdim> &Q,
                     Vector<_xdim> &g
                     )
{
    constexpr double ZERO_THRESH = 1e-7;

    g = math::gradient<_xdim, CostFunc>(cost_func, x);
    g = g.array() * (g.array().abs() > ZERO_THRESH).template cast<double>();


    // Zero out components that are less than this threshold. We do this since
    // finite differencing has numerical issues.
    Matrix<_xdim,_xdim> H 
        = math::hessian<_xdim, CostFunc>(cost_func, x);
    //Eigen::MatrixXd H = g * g.transpose();
    Q = H.array() * (H.array().abs() > ZERO_THRESH).template cast<double>();

    Q = (Q + Q.transpose())/2.0;
    Q = math::project_to_psd(Q, 1e-11);
    math::check_psd(Q, 1e-12);
}

} // namespace ilqr
