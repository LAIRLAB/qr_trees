#pragma once

#include <templated/iLQR.hh>
#include <templated/taylor_expansion.hh>

#include <Eigen/Dense>

template<int _rows>
using Vector = ilqr::Vector<_rows>;

template<int _rows, int _cols>
using Matrix = ilqr::Matrix<_rows, _cols>;

template<int _xdim, int _udim>
Vector<_xdim> linear_dynamics(const Vector<_xdim> &x, const Vector<_udim> &u)
{
    Matrix<_xdim, _udim> B = 2*Matrix<_xdim, _udim>::Identity();
    return x + B*u;
}

template<int _xdim, int _udim>
double quadratic_cost(const Vector<_xdim> &x, const Vector<_udim> &u)
{
    Matrix<_xdim, _xdim> Q = 5*Matrix<_xdim, _xdim>::Identity();
    Matrix<_udim, _udim> R = 2*Matrix<_udim, _udim>::Identity();
    const Vector<1> c = 0.5*(x.transpose()*Q*x + u.transpose()*R*u);
    return c[0];
}
