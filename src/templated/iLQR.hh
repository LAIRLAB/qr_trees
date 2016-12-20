// 
// Implements iLQR (on a traditional chain) for nonlinear dynamics and cost.
//

#pragma once


#include <templated/taylor_expansion.hh>
#include <utils/debug_utils.hh>

#include <Eigen/Dense>

#include <tuple>
#include <vector>

namespace ilqr
{

// Defined in templated/taylor_expansion.hh
//template<int _rows, int _cols>
//using Matrix = Eigen::Matrix<double, _rows, _cols>;
//
//template<int _rows>
//using Vector = Eigen::Matrix<double, _rows, 1>;
//
//template<int _xdim, int _udim>
//using DynamicsPtr = Vector<_xdim>(Vector<_xdim>, Vector<_udim>);
//
//template<int _xdim, int _udim>
//using CostPtr = double(Vector<_xdim>, Vector<_udim>);

template<int _xdim, int _udim>
class iLQR
{
    static_assert(_xdim > 0, "State dimension should be greater than 0");
    static_assert(_udim > 0, "Control dimension should be greater than 0");
public:
    iLQR(const int T,
         DynamicsPtr<_xdim,_udim> dynamics, 
         CostPtr<_xdim,_udim> final_cost,
         CostPtr<_xdim,_udim> cost,
         const Vector<_udim> u_nominal,
         const Vector<_xdim> x_init)
    {
        T_ = T;
        this->true_dynamics_ = dynamics;
        this->true_cost_ = cost;
        this->true_final_cost_ = final_cost;
        u_nominal_ = u_nominal;
        x_init_ = x_init_;

        this->uhat = std::vector<Vector<_udim>>(T, u_nominal);
        this->xhat = std::vector<Vector<_xdim>>(T, Vector<_xdim>::Zero());

        this->Ks_ = std::vector<Matrix<_udim, _xdim>>(T+1, Matrix<_udim, _xdim>::Zero());
        this->ks_ = std::vector<Vector<_udim>>(T+1, Vector<_udim>::Zero());
    }

    void solve()
    {
        uhat = std::vector<Vector<_udim>>(T_, u_nominal_);
        xhat = std::vector<Vector<_xdim>>(T_+1, Vector<_xdim>::Zero());

        std::vector<Vector<_udim>> uhat_new (T_, Vector<_udim>::Zero());
        std::vector<Vector<_xdim>> xhat_new(T_+1, Vector<_xdim>::Zero());

        double old_cost = std::numeric_limits<double>::infinity();
        double new_cost = 0;
        do
        {
            new_cost = 0;
            xhat_new[0] = x_init_;
            for (int t = 0; t < T_; ++t)
            {
                const Matrix<_udim, _xdim> &Kt = Ks_[t];
                const Vector<_udim> &kt = ks_[t];
                Vector<_xdim> zt = (xhat_new[t] - xhat[t]);

                const Vector<_udim> vt = Kt * zt + kt;

                uhat_new[t] = vt + uhat[t];

                const double cost = true_cost_(xhat_new[t], uhat_new[t]);
                new_cost += cost;

                // Roll forward the dynamics.
                xhat_new[t+1] = true_dynamics_(xhat_new[t], uhat_new[t]);
            }
            new_cost += true_final_cost_(xhat_new[T_], Vector<_udim>::Zero());
        } while(0);

        xhat = xhat_new;
        uhat = uhat_new;

        // prevent unused variable
        if (old_cost < new_cost)
        {
        }

        old_cost = new_cost;

        Matrix<_xdim, _xdim> Vt1; Vt1.setZero();
        Matrix<1, _xdim> Gt1; Gt1.setZero();

        // Backwards pass
        for (int t = T_; t != -1; --t)
        {
            Matrix<_xdim, _xdim> A;
            Matrix<_xdim, _udim> B;
            linearize_dynamics(this->true_dynamics_, xhat[t], uhat[t], A, B);

            Matrix<_xdim,_xdim> Q;
            Matrix<_udim,_udim> R;
            Matrix<_xdim,_udim> P;
            Vector<_xdim> g_x;
            Vector<_udim> g_u;
            double c;
            if (t == T_)
            {
                quadratize_cost(this->true_final_cost_, xhat[t], Vector<_udim>::Zero(), Q, R, P, g_x, g_u, c);
            }
            else
            {
                quadratize_cost(this->true_cost_, xhat[t], uhat[t], Q, R, P, g_x, g_u, c);
            }

            const Matrix<_udim, _udim> inv_term = -1.0*(R + B.transpose()*Vt1*B).inverse();
            const Matrix<_udim, _xdim> Kt = inv_term * (P.transpose() + B.transpose()*Vt1*A); 
            const Vector<_udim> kt = inv_term * (g_u + B.transpose()*Gt1.transpose());

            const Matrix<_xdim, _xdim> tmp = (A + B*Kt);
            const Matrix<_xdim, _xdim> Vt = Q + 2.0*(P*Kt) + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp;

            const Matrix<1, _xdim> Gt = kt.transpose()*P.transpose() + kt.transpose()*R*Kt + g_x.transpose() 
                + g_u.transpose()*Kt + kt.transpose()*B.transpose()*Vt1*tmp + Gt1*tmp;
            Vt1 = Vt;
            Gt1 = Gt;
            Ks_[t] = Kt;
            ks_[t] = kt;
        }
    }


private:
    int T_ = -1; // time horizon

    DynamicsPtr<_xdim, _udim> *true_dynamics_; 
    CostPtr<_xdim, _udim> *true_cost_; 
    CostPtr<_xdim, _udim> *true_final_cost_; 

    Vector<_udim> u_nominal_;
    Vector<_xdim> x_init_;

    // Feedback control gains.
    std::vector<Matrix<_udim, _xdim>> Ks_;
    std::vector<Vector<_udim>> ks_;

    // Linearization points.
    std::vector<Vector<_xdim>> xhat;
    std::vector<Vector<_udim>> uhat;

};

} // namespace lqr

