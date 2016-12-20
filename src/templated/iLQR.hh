// 
// Implements iLQR (on a traditional chain) for nonlinear dynamics and cost.
//

#pragma once


#include <templated/taylor_expansion.hh>
#include <utils/debug_utils.hh>

#include <Eigen/Dense>

#include <tuple>
#include <vector>
#include <ostream>

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

template<int _dim>
std::ostream& operator<<(std::ostream &os, const std::vector<Vector<_dim>> &vectors)
{
    for (size_t t = 0; t < vectors.size(); ++t)
    {
        const Vector<_dim> &vec = vectors[t];
        os << "t=" << t  << ": " << vec.transpose(); 
        if (t < vectors.size() -1)
        {
            os << std::endl;
        }
    }
    return os;
}

template<int _xdim, int _udim>
class iLQRSolver
{
    static_assert(_xdim > 0, "State dimension should be greater than 0");
    static_assert(_udim > 0, "Control dimension should be greater than 0");
public:
    iLQRSolver(const int T,
         DynamicsPtr<_xdim,_udim> dynamics, 
         CostPtr<_xdim,_udim> final_cost,
         CostPtr<_xdim,_udim> cost
         )
    {
        T_ = T;
        this->true_dynamics_ = dynamics;
        this->true_cost_ = cost;
        this->true_final_cost_ = final_cost;

        this->uhat_ = std::vector<Vector<_udim>>(T, Vector<_udim>::Zero());
        this->xhat_ = std::vector<Vector<_xdim>>(T, Vector<_xdim>::Zero());

        this->Ks_ = std::vector<Matrix<_udim, _xdim>>(T+1, Matrix<_udim, _xdim>::Zero());
        this->ks_ = std::vector<Vector<_udim>>(T+1, Vector<_udim>::Zero());
    }

    inline Vector<_udim> compute_control_stepsize(const Vector<_xdim> &xt, int t, double alpha) const
    {
        const Matrix<_udim, _xdim> &Kt = Ks_[t];
        const Vector<_udim> &kt = ks_[t];

        const Vector<_xdim> zt = (xt - xhat_[t]);
        const Vector<_udim> vt = Kt * zt + alpha*kt;

        return Vector<_udim>(vt + uhat_[t]);
    }

    // Computes the control at timestep t at xt.
    inline Vector<_udim> compute_control(const Vector<_xdim> &xt, int t) const
    {
        return compute_control_stepsize(xt, t, 1.0);
    }

    inline void solve(const Vector<_xdim> x_init, const Vector<_udim> u_nominal, bool verbose = false)
    {
        const int max_iters = 30;

        Ks_ = std::vector<Matrix<_udim, _xdim>>(T_+1, Matrix<_udim, _xdim>::Zero());
        ks_ = std::vector<Vector<_udim>>(T_+1, Vector<_udim>::Zero());

        uhat_ = std::vector<Vector<_udim>>(T_, u_nominal);
        xhat_ = std::vector<Vector<_xdim>>(T_+1, Vector<_xdim>::Zero());

        std::vector<Vector<_udim>> uhat_new (T_, Vector<_udim>::Zero());
        std::vector<Vector<_xdim>> xhat_new(T_+1, Vector<_xdim>::Zero());

        double old_cost = std::numeric_limits<double>::infinity();
        int iter = 0;
        for (iter = 0; iter < max_iters; ++iter)
        {
            double new_cost = 0;
            // Line search as decribed in 
            // https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
            double alpha = 1.0;
            do
            {
                new_cost = 0;
                xhat_new[0] = x_init;
                for (int t = 0; t < T_; ++t)
                {
                    uhat_new[t] = compute_control_stepsize(xhat_new[t], t, alpha); 

                    const double cost = true_cost_(xhat_new[t], uhat_new[t]);
                    new_cost += cost;

                    // Roll forward the dynamics.
                    const Vector<_xdim> xt1 = true_dynamics_(xhat_new[t], uhat_new[t]);
                    xhat_new[t+1] = xt1;
                }
                new_cost += true_final_cost_(xhat_new[T_], Vector<_udim>::Zero());
                // If we fail to do better than before, then try to half the step size.
                // It will stop halfing it when the ratio (change in cost-to-go) gets too small.
                // That is, we have a step size for which we have converged. 
                alpha *= 0.5;
            } while(!(new_cost < old_cost 
                    || std::abs((old_cost - new_cost) / new_cost) < 1e-4));
            // Since we always half it at the end of the iteration, double it
            alpha = 2.0*alpha;

            xhat_ = xhat_new;
            uhat_ = uhat_new;

            const double cost_diff_ratio = std::abs((old_cost - new_cost) / new_cost);
            if (verbose)
            {
                PRINT("[Iter " << iter << "]: Alpha: " << alpha << ", Cost ratio: " << cost_diff_ratio 
                        << ", New Cost: " << new_cost
                        << ", Old Cost: " << old_cost);
            }

            if (cost_diff_ratio < 1e-4) 
            {
                break;
            }

            old_cost = new_cost;

            Matrix<_xdim, _xdim> Vt1; Vt1.setZero();
            Matrix<1, _xdim> Gt1; Gt1.setZero();

            // Backwards pass
            for (int t = T_; t != -1; --t)
            {
                Matrix<_xdim, _xdim> A; A.setZero();
                Matrix<_xdim, _udim> B; B.setZero();
                linearize_dynamics(this->true_dynamics_, xhat_[t], uhat_[t], A, B);

                Matrix<_xdim,_xdim> Q; Q.setZero();
                Matrix<_udim,_udim> R; R.setZero();
                Matrix<_xdim,_udim> P; P.setZero();
                Vector<_xdim> g_x; g_x.setZero();
                Vector<_udim> g_u; g_u.setZero();
                double c = 0;
                if (t == T_)
                {
                    quadratize_cost(this->true_final_cost_, xhat_[t], Vector<_udim>::Zero(), 
                            Q, R, P, g_x, g_u, c);
                }
                else
                {
                    quadratize_cost(this->true_cost_, xhat_[t], uhat_[t], 
                            Q, R, P, g_x, g_u, c);
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

        SUCCESS("Converged after " << iter << " iterations.");
    }


private:
    int T_ = -1; // time horizon

    DynamicsPtr<_xdim, _udim> *true_dynamics_; 
    CostPtr<_xdim, _udim> *true_cost_; 
    CostPtr<_xdim, _udim> *true_final_cost_; 

    // Feedback control gains.
    std::vector<Matrix<_udim, _xdim>> Ks_;
    std::vector<Vector<_udim>> ks_;

    // Linearization points.
    std::vector<Vector<_xdim>> xhat_;
    std::vector<Vector<_udim>> uhat_;

};

} // namespace lqr

