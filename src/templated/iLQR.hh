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
    iLQRSolver(DynamicsPtr<_xdim,_udim> dynamics, 
         CostPtr<_xdim,_udim> final_cost,
         CostPtr<_xdim,_udim> cost
         )
    {
        this->true_dynamics_ = dynamics;
        this->true_cost_ = cost;
        this->true_final_cost_ = final_cost;
    }

    // Computes the control at timestep t at xt.
    // :param alpha - Backtracking line search parameter. Setting to 1 gives regular forward pass.
    inline Vector<_udim> compute_control_stepsize(const Vector<_xdim> &xt, int t, double alpha) const
    {
        const Matrix<_udim, _xdim> &Kt = Ks_[t];
        const Vector<_udim> &kt = ks_[t];

        const Vector<_xdim> zt = (xt - xhat_[t]);
        const Vector<_udim> vt = Kt * zt + alpha*kt;

        return Vector<_udim>(vt + uhat_[t]);
    }

    // :param x_init - Initial state from which to start the system from.
    // :param u_nominal - Initial control used for the whole sequence during the first forward pass.
    // :param mu - Levenberg-Marquardt parameter for damping the least-squares. Setting it to 0 gets
    //      the default behavior. The damping makes the state-space steps smaller over
    //      iterations. 
    inline void solve(const int T, const Vector<_xdim> &x_init, const Vector<_udim> u_nominal, 
            double mu = 0, const int max_iters = 1000, bool verbose = false)
    {
        IS_GREATER(T, 0);
        IS_GREATER_EQUAL(mu, 0);
        IS_GREATER(max_iters, 0);

        constexpr double COST_RATIO_CONVG = 1e-4;

        Ks_ = std::vector<Matrix<_udim, _xdim>>(T+1, Matrix<_udim, _xdim>::Zero());
        ks_ = std::vector<Vector<_udim>>(T+1, Vector<_udim>::Zero());

        uhat_ = std::vector<Vector<_udim>>(T, u_nominal);
        xhat_ = std::vector<Vector<_xdim>>(T+1, Vector<_xdim>::Zero());

        std::vector<Vector<_udim>> uhat_new (T, Vector<_udim>::Zero());
        std::vector<Vector<_xdim>> xhat_new(T+1, Vector<_xdim>::Zero());

        // Levenberg-Marquardt parameter for damping. ie. eigenvalue inflation matrix.
        const Matrix<_xdim, _xdim> LM = mu * Matrix<_xdim, _xdim>::Identity();

        double old_cost = std::numeric_limits<double>::infinity();
        int iter = 0;
        for (iter = 0; iter < max_iters; ++iter)
        {
            // Line search as decribed in 
            // http://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
            // https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
            
            // Initial step-size
            double alpha = 1.0;

            // The step-size adaptation paramter
            constexpr double beta = 0.5; 
            static_assert(beta > 0.0 && beta < 1.0, 
                    "Step-size adaptation parameter should decrease the step-size");
            
            // Before we start line search, NaN makes sure termination conditions won't be met.
            double new_cost = std::numeric_limits<double>::quiet_NaN();
            double cost_diff_ratio = std::abs((old_cost - new_cost) / new_cost);

            while(!(new_cost < old_cost || cost_diff_ratio < COST_RATIO_CONVG))
            {
                new_cost = forward_pass(x_init, xhat_new, uhat_new, alpha);

                cost_diff_ratio = std::abs((old_cost - new_cost) / new_cost);

                // Try decreasing the step-size by beta-times. 
                alpha *= beta;
            } 

            xhat_ = xhat_new;
            uhat_ = uhat_new;

            if (verbose)
            {
                // Since we always half it at the end of the iteration, double it
                alpha = (1.0/beta)*alpha;

                PRINT("[Iter " << iter << "]: Alpha: " << alpha << ", Cost ratio: " << cost_diff_ratio 
                        << ", New Cost: " << new_cost
                        << ", Old Cost: " << old_cost);
            }

            if (cost_diff_ratio < COST_RATIO_CONVG) 
            {
                break;
            }

            old_cost = new_cost;

            Matrix<_xdim, _xdim> Vt1; Vt1.setZero();
            Matrix<1, _xdim> Gt1; Gt1.setZero();

            // Backwards pass
            for (int t = T; t != -1; --t)
            {
                Matrix<_xdim, _xdim> A; 
                Matrix<_xdim, _udim> B;
                linearize_dynamics(this->true_dynamics_, xhat_[t], uhat_[t], A, B);

                Matrix<_xdim,_xdim> Q;
                Matrix<_udim,_udim> R;
                Matrix<_xdim,_udim> P;
                Vector<_xdim> g_x;
                Vector<_udim> g_u;
                double c = 0;
                if (t == T)
                {
                    quadratize_cost(this->true_final_cost_, xhat_[t], Vector<_udim>::Zero(), 
                            Q, R, P, g_x, g_u, c);
                }
                else
                {
                    quadratize_cost(this->true_cost_, xhat_[t], uhat_[t], 
                            Q, R, P, g_x, g_u, c);
                }

                const Matrix<_udim, _udim> inv_term = -1.0*(R + B.transpose()*(Vt1+LM)*B).inverse();
                const Matrix<_udim, _xdim> Kt = inv_term * (P.transpose() + B.transpose()*(Vt1+LM)*A); 
                const Vector<_udim> kt = inv_term * (g_u + B.transpose()*Gt1.transpose());

                const Matrix<_xdim, _xdim> tmp = (A + B*Kt);
                const Matrix<_xdim, _xdim> Vt = Q + 2.0*(P*Kt) 
                    + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp;

                const Matrix<1, _xdim> Gt = kt.transpose()*P.transpose() 
                    + kt.transpose()*R*Kt + g_x.transpose() 
                    + g_u.transpose()*Kt + kt.transpose()*B.transpose()*Vt1*tmp + Gt1*tmp;
                Vt1 = Vt;
                Gt1 = Gt;
                Ks_[t] = Kt;
                ks_[t] = kt;
            }
        }

        if (verbose)
        {
            SUCCESS("Converged after " << iter << " iterations.");
        }
    }

    // :param alpha - Backtracking line search parameter. Setting to 1 gives regular forward pass.
    inline double forward_pass(const Vector<_xdim> x_init,  
                std::vector<Vector<_xdim>> &states,
                std::vector<Vector<_udim>> &controls,
                double alpha
            ) const
    {
        const int T = timesteps();

        controls.resize(T);
        states.resize(T+1);

        states[0] = x_init;
        double cost_to_go = 0;
        for (int t = 0; t < T; ++t)
        {
            controls[t] = compute_control_stepsize(states[t], t, alpha); 

            const double cost = true_cost_(states[t], controls[t]);
            cost_to_go += cost;

            // Roll forward the dynamics.
            const Vector<_xdim> xt1 = true_dynamics_(states[t], controls[t]);
            states[t+1] = xt1;
        }
        const double final_cost = true_final_cost_(states[T], Vector<_udim>::Zero());
        cost_to_go += final_cost;

        return cost_to_go;
    }

    // Returns how many timesteps we have computed control policies for.
    int timesteps() const
    { 
        const size_t T = uhat_.size(); 
        // Confirm that all the required parts the same size
        IS_EQUAL(T+1, ks_.size());
        IS_EQUAL(T+1, Ks_.size());
        IS_EQUAL(T+1, xhat_.size());
        return static_cast<int>(T); 
    }

private:
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

