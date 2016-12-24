
#pragma once

#include <templated/iLQR_hindsight.hh>

#include <ostream>

namespace ilqr
{

// Computes the control at timestep t at xt.
// :param alpha - Backtracking line search parameter. Setting to 1 gives regular forward pass.
template<int xdim, int udim>
inline Vector<udim> iLQRHindsightSolver<xdim,udim>::compute_control_stepsize(
        const int branch_num, const Vector<xdim> &xt, const int t, const double alpha) const
{
    IS_BETWEEN_LOWER_INCLUSIVE(branch_num, 0, branches_.size());
    const HindsightBranch<xdim,udim> &branch = branches_[branch_num];

    const Vector<xdim> xhat = branch.xhat[t];
    const Vector<udim> uhat = branch.uhat[t];

    const Matrix<udim, xdim> Kt = branch.Ks[t];
    const Vector<udim> kt = branch.ks[t];

    const Vector<xdim> zt = (xt - xhat);
    const Vector<udim> vt = Kt * zt + alpha*kt;

    return Vector<udim>(vt + uhat);
}

template<int xdim, int udim>
inline double iLQRHindsightSolver<xdim,udim>::forward_pass(const int branch_num,
            const Vector<xdim> x_init,  
            std::vector<Vector<xdim>> &states,
            std::vector<Vector<udim>> &controls,
            const double alpha
        ) const
{
    IS_BETWEEN_LOWER_INCLUSIVE(branch_num, 0, branches_.size());

    const int T = timesteps();

    const HindsightBranch<xdim,udim> &branch = branches_[branch_num];
    const auto &dynamics_fnc = branch.split.dynamics;
    const auto &cost_fnc = branch.split.cost;
    const auto &final_cost_fnc = branch.split.final_cost;

    controls.resize(T);
    states.resize(T+1);

    states[0] = x_init;
    double cost_to_go = 0;
    for (int t = 0; t < T; ++t)
    {
        controls[t] = compute_control_stepsize(branch_num, states[t], t, alpha); 

        const double cost = cost_fnc(states[t], controls[t], t);
        cost_to_go += cost;

        // Roll forward the dynamics.
        const Vector<xdim> xt1 = dynamics_fnc(states[t], controls[t]);
        states[t+1] = xt1;
    }
    const double final_cost = final_cost_fnc(states[T]); 
    cost_to_go += final_cost;

    return cost_to_go;
}

template<int xdim, int udim>
inline void iLQRHindsightSolver<xdim,udim>::solve(const int T, 
        const Vector<xdim> &x_init, const Vector<udim> u_nominal, 
        const double mu, const int max_iters, bool verbose, 
        const double cost_convg_ratio, const double start_alpha)
{
    IS_GREATER(T, 1);
    IS_GREATER_EQUAL(mu, 0);
    IS_GREATER(max_iters, 0);
    IS_GREATER(cost_convg_ratio, 0);
    IS_GREATER(start_alpha, 0);

    x0_ = Vector<xdim>::Zero();
    u0_ = u_nominal;
    K0_ = Matrix<udim,xdim>::Zero();
    k0_ = Vector<udim>::Zero();

    // Initialize each branch
    const int num_branches = branches_.size(); 
    for (int branch_num = 0; branch_num < num_branches; ++branch_num)
    {
        HindsightBranch<xdim,udim> &branch = branches_[branch_num];
        branch.Ks = std::vector<Matrix<udim, xdim>>(T, Matrix<udim, xdim>::Zero());
        branch.ks = std::vector<Vector<udim>>(T, Vector<udim>::Zero());

        branch.uhat = std::vector<Vector<udim>>(T, u_nominal);
        branch.xhat = std::vector<Vector<xdim>>(T+1, Vector<xdim>::Zero());
    }

    // Holders used for the line search.
    std::vector<Vector<udim>> uhat_new (T, Vector<udim>::Zero());
    std::vector<Vector<xdim>> xhat_new(T+1, Vector<xdim>::Zero());

    double old_cost = std::numeric_limits<double>::infinity();
    int iter = 0;
    for (iter = 0; iter < max_iters; ++iter)
    {
        // Line search as decribed in 
        // http://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
        
        // Initial step-size
        double alpha = start_alpha;

        // The step-size adaptation paramter
        constexpr double beta = 0.5; 
        static_assert(beta > 0.0 && beta < 1.0, 
            "Step-size adaptation parameter should decrease the step-size");
        
        // Before we start line search, NaN makes sure termination conditions 
        // won't be met.
        double new_cost = std::numeric_limits<double>::quiet_NaN();
        double cost_diff_ratio = std::abs((old_cost - new_cost) / new_cost);

        while(!(new_cost < old_cost || cost_diff_ratio < cost_convg_ratio))
        {
            new_cost = 0;
            for (int branch_num = 0; branch_num < num_branches; ++branch_num)
            {
                const double branch_new_cost 
                    = forward_pass(branch_num, x_init, xhat_new, uhat_new, alpha);
                const double p = branches_[branch_num].split.probability;
                new_cost +=  p * branch_new_cost;
            }

            cost_diff_ratio = std::abs((old_cost - new_cost) / new_cost);

            // Try decreasing the step-size by beta-times. 
            alpha *= beta;
        } 

        // Since we always beta-it at the end of the iteration, invert it to get 
        // the step-size that worked.
        alpha = (1.0/beta)*alpha;

        // For each branch with this step size, 
        std::vector<Vector<xdim>> x0s(num_branches);
        std::vector<Vector<udim>> u0s(num_branches);
        for (int branch_num = 0; branch_num < num_branches; ++branch_num)
        {
            forward_pass(branch_num, x_init, xhat_new, uhat_new, alpha);
            x0s[branch_num] = xhat_new[0];
            u0s[branch_num] = uhat_new[0];

            branches_[branch_num].xhat.swap(xhat_new);
            branches_[branch_num].uhat.swap(uhat_new);
        }
        //TODO check that all branches have the same value.
        x0_ = x0s[0];
        u0_ = u0s[0];

        if (verbose)
        {
            PRINT("[Iter " << iter << "]: Alpha: " << alpha 
                    << ", Cost ratio: " << cost_diff_ratio 
                    << ", New Cost: " << new_cost
                    << ", Old Cost: " << old_cost);
        }

        if (cost_diff_ratio < cost_convg_ratio) 
        {
            break;
        }

        old_cost = new_cost;

        // Backup each branch separately.
        std::vector<Matrix<xdim,xdim>> branch_V1(num_branches);
        std::vector<Matrix<1, xdim>> branch_G1(num_branches);;
        for (int branch_num = 0; branch_num < num_branches; ++branch_num)
        {
            const HindsightBranch<xdim,udim> &branch = branches_[branch_num];
            Matrix<xdim,xdim> Vt1; Vector<xdim> Gt1;
            quadratize_cost(branch.split.final_cost, branch.xhat.back(), Vt1, Gt1);

            // Storage for backing up the value function.
            Matrix<xdim, xdim> Vt; 
            Matrix<1, xdim> Gt;

            // Backwards pass for this branch.
            for (int t = T-1; t != -1; --t)
            {
                if (t > 0)
                {
                    bellman_backup(branch_num, t, mu, Vt1, Gt1, Vt, Gt);
                    Vt1 = Vt;
                    Gt1 = Gt;
                }
            }
            branch_V1[branch_num] = Vt1; 
            branch_G1[branch_num] = Gt1; 
        }

        // Now merge the branches at the top timestep.
        Matrix<udim,udim> weighted_inv_term; weighted_inv_term.setZero();
        Matrix<udim,xdim> weighted_Kt_term; weighted_Kt_term.setZero();
        Vector<udim> weighted_kt_term; weighted_kt_term.setZero();

        // Levenberg-Marquardt parameter for damping. 
        // ie. eigenvalue inflation matrix.
        const Matrix<xdim, xdim> LM = mu * Matrix<xdim, xdim>::Identity();

        // w stands for weighted by probability of each branch.
        Matrix<xdim,xdim> wQ; wQ.setZero();
        Matrix<udim,udim> wR; wR.setZero();
        Matrix<xdim,udim> wP; wP.setZero();
        Vector<xdim> wg_x; wg_x.setZero();
        Vector<udim> wg_u; wg_u.setZero();
        for (int branch_num = 0; branch_num < num_branches; ++branch_num)
        {
            const HindsightBranch<xdim,udim> &branch = branches_[branch_num];
            const double p = branch.split.probability;

            const Vector<xdim> &x = x0_;
            const Vector<udim> &u = u0_;
            
            Matrix<xdim, xdim> A; 
            Matrix<xdim, udim> B;
            linearize_dynamics(branch.split.dynamics, x, u, A, B);

            const Matrix<xdim,xdim> &Vt1 = branch_V1[branch_num];
            const Matrix<1,xdim> &Gt1 = branch_G1[branch_num];
            weighted_inv_term.noalias() += p * (B.transpose()*(Vt1+LM)*B);
            weighted_Kt_term.noalias()  += p * (B.transpose()*(Vt1+LM)*A);
            weighted_kt_term.noalias()  += p * (B.transpose()*Gt1.transpose());

            Matrix<xdim,xdim> Q; Matrix<udim,udim> R; Matrix<xdim,udim> P;
            Vector<xdim> g_x; Vector<udim> g_u;
            quadratize_cost(branch.split.cost, 0, x, u, Q, R, P, g_x, g_u);

            wQ += p * Q;
            wR += p * R;
            wP += p * P;
            wg_x += p * g_x;
            wg_u += p * g_u;
        }

        const Matrix<udim,udim> inv_term = -1.0*(wR + weighted_inv_term).inverse();
        K0_.noalias() = inv_term * (wP.transpose() + weighted_Kt_term);
        k0_.noalias() = inv_term * (wg_u + weighted_kt_term);
    }

    // Copy the first timestep K to all the branches.
    for (int branch_num = 0; branch_num < num_branches; ++branch_num)
    {
        HindsightBranch<xdim,udim> &branch = branches_[branch_num];
        branch.Ks[0] = K0_;
        branch.ks[0] = k0_;
    }

    if (verbose)
    {
        SUCCESS("Converged after " << iter << " iterations.");
    }
}
 

template<int xdim, int udim>
inline void iLQRHindsightSolver<xdim,udim>::bellman_backup(
        const int branch_num,
        const int t,
        const double mu, // Levenberg-Marquardt parameter
        const Matrix<xdim,xdim> &Vt1, 
        const Matrix<1,xdim> &Gt1, 
        Matrix<xdim,xdim> &Vt, 
        Matrix<1,xdim> &Gt
        )
{
    HindsightBranch<xdim,udim> &branch = branches_[branch_num];
    const Vector<xdim> x = branch.xhat[t];
    const Vector<udim> u = branch.uhat[t];

    Matrix<xdim, xdim> A; 
    Matrix<xdim, udim> B;
    linearize_dynamics(branch.split.dynamics, x, u, A, B);

    Matrix<xdim,xdim> Q;
    Matrix<udim,udim> R;
    Matrix<xdim,udim> P;
    Vector<xdim> g_x;
    Vector<udim> g_u;
    quadratize_cost(branch.split.cost, t, x, u, Q, R, P, g_x, g_u);

    // Levenberg-Marquardt parameter for damping. 
    // ie. eigenvalue inflation matrix.
    const Matrix<xdim, xdim> LM = mu * Matrix<xdim, xdim>::Identity();

    const Matrix<udim, udim> inv_term 
        = -1.0*(R + B.transpose()*(Vt1+LM)*B).inverse();

    branch.Ks[t] = inv_term * (P.transpose() + B.transpose()*(Vt1+LM)*A); 
    branch.ks[t] = inv_term * (g_u + B.transpose()*Gt1.transpose());
    const Matrix<udim, xdim> &Kt = branch.Ks[t];
    const Vector<udim> &kt = branch.ks[t];

    const Matrix<xdim, xdim> tmp = (A + B*Kt);
    Vt = Q + 2.0*(P*Kt) 
        + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp;

    Gt = kt.transpose()*P.transpose() 
        + kt.transpose()*R*Kt + g_x.transpose() 
        + g_u.transpose()*Kt + kt.transpose()*B.transpose()*Vt1*tmp + Gt1*tmp;
}

template<int xdim, int udim>
inline int iLQRHindsightSolver<xdim,udim>::timesteps() const
{ 
    IS_GREATER(branches_.size(), 0);
    const size_t T = branches_[0].uhat.size(); 
    // Confirm that all the required parts the same size
    IS_EQUAL(T, branches_[0].ks.size());
    IS_EQUAL(T, branches_[0].Ks.size());
    IS_EQUAL(T+1, branches_[0].xhat.size());
    return static_cast<int>(T); 
}


} // namespace ilqr
