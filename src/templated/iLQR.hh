// 
// Implements iLQR (on a traditional chain) for nonlinear dynamics and cost.
//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// December 2016
//

#pragma once


#include <templated/taylor_expansion.hh>
#include <utils/debug_utils.hh>

#include <Eigen/Dense>

#include <functional>
#include <vector>

namespace ilqr
{

// Defined in templated/taylor_expansion.hh
//template<int _rows, int _cols>
//using Matrix = Eigen::Matrix<double, _rows, _cols>;
//
//template<int _rows>
//using Vector = Eigen::Matrix<double, _rows, 1>;


template<int xdim, int udim>
class iLQRSolver
{
    static_assert(xdim > 0, "State dimension should be greater than 0");
    static_assert(udim > 0, "Control dimension should be greater than 0");
public:
    using Dynamics = std::function<Vector<xdim>(const Vector<xdim> &x, const Vector<udim> &u)>;
    using Cost = std::function<double(const Vector<xdim> &x, const Vector<udim> &u, const int t)>;
    using FinalCost = std::function<double(const Vector<xdim> &x)>;

    iLQRSolver(const Dynamics &dynamics,
         const FinalCost &final_cost,
         const Cost &cost
         )
    {
        this->dynamics_ = dynamics;
        this->cost_ = cost;
        this->final_cost_ = final_cost;
    }

    // Computes the control at timestep t at xt.
    // :param alpha - Backtracking line search parameter. Setting to 1 gives regular forward pass.
    inline Vector<udim> compute_control_stepsize(const Vector<xdim> &xt, 
            const int t, const double alpha) const;

    // :param x_init - Initial state from which to start the system from.
    // :param u_nominal - Initial control used for the whole sequence during 
    //      the first forward pass.
    // :param mu - Levenberg-Marquardt parameter for damping the least-squares. 
    //      Setting it to 0 gets the default behavior. The damping makes the 
    //      state-space steps smaller over iterations. 
    inline void solve(const int T, const Vector<xdim> &x_init, 
            const Vector<udim> u_nominal, const double mu, 
            const int max_iters = 1000, bool verbose = false, 
            const double cost_convg_ratio = 1e-4, const double start_alpha = 1.0);

    // :param alpha - Backtracking line search parameter. Setting to 1 gives regular forward pass.
    inline double forward_pass(const Vector<xdim> x_init, 
            std::vector<Vector<xdim>> &states, std::vector<Vector<udim>> &controls, 
            const double alpha ) const;

    // Returns how many timesteps we have computed control policies for.
    inline int timesteps() const;

private:
    Dynamics dynamics_; 
    Cost cost_; 
    FinalCost final_cost_; 

    // Feedback control gains.
    std::vector<Matrix<udim, xdim>> Ks_;
    std::vector<Vector<udim>> ks_;

    // Linearization points.
    std::vector<Vector<xdim>> xhat_;
    std::vector<Vector<udim>> uhat_;

    // Performs one timestep of the bellman backup.
    // :param t - passed to the cost runction
    // :param mu - Levenberg-Marquardt parameter
    void bellman_backup(const int t, const double mu, 
        const Matrix<xdim,xdim> &Vt1, const Matrix<1,xdim> &Gt1, 
        Matrix<xdim,xdim> &Vt, Matrix<1,xdim> &Gt, 
        Matrix<udim,xdim> &Kt, Vector<udim> &kt);

};

} // namespace lqr

#include <templated/iLQR_impl.hh>

