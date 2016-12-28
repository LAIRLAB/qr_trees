// 
// Implements iLQR (on a traditional chain) for nonlinear dynamics and cost.
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
struct HindsightBranch
{
    using Dynamics = std::function<Vector<xdim>(const Vector<xdim> &x, const Vector<udim> &u)>;
    using Cost = std::function<double(const Vector<xdim> &x, const Vector<udim> &u, const int t)>;
    using FinalCost = std::function<double(const Vector<xdim> &x)>;

    HindsightBranch(const Dynamics &dyn, const FinalCost &cost_final, const Cost &cost_regular, const double prob)
        : dynamics(dyn), final_cost(cost_final), cost(cost_regular), probability(prob)
    {
    }

    Dynamics dynamics;
    FinalCost final_cost;
    Cost cost;

    double probability = 0;

    // Feedback control gains.
    std::vector<Matrix<udim, xdim>> Ks;
    std::vector<Vector<udim>> ks;

    // Linearization points.
    std::vector<Vector<xdim>> xhat;
    std::vector<Vector<udim>> uhat;
};

template<int xdim, int udim>
class iLQRHindsightSolver
{
    static_assert(xdim > 0, "State dimension should be greater than 0");
    static_assert(udim > 0, "Control dimension should be greater than 0");
public:
    using Dynamics = std::function<Vector<xdim>(const Vector<xdim> &x, const Vector<udim> &u)>;
    using Cost = std::function<double(const Vector<xdim> &x, const Vector<udim> &u, const int t)>;
    using FinalCost = std::function<double(const Vector<xdim> &x)>;

    iLQRHindsightSolver(const std::vector<HindsightBranch<xdim,udim>> &branches)
    {
        IS_GREATER(branches.size(), 0);
        branches_ = branches; 
        double total_prob = 0;
        for (const auto &branch : branches_)
        {
            total_prob += branch.probability;
        }
        IS_ALMOST_EQUAL(total_prob, 1.0, 1e-3);
    }

    // Computes the control at timestep 0 using K0_, k0_ that is
    // shared across all branches.
    inline Vector<udim> compute_first_control(const Vector<xdim> &x0) const;

    // Computes the control at timestep t at xt.
    // :param alpha - Backtracking line search parameter. 
    //      Setting to 1 gives regular forward pass.
    inline Vector<udim> compute_control_stepsize(const int branch_num, const Vector<xdim> &xt, const int t, const double alpha) const;

    // :param alpha - Backtracking line search parameter. 
    //      Setting to 1 gives regular forward pass.
    inline double forward_pass(const int branch_num,
            const Vector<xdim> x_init, 
            std::vector<Vector<xdim>> &states, std::vector<Vector<udim>> &controls, 
            const double alpha ) const;


    // :param x_init - Initial state from which to start the system from.
    // :param u_nominal - Initial control used for the whole sequence during 
    //      the first forward pass.
    // :param mu - Levenberg-Marquardt parameter for damping the least-squares. 
    //      Setting it to 0 gets the default behavior. The damping makes the 
    //      state-space steps smaller over iterations. 
    inline void solve(const int T, const Vector<xdim> &x_init, 
            const Vector<udim> u_nominal, const double mu, 
            const int max_iters = 1000, bool verbose = false, 
            const double cost_convg_ratio = 1e-4, const double start_alpha = 1.0,
            const bool warm_start = false, const int t_offset = 0);

    // Returns how many timesteps we have computed control policies for.
    inline int timesteps() const;

private:
    std::vector<HindsightBranch<xdim,udim>> branches_;

    // Feedback control gains.
    Matrix<udim, xdim> K0_ = Matrix<udim, xdim>::Zero();
    Vector<udim> k0_ = Vector<udim>::Zero();

    // Linearization points for the first timestep.
    Vector<xdim> xhat0_ = Vector<xdim>::Zero();
    Vector<udim> uhat0_ = Vector<udim>::Zero();

    // Performs one timestep of the bellman backup.
    // :param t - passed to the cost runction
    // :param mu - Levenberg-Marquardt parameter
    void bellman_backup(const int branch_num, const int t, const double mu, 
        const Matrix<xdim,xdim> &Vt1, const Matrix<1,xdim> &Gt1, 
        Matrix<xdim,xdim> &Vt, Matrix<1,xdim> &Gt);

};

} // namespace lqr

#include <templated/iLQR_hindsight_impl.hh>

