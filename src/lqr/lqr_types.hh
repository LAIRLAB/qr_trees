//
// Helper data structures and types for Tree-iLQR.
//

#pragma once

#include <Eigen/Dense>

#include <functional>
#include <ostream>

namespace lqr 
{

// Dynamics Function Prototype. Takes state, control and returns the next state.
using DynamicsFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&,const Eigen::VectorXd&)>; 

// Cost Function Prototype. Takes state, control and returns a real-valued cost.
using CostFunc = std::function<double(const Eigen::VectorXd&,const Eigen::VectorXd&)>; 

// Linearized dynamics parameters in terms of the extended-state [x, 1].
struct Dynamics
{
    // Extended linear dynamics matrix. [dim(x) + 1] x [dim(x) + 1]
    // (extension is last row is [\vec{0}, 1])
    Eigen::MatrixXd A;

    // Extended controls matrix. [dim(x) + 1] x [dim(u)]
    // (extension is last row is [\vec{0}])
    Eigen::MatrixXd B;
};

// Quadratic cost parameters in terms of the extended-state [x, 1].
struct Cost 
{
    // Extended quadratic state-cost matrix. [dim(x) + 1] x [dim(x) + 1]
    Eigen::MatrixXd Q;

    // Extended quadratic cross-term matrix. [dim(x) + 1] x [dim(u)]
    Eigen::MatrixXd P;

    // Quadratic control-cost matrix. [dim(u)] x [dim(u)]
    Eigen::MatrixXd R;

    // Linear control-cost  matrix. [dim(u)] x [1]
    Eigen::VectorXd b_u;
};

// Each plan node represents a timestep.
class PlanNode
{
public:
    PlanNode(int state_dim, 
             int control_dim, 
             const DynamicsFunc &dynamics_func, 
             const CostFunc &cost_func,
             const double probablity);

    // Throws exception if a size of an item (dynamics, cost, etc.) 
    // doesn't match expected sizes. Used for debugging. 
    void check_sizes();

    // Linearized dynamics.
    Dynamics dynamics_;

    // Quadratic approximation of the cost.
    Cost cost_;

    // Feedback gain matrix on the extended-state, [dim(u)] x [dim(x) + 1]
    Eigen::MatrixXd K_; 
    // Feed-forward control matrix, [dim(u)] x [dim(1)].
    Eigen::VectorXd k_; 

    // Value matrix in x^T V x. [dim(x) + 1] x [dim(x) + 1]
    Eigen::MatrixXd V_; 

    double probability_;

    // Uses the x_ and u_ to update the linearization of the dynamics.
    void update_dynamics();

    // Uses the x_ and u_ to update the quadraticization of the cost.
    void update_cost();

    //
    // Get and set the  nominal xstar and ustar.
    //
    void set_xstar(const Eigen::VectorXd &x_star);
    void set_ustar(const Eigen::VectorXd &u_star);
    // Returns the nominal extended-state [xstar, 1].
    const Eigen::VectorXd& xstar() const { return x_star_; }
    // Returns the nominal control [ustar].
    const Eigen::VectorXd& ustar() const { return u_star_; };

    //
    // Get and set the forward iLQR pass x and u.
    //
    
    // Set the state with a [dim(x)]  (note, not extended) vector.
    void set_x(const Eigen::VectorXd &x);
    void set_u(const Eigen::VectorXd &u);
    // Returns the extended-state [x, 1] for the iLQR forward pass.
    const Eigen::VectorXd& x() const { return x_; }
    // Returns the control [u] for the iLQR forward pass.
    const Eigen::VectorXd& u() const { return u_; };

    const DynamicsFunc& dynamics_func() const { return dynamics_func_; };
    const CostFunc& cost_func() const { return cost_func_; };

    void set_dynamics_func(const DynamicsFunc& dynamics_func) { dynamics_func_ = dynamics_func; };
    void set_cost_func(const CostFunc& cost_func) { cost_func_ = cost_func; };

private:
    int state_dim_;
    int control_dim_;

    DynamicsFunc dynamics_func_;
    CostFunc cost_func_;

    // Helper types needed to call the math utils functions.
    using DynamicsWrapper = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
    using CostWrapper = std::function<double(const Eigen::VectorXd&)>;
    DynamicsWrapper dynamics_wrapper_;
    CostWrapper cost_wrapper_;

    // State from this iLQR forward pass. [dim(x)] x [1]
    Eigen::VectorXd x_; 
    // Control from this iLQR forward pass. [dim(u)] x [1]
    Eigen::VectorXd u_; 
    // Combined vector [x, u], used to call numerical differentiators (e.g. Jacobian).
    Eigen::VectorXd xu_;

    //TODO: Do I need these? Maybe for adding a cost function term lambda*(x-x*)I(x-x*)
    // Nominal (desired) state specified at the beginning of iLQR. [dim(x)] x [1]
    Eigen::VectorXd x_star_; 
    // Nominal (desired) control specified at the beginning of iLQR. [dim(u)] x [1]
    Eigen::VectorXd u_star_; 
};

// Allows the PlanNode to be printed.
std::ostream& operator<<(std::ostream& os, const lqr::PlanNode& node);

} // namespace ilqr

