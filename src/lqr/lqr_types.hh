//
// Helper data structures and types for Tree-iLQR.
//

#pragma once

#include <Eigen/Dense>

#include <functional>
#include <ostream>

namespace lqr 
{

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

    // Quadratic control-cost matrix. [dim(u)] x [dim(u)]
    Eigen::MatrixXd R;
};

// Each plan node represents a timestep.
class PlanNode
{
public:
    PlanNode(int state_dim, 
             int control_dim, 
             const Eigen::MatrixXd A,
             const Eigen::MatrixXd B,
             const Eigen::MatrixXd Q,
             const Eigen::MatrixXd R,
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

    // Value matrix in x^T V x. [dim(x) + 1] x [dim(x) + 1]
    Eigen::MatrixXd V_; 

    double probability_;

    // Uses the x_ and u_ to update the linearization of the dynamics.
    void update_dynamics();

    // Uses the x_ and u_ to update the quadraticization of the cost.
    void update_cost();

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

private:
    int state_dim_;
    int control_dim_;

    // State from this iLQR forward pass. [dim(x)] x [1]
    Eigen::VectorXd x_; 
    // Control from this iLQR forward pass. [dim(u)] x [1]
    Eigen::VectorXd u_; 
    // Combined vector [x, u], used to call numerical differentiators (e.g. Jacobian).
    Eigen::VectorXd xu_;

};

// Allows the PlanNode to be printed.
std::ostream& operator<<(std::ostream& os, const lqr::PlanNode& node);

} // namespace ilqr

