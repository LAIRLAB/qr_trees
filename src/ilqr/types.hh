//
// Helper data structures and types for Tree-iLQR.
//

#pragma once

#include <functional>

#include <Eigen/Dense>

// Takes state, control and returns the next state.
using DynamicsFunc = std::function<Eigen::VectorXd(Eigen::VectorXd,Eigen::VectorXd)>; 

// Takes state, control and returns a real-valued cost.
using CostFunc = std::function<double(Eigen::VectorXd,Eigen::VectorXd)>; 

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
friend std::ostream& operator<<(std::ostream& os, const PlanNode& node);

public:
    PlanNode(int state_dim, int control_dim);

    // Linearized dynamics.
    Dynamics dynamics;

    // Quadratic approximation of the cost.
    Cost cost;

    // Feedback gain matrix, [dim(u)] x [dim(x)]
    Eigen::MatrixXd K; 

    // Value matrix in x^T V x. [dim(x) + 1] x [dim(x) + 1]
    Eigen::MatrixXd V; 

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

    // Nominal (desired) state specified at the beginning of iLQR. [dim(x)] x [1]
    Eigen::VectorXd x_star_; 
    // Nominal (desired) control specified at the beginning of iLQR. [dim(u)] x [1]
    Eigen::VectorXd u_star_; 
    // Combined vector [x*, u*], used to call numerical differentiators (e.g. Jacobian).
    Eigen::VectorXd xu_star_;
};

std::ostream& operator<<(std::ostream& os, const PlanNode& node)
{
    os << "x: " << node.x_ << "u: " << node.u_;
    return os;
}
