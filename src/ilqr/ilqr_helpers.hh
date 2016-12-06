
#pragma once

#include <Eigen/Dense>

namespace ilqr
{

// Dynamics Function Prototype. Takes state, control and returns the next state.
using DynamicsFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&,const Eigen::VectorXd&)>; 

// Cost Function Prototype. Takes state, control and returns a real-valued cost.
using CostFunc = std::function<double(const Eigen::VectorXd&,const Eigen::VectorXd&)>; 

// Linearized dynamics parameters in terms of the state [x].
struct Dynamics
{
    // Linear dynamics matrix. [dim(x)] x [dim(x)]
    // (extension is last row is [\vec{0}, 1])
    Eigen::MatrixXd A;

    // Controls matrix. [dim(x)] x [dim(u)]
    // (extension is last row is [\vec{0}])
    Eigen::MatrixXd B;
};

// Quadratic cost parameters in terms of the state x and control u.
struct Cost 
{
    // Quadratic state-cost matrix. [dim(x)] x [dim(x)]
    Eigen::MatrixXd Q;

    // Cost cross-term matrix. [dim(x)] x [dim(u)]
    Eigen::MatrixXd P;

    // Quadratic control-cost matrix. [dim(u)] x [dim(u)]
    Eigen::MatrixXd R;

    // Linear control-cost  matrix. [dim(u)] x [1]
    Eigen::VectorXd g_u;

    // Linear control-cost matrix. [dim(x)] x [1]
    Eigen::VectorXd g_x;

    // Constant offset term
    double c;
};


// Taylor series expansion points of dynamics and cost functions.
struct TaylorExpansion
{
    Eigen::VectorXd x;
    Eigen::VectorXd u;
    ilqr::Dynamics dynamics;
    ilqr::Cost cost;
};


// Helper class to store the terms of the quadratic value function.
class QuadraticValue 
{
public:
    // Inititalizes the Value function terms V_, G_, W_ to zeros.
    QuadraticValue(const int state_dim);

    Eigen::MatrixXd& V() { return V_; }
    Eigen::MatrixXd& G() { return G_; }
    double& W() { return W_; }

    const Eigen::MatrixXd& V() const { return V_; }
    const Eigen::MatrixXd& G() const { return G_; }
    const double& W() const { return W_; }

private:
    // Quadratic term [dim(x)] x [dim(x)]
    Eigen::MatrixXd V_;
    // Linear term [1] x [dim(x)]
    Eigen::MatrixXd G_;
    // Constant term [1] x [1]
    double W_;
};

// Compute a standard iLQR backup.
void compute_backup(const TaylorExpansion &expansion, const QuadraticValue &Jt1,
        Eigen::MatrixXd &Kt, Eigen::VectorXd &kt, QuadraticValue &Jt);
        
void update_dynamics(const DynamicsFunc &dynamics_func, TaylorExpansion &expansion);

ilqr::Dynamics linearize_dynamics(const DynamicsFunc &dynamics_func, 
                        const Eigen::VectorXd &x, 
                        const Eigen::VectorXd &u);

void update_cost(const CostFunc &cost_func, TaylorExpansion &expansion);

ilqr::Cost quadraticize_cost(const CostFunc &cost_func, 
                        const Eigen::VectorXd &x, 
                        const Eigen::VectorXd &u);

} // namespace ilqr

