#include <ilqr/types.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace ilqr
{


PlanNode::PlanNode(const int state_dim, 
                   const int control_dim, 
                   const DynamicsFunc &dynamics_func, 
                   const CostFunc &cost_func, 
                   const double probability) :
    dynamics_func_(dynamics_func),
    cost_func_(cost_func),
    probability_(probability),
    J_(state_dim)
{
    // No point controlling 0 dimensional state or control spaces.
    IS_GREATER(state_dim, 0);
    IS_GREATER(control_dim, 0);

    // Confirm the probablity is within [0,1].
    IS_BETWEEN_INCLUSIVE(probability, 0.0, 1.0);

    // Initialize other matrices
    K_ = Eigen::MatrixXd::Zero(control_dim, state_dim);  
    k_ = Eigen::VectorXd(control_dim);  
}

PlanNode::PlanNode(const Eigen::VectorXd &x_star,
                   const Eigen::VectorXd &u_star, 
                   const DynamicsFunc &dynamics_func, 
                   const CostFunc &cost_func, 
                   const double probability) :
    PlanNode(x_star.size(), u_star.size(), dynamics_func, cost_func, probability)
{
    orig_xstar_ = x_star;
    orig_ustar_ = u_star;
    update_expansion(x_star, u_star);
}

void PlanNode::update_expansion(const Eigen::VectorXd &x, const Eigen::VectorXd &u)
{
    expansion_.x = x;
    expansion_.u = u;
    ilqr::update_dynamics(dynamics_func_, expansion_);
    ilqr::update_cost(cost_func_, expansion_);
}

void PlanNode::bellman_backup(const QuadraticValue &Jt1)
{
    ilqr::compute_backup(expansion_, Jt1, K_, k_, J_);
}

std::ostream& operator<<(std::ostream& os, const PlanNode& node)
{
    os << "x: [" << node.x().transpose() << "], u: [" << node.u().transpose() << "]";
    return os;
}

} // namespace ilqr
