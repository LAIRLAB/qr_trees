#include <ilqr/types.hh>

#include <utils/debug_utils.hh>

PlanNode::PlanNode(int state_dim, int control_dim) :
    state_dim_(state_dim),
    control_dim_(control_dim)
{
    // Initialize the extended-state [x,1] dynamics matrices.
    dynamics.A = Eigen::MatrixXd(state_dim + 1, state_dim + 1);
    dynamics.B = Eigen::MatrixXd(state_dim + 1, control_dim);

    // Initialize the extended-state [x,1] and control cost matrices.
    cost.Q = Eigen::MatrixXd(state_dim + 1, state_dim + 1);
    cost.P = Eigen::MatrixXd(state_dim + 1, control_dim);
    cost.R = Eigen::MatrixXd(control_dim, control_dim);
    cost.b_u = Eigen::VectorXd(control_dim);

    // Initialize other matrices
    K = Eigen::MatrixXd(control_dim, state_dim);  
    V = Eigen::MatrixXd(state_dim + 1, state_dim + 1);  

    x_ = Eigen::VectorXd(state_dim + 1);
    u_ = Eigen::VectorXd(control_dim);

    x_star_ = Eigen::VectorXd(state_dim + 1);
    u_star_ = Eigen::VectorXd(control_dim);

    xu_star_ = Eigen::VectorXd(state_dim + control_dim);
}

void PlanNode::set_xstar(const Eigen::VectorXd &x_star)
{
    IS_EQUAL(x_star.size(), state_dim_);
    x_star_.topRows(state_dim_) = x_star;
    xu_star_.topRows(state_dim_) = x_star;
}

void PlanNode::set_ustar(const Eigen::VectorXd &u_star) 
{
    IS_EQUAL(u_star.size(), control_dim_);
    u_star_ = u_star;
    xu_star_.bottomRows(control_dim_) = u_star;
}
void PlanNode::set_x(const Eigen::VectorXd &x)
{
    x_.topRows(state_dim_) = x;
}
void PlanNode::set_u(const Eigen::VectorXd &u)
{
    u_ = u;
}
