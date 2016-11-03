#include <ilqr/types.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace
{
    // Minimum eigenvalue for the PSD Q and PD R matrices in the quadratic cost.
    constexpr double Q_MIN_EVAL = 1e-7;
    constexpr double R_MIN_EVAL = 1e-3;
} // namespace

PlanNode::PlanNode(int state_dim, 
                   int control_dim, 
                   const DynamicsFunc &dynamics_func, 
                   const CostFunc &cost_func, 
                   const double probability) :
    probability_(probability),
    state_dim_(state_dim),
    control_dim_(control_dim)
{
    // Confirm the probablity is within [0,1].
    IS_BETWEEN_INCLUSIVE(probability, 0.0, 1.0);

    // Initialize the extended-state [x,1] dynamics matrices.
    dynamics_.A = Eigen::MatrixXd::Zero(state_dim + 1, state_dim + 1);
    dynamics_.A(state_dim, state_dim) = 1.0; // last element (bottom right corner) should be 1.
    dynamics_.B = Eigen::MatrixXd::Zero(state_dim + 1, control_dim);

    // Initialize the extended-state [x,1] and control cost matrices.
    cost_.Q = Eigen::MatrixXd::Identity(state_dim + 1, state_dim + 1);
    cost_.P = Eigen::MatrixXd::Zero(state_dim + 1, control_dim);
    cost_.R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    cost_.b_u = Eigen::VectorXd(control_dim);

    // Initialize other matrices
    K_ = Eigen::MatrixXd::Zero(control_dim, state_dim);  
    V_ = Eigen::MatrixXd::Identity(state_dim + 1, state_dim + 1);  

    x_ = Eigen::VectorXd(state_dim + 1);
    u_ = Eigen::VectorXd(control_dim);

    x_star_ = Eigen::VectorXd(state_dim + 1);
    u_star_ = Eigen::VectorXd(control_dim);

    xu_ = Eigen::VectorXd(state_dim + control_dim);

    dynamics_wrapper_ = [state_dim, control_dim, dynamics_func](const Eigen::VectorXd &pt) 
    { 
        return dynamics_func(pt.topRows(state_dim), pt.bottomRows(control_dim));
    };

    cost_wrapper_ = [state_dim, control_dim, cost_func](const Eigen::VectorXd &pt) 
    { 
        return cost_func(pt.topRows(state_dim), pt.bottomRows(control_dim));
    };
}

void PlanNode::update_dynamics()
{
    const Eigen::MatrixXd AB = math::jacobian(dynamics_wrapper_, xu_);
    IS_EQUAL(AB.rows(), state_dim_);
    IS_EQUAL(AB.cols(), state_dim_ + control_dim_);
    dynamics_.A.topLeftCorner(state_dim_, state_dim_) = AB.topLeftCorner(state_dim_, state_dim_);
    dynamics_.B = AB.topRightCorner(state_dim_, control_dim_);

    // The offset (0th order term) for the Taylor expansion.
    const Eigen::VectorXd a = dynamics_wrapper_(xu_);
    IS_EQUAL(a.size(), state_dim_);
    dynamics_.A.topRightCorner(state_dim_, 1) = a;

    // Confirm we didn't change the bottom right corner. 
    IS_EQUAL(dynamics_.A(state_dim_, state_dim_), 1.0);
}

void PlanNode::update_cost()
{
    const Eigen::MatrixXd H = math::hessian(cost_wrapper_, xu_);
    IS_EQUAL(H.rows(), state_dim_ + control_dim_);
    IS_EQUAL(H.cols(), state_dim_ + control_dim_);

    const Eigen::VectorXd g = math::gradient(cost_wrapper_, xu_);
    IS_EQUAL(g.size(), state_dim_ + control_dim_);

    // The constant offset (0th order term) of the Taylor expansion.
    const double c = cost_wrapper_(xu_);

    // Set the extended-state quadratic cost term.
    cost_.Q.topLeftCorner(state_dim_, state_dim_) = H.topLeftCorner(state_dim_, state_dim_);
    cost_.Q.topRightCorner(state_dim_, 1) = g.topRows(state_dim_);
    cost_.Q.bottomLeftCorner(1, state_dim_) = g.topRows(state_dim_).transpose();
    cost_.Q(state_dim_, state_dim_) = c;

    cost_.Q = math::project_to_psd(cost_.Q, Q_MIN_EVAL);

    cost_.P.topRows(state_dim_) = H.topRightCorner(state_dim_, control_dim_);

    cost_.R = H.bottomRightCorner(control_dim_, control_dim_);
    cost_.R = math::project_to_psd(cost_.R, R_MIN_EVAL);

    cost_.b_u = g.bottomRows(control_dim_);
}

void PlanNode::set_xstar(const Eigen::VectorXd &x_star)
{
    IS_EQUAL(x_star.size(), state_dim_);
    x_star_.topRows(state_dim_) = x_star;
}

void PlanNode::set_ustar(const Eigen::VectorXd &u_star) 
{
    IS_EQUAL(u_star.size(), control_dim_);
    u_star_ = u_star;
}
void PlanNode::set_x(const Eigen::VectorXd &x)
{
    IS_EQUAL(x.size(), state_dim_);
    x_.topRows(state_dim_) = x;
    xu_.topRows(state_dim_) = x;
}
void PlanNode::set_u(const Eigen::VectorXd &u)
{
    IS_EQUAL(u.size(), control_dim_);
    u_ = u;
    xu_.bottomRows(control_dim_) = u;
}

std::ostream& operator<<(std::ostream& os, const PlanNode& node)
{
    os << "x: [" << node.x().transpose() << "], u: [" << node.u().transpose() << "]";
    return os;
}
