#include <ilqr/types.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace
{
    // Minimum eigenvalue for the PSD Q and PD R matrices in the quadratic cost.
    constexpr double Q_MIN_EVAL = 1e-7;
    constexpr double R_MIN_EVAL = 1e-3;
} // namespace

namespace ilqr
{


PlanNode::PlanNode(int state_dim, 
                   int control_dim, 
                   const DynamicsFunc &dynamics_func, 
                   const CostFunc &cost_func, 
                   const double probability) :
    probability_(probability),
    state_dim_(state_dim),
    control_dim_(control_dim),
    dynamics_func_(dynamics_func),
    cost_func_(cost_func)
{
    // Confirm the probablity is within [0,1].
    IS_BETWEEN_INCLUSIVE(probability, 0.0, 1.0);

    // Initialize the extended-state [x,1] dynamics matrices.
    dynamics_.A = Eigen::MatrixXd::Zero(state_dim + 1, state_dim + 1);
    dynamics_.A(state_dim, state_dim) = 1.0; // last element (bottom right corner) should be 1.
    dynamics_.B = Eigen::MatrixXd::Zero(state_dim + 1, control_dim);

    // Initialize the extended-state [x,1] and control cost matrices.
    cost_.Q = Eigen::MatrixXd::Zero(state_dim + 1, state_dim + 1);
    cost_.P = Eigen::MatrixXd::Zero(state_dim + 1, control_dim);
    cost_.R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    cost_.b_u = Eigen::VectorXd(control_dim);

    // Initialize other matrices
    K_ = Eigen::MatrixXd::Zero(control_dim, state_dim+1);  
    k_ = Eigen::VectorXd(control_dim);  

    V_ = Eigen::MatrixXd::Identity(state_dim + 1, state_dim + 1);  

    x_ = Eigen::VectorXd::Zero(state_dim + 1);
    x_(state_dim) = 1;
    u_ = Eigen::VectorXd::Zero(control_dim);

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

void PlanNode::check_sizes()
{
    IS_EQUAL(dynamics_.A.rows(), state_dim_ + 1);
    IS_EQUAL(dynamics_.A.cols(), state_dim_ + 1);
    IS_EQUAL(dynamics_.B.rows(), state_dim_ + 1);
    IS_EQUAL(dynamics_.B.cols(), control_dim_);

    // Initialize the extended-state [x,1] and control cost matrices.
    IS_EQUAL(cost_.Q.rows(), state_dim_ + 1);
    IS_EQUAL(cost_.Q.cols(), state_dim_ + 1);
    IS_EQUAL(cost_.P.rows(), state_dim_ + 1);
    IS_EQUAL(cost_.P.cols(), control_dim_);
    IS_EQUAL(cost_.R.rows(), control_dim_);
    IS_EQUAL(cost_.R.cols(), control_dim_);
    IS_EQUAL(cost_.b_u.size(), control_dim_);

    // Initialize other matrices
    IS_EQUAL(K_.rows(), control_dim_);
    IS_EQUAL(K_.cols(), state_dim_ + 1);
    IS_EQUAL(k_.size(), control_dim_);

    IS_EQUAL(V_.rows(), state_dim_ + 1);
    IS_EQUAL(V_.cols(), state_dim_ + 1);

    IS_EQUAL(x_.size(), state_dim_ + 1);
    IS_EQUAL(u_.size(), control_dim_);

    IS_EQUAL(x_star_.size(), state_dim_ + 1);
    IS_EQUAL(u_star_.size(), control_dim_);

    IS_EQUAL(xu_.size(), state_dim_ + control_dim_);
}

void PlanNode::update_dynamics()
{
    IS_EQUAL(xu_.topRows(state_dim_), x_.topRows(state_dim_));
    IS_EQUAL(xu_.bottomRows(control_dim_), u_);
    IS_EQUAL(x_.bottomRows(1)[0], 1.0);
    const Eigen::MatrixXd AB = math::jacobian(dynamics_wrapper_, xu_);
    IS_EQUAL(AB.rows(), state_dim_);
    IS_EQUAL(AB.cols(), state_dim_ + control_dim_);
    // The A matrix is augmented A := [A, a; 0, 1]. 
    dynamics_.A.topLeftCorner(state_dim_, state_dim_) 
        = AB.topLeftCorner(state_dim_, state_dim_);
    // The B is zero-augmented on the last rows to support the extended-state.
    dynamics_.B.topLeftCorner(state_dim_, control_dim_) 
        = AB.topRightCorner(state_dim_, control_dim_);

    // The offset (0th order term) for the Taylor expansion.
    const Eigen::VectorXd a = dynamics_wrapper_(xu_);
    IS_EQUAL(a.size(), state_dim_);
    dynamics_.A.topRightCorner(state_dim_, 1) = a;

    // Confirm we didn't change the bottom right corner. 
    IS_EQUAL(dynamics_.A(state_dim_, state_dim_), 1.0);
}

void PlanNode::update_cost()
{
    Eigen::MatrixXd H = math::hessian(cost_wrapper_, xu_, 1e-3);
    // Remove small elements due to numerical precision issues.
    H = H.array() * (H.array().abs() > 1e-5).cast<double>();

    IS_EQUAL(H.rows(), state_dim_ + control_dim_);
    IS_EQUAL(H.cols(), state_dim_ + control_dim_);

    Eigen::VectorXd g = math::gradient(cost_wrapper_, xu_);
    // Remove small elements due to numerical precision issues.
    g = g.array() * (g.array() > 1e-5).cast<double>();
    IS_EQUAL(g.size(), state_dim_ + control_dim_);

    // The constant offset (0th order term) of the Taylor expansion.
    const double c = cost_wrapper_(xu_);

    // Set the extended-state quadratic cost term.
    cost_.Q.topLeftCorner(state_dim_, state_dim_) 
        = H.topLeftCorner(state_dim_, state_dim_);
    cost_.Q.topRightCorner(state_dim_, 1) = g.topRows(state_dim_);
    cost_.Q.bottomLeftCorner(1, state_dim_) = g.topRows(state_dim_).transpose();
    cost_.Q(state_dim_, state_dim_) = 2.0*c;

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
    if (x.size() ==  state_dim_)
    {
        x_.topRows(state_dim_) = x;
        xu_.topRows(state_dim_) = x;
    }
    else if (x.size() ==  state_dim_ + 1)
    {
        x_ = x;
        xu_.topRows(state_dim_) = x.topRows(state_dim_);
    }
    else
    {
        throw std::logic_error("set_x input size (" + std::to_string(x.size()) 
                + ") should be state_dim (" + std::to_string(state_dim_) 
                + ") or state_dim + 1");
    }
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

} // namespace ilqr
