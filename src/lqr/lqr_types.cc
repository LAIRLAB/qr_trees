#include <lqr/lqr_types.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace
{
    // Minimum eigenvalue for the PSD Q and PD R matrices in the quadratic cost.
    constexpr double Q_MIN_EVAL = 1e-7;
    constexpr double R_MIN_EVAL = 1e-3;
} // namespace

namespace lqr
{


PlanNode::PlanNode(int state_dim, 
                   int control_dim, 
                   const Eigen::MatrixXd A,
                   const Eigen::MatrixXd B,
                   const Eigen::MatrixXd Q,
                   const Eigen::MatrixXd R,
                   const double probability) :
    probability_(probability),
    state_dim_(state_dim),
    control_dim_(control_dim)
{
    // Confirm the probablity is within [0,1].
    IS_BETWEEN_INCLUSIVE(probability, 0.0, 1.0);

    // Initialize the extended-state [x,1] dynamics matrices.
    dynamics_.A = A;
    dynamics_.B = B;

    // Initialize the extended-state [x,1] and control cost matrices.
    cost_.Q = Q;
    math::check_psd(Q, Q_MIN_EVAL);
    cost_.R = R;
    math::check_psd(R, R_MIN_EVAL);

    // Initialize other matrices
    K_ = Eigen::MatrixXd::Zero(control_dim, state_dim);  

    V_ = Eigen::MatrixXd::Identity(state_dim, state_dim);  

    x_ = Eigen::VectorXd::Zero(state_dim);
    u_ = Eigen::VectorXd::Zero(control_dim);

    xu_ = Eigen::VectorXd(state_dim + control_dim);
}

void PlanNode::check_sizes()
{
    IS_EQUAL(dynamics_.A.rows(), state_dim_);
    IS_EQUAL(dynamics_.A.cols(), state_dim_);
    IS_EQUAL(dynamics_.B.rows(), state_dim_);
    IS_EQUAL(dynamics_.B.cols(), control_dim_);

    // Initialize the extended-state [x,1] and control cost matrices.
    IS_EQUAL(cost_.Q.rows(), state_dim_);
    IS_EQUAL(cost_.Q.cols(), state_dim_);
    IS_EQUAL(cost_.R.rows(), control_dim_);
    IS_EQUAL(cost_.R.cols(), control_dim_);

    // Initialize other matrices
    IS_EQUAL(K_.rows(), control_dim_);
    IS_EQUAL(K_.cols(), state_dim_);

    IS_EQUAL(V_.rows(), state_dim_);
    IS_EQUAL(V_.cols(), state_dim_);

    IS_EQUAL(x_.size(), state_dim_);
    IS_EQUAL(u_.size(), control_dim_);

    IS_EQUAL(xu_.size(), state_dim_ + control_dim_);
}

void PlanNode::update_dynamics()
{
   // do nothing since actually linear dynamics.
}

void PlanNode::update_cost()
{
   // do nothing since actually quadratic cost.
}

void PlanNode::set_x(const Eigen::VectorXd &x)
{
    if (x.size() ==  state_dim_)
    {
        x_.topRows(state_dim_) = x;
        xu_.topRows(state_dim_) = x;
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
