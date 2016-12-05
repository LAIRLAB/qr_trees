//
// Underlying iLQRNode for Tree-iLQR.
//

#include <ilqr/ilqr_node.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace
{
void add_weighted_value(const ilqr::QuadraticValue &a, 
                        const double probability, 
                        ilqr::QuadraticValue &b) 
{
    b.V() += probability * a.V();
    b.G() += probability * a.G();
    b.W() += probability * a.W();
}
}

namespace ilqr
{


iLQRNode::iLQRNode(const int state_dim, 
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

iLQRNode::iLQRNode(const Eigen::VectorXd &x_star,
                   const Eigen::VectorXd &u_star, 
                   const DynamicsFunc &dynamics_func, 
                   const CostFunc &cost_func, 
                   const double probability) :
    iLQRNode(x_star.size(), u_star.size(), dynamics_func, cost_func, probability)
{
    orig_xstar_ = x_star;
    orig_ustar_ = u_star;
    update_expansion(x_star, u_star);
}

void iLQRNode::update_expansion(const Eigen::VectorXd &x, const Eigen::VectorXd &u)
{
    expansion_.x = x;
    expansion_.u = u;
    ilqr::update_dynamics(dynamics_func_, expansion_);
    ilqr::update_cost(cost_func_, expansion_);
}

void iLQRNode::bellman_backup(const std::vector<std::shared_ptr<iLQRNode>> &children)
{
    // Compute the expected future value function terms weighted by the probability
    // of each transition.
    const int state_dim = x().size();
    const int control_dim = u().size();

    QuadraticValue weighted_value(state_dim);
    Eigen::MatrixXd weighted_inv_term = Eigen::MatrixXd::Zero(control_dim, control_dim);
    Eigen::MatrixXd weighted_Kt_term = Eigen::MatrixXd::Zero(control_dim, state_dim);
    Eigen::VectorXd weighted_kt_term = Eigen::VectorXd::Zero(control_dim);

    const size_t num_children = children.size();
    for (size_t i = 0; i < num_children; ++i)
    {
        const auto &child = children[i];
        // Dynamics come from child node.
        const Eigen::MatrixXd &A = child->expansion().dynamics.A; 
        const Eigen::MatrixXd &B = child->expansion().dynamics.B;
        const Eigen::MatrixXd &Vt1 = child->value().V();
        const Eigen::MatrixXd &Gt1 = child->value().G();
        const double p = child->probability();
        weighted_inv_term.noalias() += p * (B.transpose()*Vt1*B);
        weighted_Kt_term.noalias()  += p * (B.transpose()*Vt1*A);
        weighted_kt_term.noalias()  += p * (B.transpose()*Gt1.transpose());

    }

    // Cost terms come from parent (this) node.
    const Eigen::MatrixXd &Q = expansion_.cost.Q;        
    const Eigen::MatrixXd &P = expansion_.cost.P;
    const Eigen::MatrixXd &R = expansion_.cost.R;
    const Eigen::VectorXd &g_u = expansion_.cost.g_u;
    const Eigen::VectorXd &g_x = expansion_.cost.g_x;
    const double &c= expansion_.cost.c;

    const Eigen::MatrixXd inv_term = -1.0*(R + weighted_inv_term).inverse();
    const Eigen::MatrixXd Kt = inv_term * (P.transpose() + weighted_Kt_term); 
    const Eigen::VectorXd kt = inv_term * (g_u + weighted_kt_term);

    // Compute the value function of the parent using the control policy.
    QuadraticValue weighted_child_Jt1(state_dim);
    for (size_t i = 0; i < num_children; ++i)
    {
        QuadraticValue Jt1(state_dim);
        const auto &child = children[i];
        const Eigen::MatrixXd &A = child->expansion().dynamics.A; 
        const Eigen::MatrixXd &B = child->expansion().dynamics.B;
        const Eigen::MatrixXd tmp = (A + B*Kt);

        const Eigen::MatrixXd &Vt1 = child->value().V();
        const Eigen::MatrixXd &Gt1 = child->value().G();
        const double Wt1 = child->value().W();
        Jt1.V() = tmp.transpose()*Vt1*tmp;
        Jt1.G() = kt.transpose()*B.transpose()*Vt1*tmp + Gt1*tmp;
        const Eigen::VectorXd Wt_mat = Gt1*B*kt + 0.5*(kt.transpose()*B.transpose()*Vt1*B*kt);
        IS_EQUAL(Wt_mat.size(), 1);
        Jt1.W() = Wt_mat(0) + Wt1;;
        add_weighted_value(Jt1, child->probability(), weighted_child_Jt1);
    }

    J_.V() = Q + 2.0*(P*Kt) + Kt.transpose()*R*Kt + weighted_child_Jt1.V();
    J_.G() = kt.transpose()*P.transpose() + kt.transpose()*R*Kt + g_x.transpose() 
        + g_u.transpose()*Kt  + weighted_child_Jt1.G();
    const Eigen::VectorXd Wt_mat = 0.5*(kt.transpose()*R*kt) + g_u.transpose()*kt;
    IS_EQUAL(Wt_mat.size(), 1);
    J_.W() = Wt_mat(0) + c  + weighted_child_Jt1.W();

    K_ = Kt;
    k_ = kt;

    //ilqr::compute_backup(expansion_, children[0]->value(), K_, k_, J_);
}

std::ostream& operator<<(std::ostream& os, const iLQRNode& node)
{
    os << "x: [" << node.x().transpose() << "], u: [" << node.u().transpose() << "]";
    return os;
}

} // namespace ilqr
