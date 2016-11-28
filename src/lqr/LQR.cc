// 
// Implements traditional LQR for linear dynamics and cost.
//

#include <lqr/LQR.hh>

#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>

namespace
{
  void check_lqr_matrix_sizes(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, 
          const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
  {
     const int state_dim = A.rows();
     const int control_dim = B.cols();
     IS_EQUAL(A.cols(), state_dim);
     IS_EQUAL(B.rows(), state_dim);
     IS_EQUAL(Q.rows(), state_dim);
     IS_EQUAL(Q.cols(), state_dim);
     IS_EQUAL(R.rows(), control_dim);
     IS_EQUAL(R.cols(), control_dim);

     math::check_psd(Q, 0.);
     math::check_psd(R, 1e-8);
  }
}

namespace lqr 
{

void compute_backup( 
        const Eigen::MatrixXd &A,
        const Eigen::MatrixXd &B,
        const Eigen::MatrixXd &Q,
        const Eigen::MatrixXd &R,
        const Eigen::MatrixXd &Vt1,
        Eigen::MatrixXd &Kt,
        Eigen::MatrixXd &Vt
        )
{
    const int state_dim = A.rows();
    const int control_dim = B.cols();

    Eigen::MatrixXd inv_term = -1.0*(R + B.transpose()*Vt1*B).inverse();
    Kt = inv_term*(B.transpose()*Vt1*A);
    IS_EQUAL(Kt.rows(), control_dim);
    IS_EQUAL(Kt.cols(), state_dim);

    const Eigen::MatrixXd tmp = (A + B*Kt);
    Vt = Q + Kt.transpose()* R * Kt + tmp.transpose()*Vt1*tmp;
}

LQR::LQR(const std::vector<Eigen::MatrixXd> &As, 
         const std::vector<Eigen::MatrixXd> &Bs,  
         const std::vector<Eigen::MatrixXd> &Qs, 
         const std::vector<Eigen::MatrixXd> &Rs 
        )
    : As_(As), Bs_(Bs), Qs_(Qs), Rs_(Rs)
{
    IS_GREATER(As_.size(), 0);
    state_dim_ = As_[0].rows();
    control_dim_ = Bs_[0].cols();

    const int T = As_.size();
    IS_EQUAL(Bs_.size(), T);
    IS_EQUAL(Qs_.size(), T);
    IS_EQUAL(Rs_.size(), T);
    for (int t = 0; t < T; ++t)
    {
        check_lqr_matrix_sizes(As_[t], Bs_[t], Qs_[t], Rs_[t]);
    }
}

LQR::LQR(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, 
        const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, 
        const int T)
{
    state_dim_ = A.rows();
    control_dim_ = B.cols();
    T_ = T;

    check_lqr_matrix_sizes(A, B, Q, R);
    As_ = std::vector<Eigen::MatrixXd>(T, A);
    Bs_ = std::vector<Eigen::MatrixXd>(T, B);
    Qs_ = std::vector<Eigen::MatrixXd>(T, Q);
    Rs_ = std::vector<Eigen::MatrixXd>(T, R);
}

void LQR::solve()
{
    Ks_.clear(); Ks_.resize(T_);

    Eigen::MatrixXd Vt1 = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
    for (int t = T_-1; t >= 0; t--)
    {
        Eigen::MatrixXd Vt;
        lqr::compute_backup(As_[t], Bs_[t], Qs_[t], Rs_[t], Vt1, Ks_[t], Vt);
        Vt1 = Vt;
    }

}

void LQR::forward_pass(const Eigen::VectorXd &x0, 
        std::vector<double> &costs,
        std::vector<Eigen::VectorXd> &states, 
        std::vector<Eigen::VectorXd> &controls) const
{
    states.clear();
    controls.clear();
    costs.clear();
    states.reserve(T_);
    controls.reserve(T_);
    costs.reserve(T_);

    Eigen::VectorXd xt = x0;
    Eigen::VectorXd ut = Eigen::VectorXd::Zero(control_dim_);
    for (int t = 0; t < T_; ++t)
    {
        const Eigen::MatrixXd &Kt = Ks_[t];
        ut = Kt * xt;
        const Eigen::VectorXd cost_mat = (xt.transpose()*Qs_[t]*xt)
            + (ut.transpose()*Rs_[t]*ut);
        IS_EQUAL(cost_mat.size(), 1)
        double cost = 0.5*cost_mat[0];

        states.push_back(xt); 
        controls.push_back(ut);
        costs.push_back(cost);

        IS_EQUAL(xt.rows(), state_dim_);
        xt = As_[t] * xt + Bs_[t] * ut;
    }
    IS_EQUAL(states.size(), T_);
    IS_EQUAL(controls.size(), T_);
    IS_EQUAL(costs.size(), T_);
}

} // namespace lqr
