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
     math::check_psd(R, 1e-5);
  }
}

namespace lqr 
{

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

    check_lqr_matrix_sizes(A, B, Q, R);
    As_ = std::vector<Eigen::MatrixXd>(T, A);
    Bs_ = std::vector<Eigen::MatrixXd>(T, B);
    Qs_ = std::vector<Eigen::MatrixXd>(T, Q);
    Rs_ = std::vector<Eigen::MatrixXd>(T, R);
}

void LQR::solve()
{
    const int T = As_.size();
    Ks_.clear(); Ks_.resize(T);
    //Eigen::MatrixXd Kt = Eigen::MatrixXd::Zero(control_dim_, state_dim_);
    //Ks.back() = Kt;
    //Eigen::MatrixXd Vt1 = Qs_.back(); 
    Eigen::MatrixXd Vt1 = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
    for (int t = T-1; t >= 0; t--)
    {
        const Eigen::MatrixXd &A = As_[t];
        const Eigen::MatrixXd &B = Bs_[t];
        const Eigen::MatrixXd &R = Rs_[t];
        Eigen::MatrixXd Kt = (R + B.transpose()*Vt1*B).inverse()
                                        * (B.transpose()*Vt1*A);
        Kt *= -1.0;
        IS_EQUAL(Kt.rows(), control_dim_);
        IS_EQUAL(Kt.cols(), state_dim_);
        Ks_[t] = Kt;

        const Eigen::MatrixXd &Q = Qs_[t];
        const auto tmp = (A + B*Kt);
        Vt1 = Q + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp;
    }
}

std::vector<lqr::StateCost> LQR::forward_pass(const Eigen::VectorXd &x0) const
{
    const int T = As_.size();
    std::vector<lqr::StateCost> states; 
    states.reserve(T);
    Eigen::VectorXd xt = x0;
    Eigen::VectorXd ut = Eigen::VectorXd::Zero(control_dim_);
    for (int t = 0; t < T; ++t)
    {
        const Eigen::MatrixXd &Kt = Ks_[t];
        ut = Kt * xt;
        const Eigen::VectorXd cost = xt.transpose()*Qs_[t]*xt 
            + ut.transpose()*Rs_[t]*ut;
        IS_EQUAL(cost.size(), 1)

        states.emplace_back(xt, ut, cost[0]);
        IS_EQUAL(xt.rows(), state_dim_);
        xt = As_[t] * xt + Bs_[t] * ut;
    }
    IS_EQUAL(states.size(), T);
    return states;
}

} // namespace lqr
