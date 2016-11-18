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
    Kt = (R + B.transpose()*Vt1*B).inverse() * (B.transpose()*Vt1*A);
    Kt *= -1.0;
    IS_EQUAL(Kt.rows(), control_dim);
    IS_EQUAL(Kt.cols(), state_dim);

    const auto tmp = (A + B*Kt);
    Vt = Q + Kt.transpose()*R*Kt + tmp.transpose()*Vt1*tmp;
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

    check_lqr_matrix_sizes(A, B, Q, R);
    As_ = std::vector<Eigen::MatrixXd>(T, A);
    Bs_ = std::vector<Eigen::MatrixXd>(T, B);
    Qs_ = std::vector<Eigen::MatrixXd>(T, Q);
    Rs_ = std::vector<Eigen::MatrixXd>(T, R);
}

void LQR::solve()
{
    std::vector<Eigen::MatrixXd> tmp;
    solve(tmp);
}

void LQR::solve(std::vector<Eigen::MatrixXd> &Vs)
{
    const int T = As_.size();
    Vs.clear(); Vs.resize(T);
    Ks_.clear(); Ks_.resize(T);

    Eigen::MatrixXd Vt1 = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
    for (int t = T-1; t >= 0; t--)
    {
        const Eigen::MatrixXd &A = As_[t];
        const Eigen::MatrixXd &B = Bs_[t];
        const Eigen::MatrixXd &R = Rs_[t];
        const Eigen::MatrixXd &Q = Qs_[t];
        lqr::compute_backup(A, B, Q, R, Vt1, Ks_[t], Vs[t]);
        Vt1 = Vs[t];
    }

}

void LQR::forward_pass(const Eigen::VectorXd &x0, 
        std::vector<double> &costs,
        std::vector<Eigen::VectorXd> &states, 
        std::vector<Eigen::VectorXd> &controls) const
{
    const int T = As_.size();
    states.reserve(T);
    controls.reserve(T);
    costs.reserve(T);

    Eigen::VectorXd xt = x0;
    Eigen::VectorXd ut = Eigen::VectorXd::Zero(control_dim_);
    for (int t = 0; t < T; ++t)
    {
        const Eigen::MatrixXd &Kt = Ks_[t];
        ut = Kt * xt;
        const Eigen::VectorXd cost = 0.5*(xt.transpose()*Qs_[t]*xt 
            + ut.transpose()*Rs_[t]*ut);
        IS_EQUAL(cost.size(), 1)

        states.push_back(xt); 
        controls.push_back(ut);
        costs.push_back(cost[0]);

        IS_EQUAL(xt.rows(), state_dim_);
        xt = As_[t] * xt + Bs_[t] * ut;
    }
    IS_EQUAL(states.size(), T);
    IS_EQUAL(controls.size(), T);
    IS_EQUAL(costs.size(), T);
}

} // namespace lqr
