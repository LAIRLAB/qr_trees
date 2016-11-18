// 
// Implements traditional LQR for linear dynamics and cost.
//

#pragma once

#include <vector>

#include <Eigen/Dense>

namespace lqr
{
void compute_backup(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
        const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
        const Eigen::MatrixXd &Vt1,
        Eigen::MatrixXd &Kt, Eigen::MatrixXd &Vt);

class LQR
{
public:

    LQR(const std::vector<Eigen::MatrixXd> &As, 
        const std::vector<Eigen::MatrixXd> &Bs,
        const std::vector<Eigen::MatrixXd> &Qs,
        const std::vector<Eigen::MatrixXd> &Rs);

    LQR(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
        const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, 
        const int T);

    void solve();
    void solve(std::vector<Eigen::MatrixXd> &Vs);

    void forward_pass(const Eigen::VectorXd &x0, 
        std::vector<double> &costs,
        std::vector<Eigen::VectorXd> &states, 
        std::vector<Eigen::VectorXd> &controls) const;

//private:
    int state_dim_ = -1;
    int control_dim_  = -1;

    std::vector<Eigen::MatrixXd> As_; 
    std::vector<Eigen::MatrixXd> Bs_;
    std::vector<Eigen::MatrixXd> Qs_; 
    std::vector<Eigen::MatrixXd> Rs_;

    std::vector<Eigen::MatrixXd> Ks_;
};

} // namespace lqr

