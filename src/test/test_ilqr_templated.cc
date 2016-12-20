
#include <templated/iLQR.hh>
#include <templated/taylor_expansion.hh>

#include <lqr/LQR.hh>
#include <ilqr/iLQR.hh>

#include <utils/debug_utils.hh>

#include <ctime>
#include <ostream>

namespace
{
    template<int _rows>
    using Vector = ilqr::Vector<_rows>;

    template<int _rows, int _cols>
    using Matrix = ilqr::Matrix<_rows, _cols>;

    constexpr int xdim = 3;
    constexpr int udim = 2;

    Matrix<xdim, xdim> A = Matrix<xdim, xdim>::Identity();
    Matrix<xdim, udim> B = 2*Matrix<xdim, udim>::Identity();

    Matrix<xdim, xdim> Q = 5*Matrix<xdim, xdim>::Identity();
    Matrix<udim, udim> R = 2*Matrix<udim, udim>::Identity();

    inline Vector<xdim> linear_dynamics(const Vector<xdim> &x, const Vector<udim> &u)
    {
        return A*x + B*u;
    }

    inline double quadratic_cost(const Vector<xdim> &x, const Vector<udim> &u)
    {
        const Vector<1> c = 0.5*(x.transpose()*Q*x + u.transpose()*R*u);
        return c[0];
    }

    inline double zero_cost(const Vector<xdim> &x, const Vector<udim> &u)
    {
        return 0;
    }

    std::ostream& operator<<(std::ostream &os, const std::vector<Eigen::VectorXd> &vectors)
    {
        for (size_t t = 0; t < vectors.size(); ++t)
        {
            const Eigen::VectorXd &vec = vectors[t];
            os << "t=" << t  << ": " << vec.transpose(); 
            if (t < vectors.size() -1)
            {
                os << std::endl;
            }
        }
        return os;
    }
}

int main()
{

    constexpr int T = 3;
    const auto dynamics = linear_dynamics;
    const auto cost = quadratic_cost;
    const auto final_cost = zero_cost;
    Vector<udim> u_nominal; u_nominal.setOnes();
    Vector<xdim> x_init; x_init.setOnes();


    clock_t ilqr_begin_time = clock();
    ilqr::iLQRSolver<xdim,udim> solver(T, dynamics, final_cost, cost);
    solver.solve(x_init, u_nominal, true);
    SUCCESS("iLQR Templated Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

    clock_t lqr_begin_time = clock();
    lqr::LQR regular_lqr(A, B, Q, R, T);
    regular_lqr.solve();
    SUCCESS("LQR Time: " << (clock() - lqr_begin_time) / (double) CLOCKS_PER_SEC);

    std::vector<double> ilqr_dyn_costs;
    std::vector<Eigen::VectorXd> ilqr_dyn_states; 
    std::vector<Eigen::VectorXd> ilqr_dyn_controls;
    clock_t ilqr_dyn_begin_time = clock();
    ilqr::iLQR ilqr_dynamic(dynamics, cost, std::vector<Eigen::VectorXd>(T, x_init), 
            std::vector<Eigen::VectorXd>(T, u_nominal));
    // For fair timing, do 2 passes since templatized does that since it checks for convergence
    ilqr_dynamic.backwards_pass();
    ilqr_dynamic.forward_pass(ilqr_dyn_costs, ilqr_dyn_states, ilqr_dyn_controls, true);
    ilqr_dynamic.backwards_pass();
    ilqr_dynamic.forward_pass(ilqr_dyn_costs, ilqr_dyn_states, ilqr_dyn_controls, true);
    const double ilqr_dyn_total_cost = std::accumulate(ilqr_dyn_costs.begin(), ilqr_dyn_costs.end(), 0.0);
    SUCCESS("iLQR Dynamic Time: " << (clock() - ilqr_dyn_begin_time) / (double) CLOCKS_PER_SEC);
    WARN("ilqr_dynamic cost: " << ilqr_dyn_total_cost);

    std::vector<double> lqr_costs;
    std::vector<Eigen::VectorXd> lqr_states; 
    std::vector<Eigen::VectorXd> lqr_controls;
    regular_lqr.forward_pass(x_init, lqr_costs, lqr_states, lqr_controls);
    const double lqr_total_cost = std::accumulate(lqr_costs.begin(), lqr_costs.end(), 0.0);
    WARN("lqr_cost: " << lqr_total_cost);
    WARN("lqr_controls:\n " << lqr_controls);

    return 0;
}
