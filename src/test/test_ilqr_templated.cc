
#include <templated/iLQR.hh>
#include <templated/iLQR_hindsight.hh>
#include <templated/taylor_expansion.hh>

#include <lqr/LQR.hh>
#include <ilqr/iLQR.hh>

#include <utils/debug_utils.hh>

#include <ctime>
#include <numeric>
#include <ostream>

namespace
{
    constexpr int xdim = 3;
    constexpr int udim = 2;

    // Should be higher as the cost convg. threshold for iLQRSolver.
    constexpr double TOL = 1e-5;

    template<int _rows>
    using Vector = ilqr::Vector<_rows>;

    template<int _rows, int _cols>
    using Matrix = ilqr::Matrix<_rows, _cols>;

    Matrix<xdim, xdim> A = Matrix<xdim, xdim>::Identity();
    Matrix<xdim, udim> B = 2*Matrix<xdim, udim>::Identity();

    Matrix<xdim, xdim> Q = 5*Matrix<xdim, xdim>::Identity();
    Matrix<udim, udim> R = 2*Matrix<udim, udim>::Identity();

    inline Vector<xdim> linear_dynamics(const Vector<xdim> &x, const Vector<udim> &u)
    {
        return A*x + B*u;
    }

    inline double quadratic_cost_time(const Vector<xdim> &x, const Vector<udim> &u, const int t)
    {
        const Vector<1> c = 0.5*(x.transpose()*Q*x + u.transpose()*R*u);
        return c[0];
    }

    inline double quadratic_cost(const Vector<xdim> &x, const Vector<udim> &u)
    {
        return quadratic_cost_time(x, u, 0);
    }

    inline double zero_cost(const Vector<xdim> &x)
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

void test_ilqr_vs_lqr(const int T)
{
    DEBUG("Testing T=" << T);

    const auto dynamics = linear_dynamics;
    const auto final_cost = zero_cost;
    Vector<udim> u_nominal; u_nominal.setOnes();
    Vector<xdim> x_init; x_init.setOnes();


    constexpr bool verbose = false;
    constexpr int max_iters = 300;

    double mu = 0.0;

    clock_t ilqr_begin_time = clock();
    ilqr::iLQRSolver<xdim,udim> solver(dynamics, final_cost, quadratic_cost_time);
    solver.solve(T, x_init, u_nominal, mu, max_iters, verbose);
    std::vector<Vector<xdim>> ilqr_temp_states; 
    std::vector<Vector<udim>> ilqr_temp_controls;
    const double ilqr_temp_total_cost = solver.forward_pass(x_init, ilqr_temp_states, 
            ilqr_temp_controls, 1.0);
    SUCCESS("iLQR Templated (mu=" << mu << ") Time: " 
            << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC
            << "\n\tTotal Cost: " << ilqr_temp_total_cost);

    // With a higher mu (Levenberg-Marquardt parameter), it should converge slower, but still
    // converge with almost the same value (it is slightly off due to the damping).
    mu = 0.5;
    ilqr_begin_time = clock();
    solver.solve(T, x_init, u_nominal, mu, max_iters, verbose);
    const double ilqr_temp_higher_mu_total_cost = solver.forward_pass(x_init, ilqr_temp_states, 
            ilqr_temp_controls, 1.0);
    SUCCESS("iLQR Templated (mu=" << mu << ") Time: " 
            << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC
            << "\n\tTotal Cost: " << ilqr_temp_higher_mu_total_cost);

    mu = 0.0;
    std::vector<Vector<xdim>> ilqr_hind_states; 
    std::vector<Vector<udim>> ilqr_hind_controls;
    ilqr::HindsightSplit<xdim,udim> split(dynamics, final_cost, quadratic_cost_time, 1.0);
    clock_t ilqr_hindsight_begin_time = clock();
    ilqr::iLQRHindsightSolver<xdim,udim> hindsight_solver({split});
    hindsight_solver.solve(T, x_init, u_nominal, mu, max_iters, verbose);
    const double ilqr_hind_total_cost = hindsight_solver.forward_pass(0, x_init, ilqr_hind_states, 
            ilqr_hind_controls, 1.0);
    SUCCESS("iLQR Hindsight (mu=" << mu << ") Time: " 
            << (clock() - ilqr_hindsight_begin_time) / (double) CLOCKS_PER_SEC
            << "\n\tTotal Cost: " << ilqr_hind_total_cost);

    clock_t lqr_begin_time = clock();
    lqr::LQR regular_lqr(A, B, Q, R, T);
    regular_lqr.solve();
    std::vector<double> lqr_costs;
    std::vector<Eigen::VectorXd> lqr_states; 
    std::vector<Eigen::VectorXd> lqr_controls;
    regular_lqr.forward_pass(x_init, lqr_costs, lqr_states, lqr_controls);
    const double lqr_total_cost = std::accumulate(lqr_costs.begin(), lqr_costs.end(), 0.0);
    SUCCESS("LQR Time: " << (clock() - lqr_begin_time) / (double) CLOCKS_PER_SEC
            << "\n\tTotal Cost: " << lqr_total_cost);

    std::vector<double> ilqr_dyn_costs;
    std::vector<Eigen::VectorXd> ilqr_dyn_states; 
    std::vector<Eigen::VectorXd> ilqr_dyn_controls;
    clock_t ilqr_dyn_begin_time = clock();
    ilqr::iLQR ilqr_dynamic(dynamics, quadratic_cost, std::vector<Eigen::VectorXd>(T, x_init), 
            std::vector<Eigen::VectorXd>(T, u_nominal));
    // For fair timing, do 2 passes since templatized does that since it checks for convergence
    ilqr_dynamic.backwards_pass();
    ilqr_dynamic.forward_pass(ilqr_dyn_costs, ilqr_dyn_states, ilqr_dyn_controls, true);
    ilqr_dynamic.backwards_pass();
    ilqr_dynamic.forward_pass(ilqr_dyn_costs, ilqr_dyn_states, ilqr_dyn_controls, true);
    const double ilqr_dyn_total_cost = std::accumulate(ilqr_dyn_costs.begin(), ilqr_dyn_costs.end(), 0.0);
    SUCCESS("iLQR Dynamic Time: " << (clock() - ilqr_dyn_begin_time) / (double) CLOCKS_PER_SEC
            << "\n\tTotal Cost: " << ilqr_dyn_total_cost);

    // Make sure all the costs are about equal.
    IS_ALMOST_EQUAL(ilqr_dyn_total_cost, lqr_total_cost, TOL);
    IS_ALMOST_EQUAL(ilqr_temp_total_cost, ilqr_dyn_total_cost, TOL);
    IS_ALMOST_EQUAL(ilqr_temp_total_cost, lqr_total_cost, TOL);
    IS_ALMOST_EQUAL(ilqr_temp_total_cost, ilqr_temp_higher_mu_total_cost, TOL);
    IS_ALMOST_EQUAL(ilqr_hind_total_cost, ilqr_temp_total_cost, TOL);
    IS_ALMOST_EQUAL(ilqr_hind_total_cost, lqr_total_cost, TOL);
    IS_ALMOST_EQUAL(ilqr_hind_total_cost, ilqr_dyn_total_cost, TOL);

    DEBUG("All methods' costs are almost equal.");
}

int main()
{
    test_ilqr_vs_lqr(2);
    test_ilqr_vs_lqr(3);
    test_ilqr_vs_lqr(10);
    test_ilqr_vs_lqr(100);

    return 0;
}
