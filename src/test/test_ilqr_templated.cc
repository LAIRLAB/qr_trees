#include <test/test_ilqr_templated.hh>
#include <ilqr/ilqr_taylor_expansions.cc>

#include <utils/debug_utils.hh>


int main()
{
    constexpr int xdim = 3;
    constexpr int udim = 2;

    constexpr int T = 10;
    auto dynamics = linear_dynamics<xdim,udim>;
    auto cost = quadratic_cost<xdim,udim>;
    auto final_cost = cost;
    Vector<udim> u_nominal; u_nominal.setOnes();
    Vector<xdim> x_init; x_init.setOnes();

    ilqr::iLQR<xdim,udim> solver(T, dynamics, final_cost, cost, u_nominal, x_init);
    solver.solve();

    return 0;
}
