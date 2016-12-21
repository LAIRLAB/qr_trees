
#include <experiments/simulators/pendulum.hh>
#include <templated/iLQR.hh>
#include <utils/math_utils_temp.hh>
#include <utils/debug_utils.hh>

#include <algorithm>
#include <memory>
#include <random>

namespace 
{

constexpr int STATE_DIM = simulators::pendulum::STATE_DIM;
constexpr int CONTROL_DIM = simulators::pendulum::CONTROL_DIM;


template <int rows>
using Vector = Eigen::Matrix<double, rows, 1>;

template <int rows, int cols>
using Matrix = Eigen::Matrix<double, rows, cols>;



Matrix<STATE_DIM, STATE_DIM> Q;
Matrix<STATE_DIM, STATE_DIM> QT; // Quadratic state cost for final timestep.
Matrix<CONTROL_DIM, CONTROL_DIM> R;
Vector<STATE_DIM> xT; // Goal state for final timestep.

double ct(const Vector<STATE_DIM> &x, const Vector<CONTROL_DIM> &u)
{
    const Vector<STATE_DIM> dx = x - xT;
    return 0.5*(dx.transpose()*Q*dx + u.transpose()*R*u)[0];
}

// Final timestep cost function
double cT(const Vector<STATE_DIM> &x, const Vector<CONTROL_DIM> &u)
{
    const Vector<STATE_DIM> dx = x - xT;
    return 0.5*(dx.transpose()*Q*dx)[0];
}

} // namespace


void control_pendulum(const int T, const double dt, const Vector<STATE_DIM> &x0 = Vector<STATE_DIM>::Zero())
{

    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    // Constants for the pendulum.
    constexpr double LENGTH = 1.0;
    constexpr double DAMPING_COEFF = 0.1;

    simulators::pendulum::Pendulum dynamics(dt, DAMPING_COEFF, LENGTH);

    xT << M_PI, 0;

    using State = simulators::pendulum::State;
    Q = Matrix<STATE_DIM,STATE_DIM>::Identity();
    Q(State::THETADOT,State::THETADOT) = 1e-5;

    QT = 3.*Matrix<STATE_DIM,STATE_DIM>::Identity();

    R = 1e-2*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();

    // Initial linearization points are linearly interpolated states and zero
    // control.
    Vector<CONTROL_DIM> u_nominal = Vector<CONTROL_DIM>::Ones();
    if (xT[State::THETA] < x0[State::THETA]) 
    {
        u_nominal *=-1;
    }

    constexpr bool verbose = true;
    constexpr int max_iters = 300;
    constexpr double mu = 0.01;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 10;

    clock_t ilqr_begin_time = clock();

    ilqr::iLQRSolver<STATE_DIM,CONTROL_DIM> solver(dynamics, cT, ct);
    solver.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    std::vector<Vector<STATE_DIM>> ilqr_states; 
    std::vector<Vector<CONTROL_DIM>> ilqr_controls;
    const double ilqr_total_cost = solver.forward_pass(x0, ilqr_states, ilqr_controls, 1.0);
    SUCCESS("iLQR (mu=" << mu << ") Time: " 
            << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC
            << "\nTotal Cost: " << ilqr_total_cost);

    // Run the control policy.
    constexpr double TOL =1e-4;
    Vector<STATE_DIM> xt = x0;
    double rollout_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        IS_TRUE(math::is_equal(ilqr_states[t], xt, TOL));

        const Vector<CONTROL_DIM> ut = solver.compute_control_stepsize(xt, t, 1.0); 

        IS_TRUE(math::is_equal(ilqr_controls[t], ut, TOL));

        rollout_cost += ct(xt, ut);

        const Vector<STATE_DIM> xt1 = dynamics(xt, ut);

        xt = xt1;
    }
    rollout_cost += cT(xt, Vector<CONTROL_DIM>::Zero());
    DEBUG(" x_rollout(" << T-1 << ")= " << xt.transpose());
    DEBUG(" Total cost rollout: " << rollout_cost);
    IS_ALMOST_EQUAL(ilqr_total_cost, rollout_cost, TOL);
}

int main()
{
    control_pendulum(100, 0.1);

    return 0;
}
