
#include <experiments/simulators/diffdrive.hh>
#include <experiments/simulators/circle_world.hh>
#include <templated/iLQR.hh>
#include <utils/math_utils_temp.hh>
#include <utils/debug_utils.hh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace 
{

using CircleWorld = circle_world::CircleWorld;
using Circle = circle_world::Circle;

using State = simulators::diffdrive::State;
constexpr int STATE_DIM = simulators::diffdrive::STATE_DIM;
constexpr int CONTROL_DIM = simulators::diffdrive::CONTROL_DIM;
//constexpr int OBS_DIM = circle_world::OBSTACLE_DIM;

template <int rows>
using Vector = Eigen::Matrix<double, rows, 1>;

template <int rows, int cols>
using Matrix = Eigen::Matrix<double, rows, cols>;

double robot_radius = 3.35/2.0; // iRobot create;
double obstacle_factor = 30.0;
double scale_factor = 1.0e0;

Matrix<STATE_DIM, STATE_DIM> Q;
Matrix<STATE_DIM, STATE_DIM> QT; // Quadratic state cost for final timestep.
Matrix<CONTROL_DIM, CONTROL_DIM> R;
Vector<STATE_DIM> xT; // Goal state for final timestep.
Vector<STATE_DIM> x0; // Start state for 0th timestep.
Vector<CONTROL_DIM> u_nominal; 

int T;

double obstacle_cost(const CircleWorld &world, const double robot_radius, const Vector<STATE_DIM> &xt)
{
    Eigen::Vector2d robot_pos;
    robot_pos << xt[State::POS_X], xt[State::POS_Y];

    // Compute minimum distance to the edges of the world.
    double cost = 0;

    auto obstacles = world.obstacles();
	for (size_t i = 0; i < obstacles.size(); ++i) {
		Vector<2> d = robot_pos - obstacles[i].position();
		double distr = d.norm(); 
		double dist = distr - robot_radius - obstacles[i].radius();
		cost += obstacle_factor * exp(-scale_factor*dist);
	}
    return cost;
}

double ct(const Vector<STATE_DIM> &x, const Vector<CONTROL_DIM> &u, const int t, const CircleWorld &world)
{
    double cost = 0;

    // position
    if (t == 0)
    {
        Vector<STATE_DIM> dx = x - x0;
        cost += 0.5*(dx.transpose()*Q*dx)[0];
    }

    // Control cost
    const Vector<CONTROL_DIM> du = u - u_nominal;
    cost += 0.5*(du.transpose()*R*du)[0];

    cost += 1.0*obstacle_cost(world, robot_radius, x);

    return cost;
}

// Final timestep cost function
double cT(const Vector<STATE_DIM> &x)
{
    const Vector<STATE_DIM> dx = x - xT;
    return 0.5*(dx.transpose()*QT*dx)[0];
}

void states_to_file(const Vector<STATE_DIM>& x0, const Vector<STATE_DIM>& xT, 
        const std::vector<Vector<STATE_DIM>> &states, 
        const std::string &fname)
{
    std::ofstream file(fname, std::ofstream::trunc | std::ofstream::out);
    auto print_vector = [&file](const Vector<STATE_DIM> &x)
    {
        constexpr int PRINT_WIDTH = 13;
        constexpr char DELIMITER[] = " ";

        for (int i = 0; i < STATE_DIM; ++i)
        {
            file << std::left << std::setw(PRINT_WIDTH) << x[i] << DELIMITER;
        }
        file << std::endl;
    };
    print_vector(x0); 
    print_vector(xT);
    for (const auto& state : states)
    {
        print_vector(state);
    }
    file.close();
}

void obstacles_to_file(const CircleWorld &world, const std::string &fname)
{
    std::ofstream file(fname, std::ofstream::trunc | std::ofstream::out);
    IS_TRUE(file.is_open());
    file << world;
    file.close();
}


} // namespace


void control_diffdrive(const std::string &states_fname, const std::string &obstacles_fname)
{
    using namespace std::placeholders;

    T = 150;
	const double dt = 1.0/6.0;
    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    CircleWorld world(-30, 30, -30, 30);
    Eigen::Vector2d obstacle_pos(0, 0.0);
	constexpr double obs_radius = 5.0;
    world.add_obstacle(obs_radius, obstacle_pos);

    //world.add_obstacle(obs_radius, Eigen::Vector2d(-13, -13));
    //world.add_obstacle(obs_radius, Eigen::Vector2d(-10, 3));

	xT = Vector<STATE_DIM>::Zero();
	xT[State::POS_X] = 0;
	xT[State::POS_Y] = 25;
	xT[State::THETA] = M_PI/2; 

	x0 = Vector<STATE_DIM>::Zero();
	x0[State::POS_X] = 0;
	x0[State::POS_Y] = -25;
	x0[State::THETA] = M_PI/2;

	Q = 1*Matrix<STATE_DIM,STATE_DIM>::Identity();
	const double rot_cost = 0.5;
    Q(State::THETA, State::THETA) = rot_cost;

    QT = 15*Matrix<STATE_DIM,STATE_DIM>::Identity();
    QT(State::THETA, State::THETA) = 150.0;

	R = 2*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();

    // Initial linearization points are linearly interpolated states and zero
    // control.
    u_nominal[0] = 1.5;
    u_nominal[1] = 1.5;

    std::array<double, 2> control_lims = {{-5, 5}};

    simulators::diffdrive::DiffDrive system(dt, control_lims, world.dimensions());
    auto cost_t = std::bind(ct, _1, _2, _3, world);

    auto dynamics = system;

    constexpr bool verbose = true;
    constexpr int max_iters = 300;
    constexpr double mu = 0.80;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 1;


    clock_t ilqr_begin_time = clock();

    ilqr::iLQRSolver<STATE_DIM,CONTROL_DIM> solver(dynamics, cT, cost_t);
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
    std::vector<Vector<STATE_DIM>> states;
    states.push_back(xt);
    double rollout_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        IS_TRUE(math::is_equal(ilqr_states[t], xt, TOL));

        const Vector<CONTROL_DIM> ut = solver.compute_control_stepsize(xt, t, 1.0); 

        IS_TRUE(math::is_equal(ilqr_controls[t], ut, TOL));

        rollout_cost += cost_t(xt, ut, t);

        const Vector<STATE_DIM> xt1 = dynamics(xt, ut);

        xt = xt1;
        states.push_back(xt);
    }
    rollout_cost += cT(xt);
    DEBUG(" x_rollout(" << T-1 << ")= " << xt.transpose());
    DEBUG(" Total cost rollout: " << rollout_cost);

    states_to_file(x0, xT, states, states_fname);
    obstacles_to_file(world, obstacles_fname);
}

int main()
{
    control_diffdrive("states.csv", "obstacles.csv");

    return 0;
}
