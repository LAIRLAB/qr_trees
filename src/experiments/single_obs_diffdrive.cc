
#include <experiments/simulators/diffdrive.hh>
#include <experiments/simulators/circle_world.hh>
#include <templated/iLQR_hindsight.hh>
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
double obstacle_factor = 300.0;
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

double ct(const Vector<STATE_DIM> &x, const Vector<CONTROL_DIM> &u, const int t, const CircleWorld &world, const double obs_cost)
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

    cost += 10*x[State::dTHETA]*x[State::dTHETA];
    cost += 0.05*x[State::dV_LEFT]*x[State::dV_LEFT];
    cost += 0.05*x[State::dV_RIGHT]*x[State::dV_RIGHT];

    cost += obs_cost*obstacle_cost(world, robot_radius, x);

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

int get_argmax(const std::array<double, 2> &prob)
{
    if (prob[0] > prob[1])
    {
        return 0;
    }
    return 1;
}


} // namespace


void control_diffdrive(const std::string &states_fname, const std::string &obstacles_fname)
{
    using namespace std::placeholders;

    T = 150;
	const double dt = 1.0/6.0;
    IS_GREATER(T, 1);
    IS_GREATER(dt, 0);

    // Prior that there is an obstacle, prior there is no obstacle.
    const std::array<double, 2> OBS_PRIOR = {{0.10, 0.90}};

    std::array<double, 4> world_dims = {{-30, 30, -30, 30}};

    CircleWorld world_w_obs(world_dims);
    CircleWorld world_no_obs(world_dims);

    const Eigen::Vector2d obstacle_pos(0, 0.0);
	constexpr double obs_radius = 5.0;
    world_w_obs.add_obstacle(obs_radius, obstacle_pos);

	xT = Vector<STATE_DIM>::Zero();
	xT[State::POS_X] = 0;
	xT[State::POS_Y] = 25;
	xT[State::THETA] = M_PI/2; 
	xT[State::dTHETA] = 0;

	x0 = Vector<STATE_DIM>::Zero();
	x0[State::POS_X] = 0;
	x0[State::POS_Y] = -25;
	x0[State::THETA] = M_PI/2;
	x0[State::dTHETA] = 0;

	Q = 1*Matrix<STATE_DIM,STATE_DIM>::Identity();
	const double rot_cost = 0.5;
    Q(State::THETA, State::THETA) = rot_cost;
    Q(State::dV_LEFT, State::dV_LEFT) = 0.1;

    QT = 25*Matrix<STATE_DIM,STATE_DIM>::Identity();
    QT(State::THETA, State::THETA) = 50.0;
    QT(State::dTHETA, State::dTHETA) = 5.0;
    QT(State::dV_LEFT, State::dV_RIGHT) = 5.0;

	R = 2*Matrix<CONTROL_DIM,CONTROL_DIM>::Identity();

    // Initial linearization points are linearly interpolated states and zero
    // control.
    u_nominal[0] = 2.5;
    u_nominal[1] = 2.5;

    const std::array<double, 2> CONTROL_LIMS = {{-5, 5}};
    simulators::diffdrive::DiffDrive system(dt, CONTROL_LIMS, world_dims);

    auto dynamics = system;

    constexpr bool verbose = false;
    constexpr int max_iters = 300;
    constexpr double mu = 0.80;
    constexpr double convg_thresh = 1e-4;
    constexpr double start_alpha = 1;

    auto ct_with_obs = std::bind(ct, _1, _2, _3, world_w_obs, 1.0);
    auto ct_without_obs = std::bind(ct, _1, _2, _3, world_no_obs, 1.0);
    
    const auto &TRUE_COST_t = ct_with_obs;
    const auto &TRUE_WORLD = world_w_obs;

    std::array<double, 2> obs_probability= OBS_PRIOR;

    // Setup the true system solver.
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> true_branch(dynamics, cT, TRUE_COST_t, 1.0);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> true_chain_solver({true_branch});

    // Setup the "our method" hindsight optimization approach.
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> hindsight_w_obs(dynamics, cT, ct_with_obs, obs_probability[0]);
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> hindsight_without_obs(dynamics, cT, ct_without_obs, obs_probability[1]);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> hindsight_solver({hindsight_w_obs, hindsight_without_obs});

    // The argmax approach.
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> argmax_w_obs(dynamics, cT, ct_with_obs, obs_probability[0]);
    ilqr::HindsightBranch<STATE_DIM,CONTROL_DIM> argmax_without_obs(dynamics, cT, ct_without_obs, obs_probability[1]);
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> argmax_solver({argmax_w_obs, argmax_without_obs});
    const int argmax_branch = get_argmax(obs_probability);
    const int other_branch = (argmax_branch == 0) ? 1 : 0;
    argmax_solver.set_branch_probability(argmax_branch, 1.0);
    argmax_solver.set_branch_probability(other_branch, 0.0);


    // TODO Set the solver to use based on policy.
    //ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> &solver = true_chain_solver;
    //ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> &solver = argmax_solver;
    ilqr::iLQRHindsightSolver<STATE_DIM,CONTROL_DIM> &solver = hindsight_solver;

    std::vector<Vector<STATE_DIM>> states;

    clock_t ilqr_begin_time = clock();
    solver.solve(T, x0, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha);
    PRINT("Pre-solve" << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

    double rollout_cost = 0;
    Vector<STATE_DIM> xt = x0;
    // store initial state
    states.push_back(xt);
    //TODO: How to run this for full T?
    for (int t = 0; t < T-1; ++t)
    {
        const bool t_offset = t >  0 ? 1 : 0;
        const int plan_horizon = T-t;
        //const int plan_horizon = std::min(T-t, MPC_HORIZON);
        
        ilqr_begin_time = clock();
        solver.solve(plan_horizon, xt, u_nominal, mu, max_iters, verbose, convg_thresh, start_alpha, true, t_offset);
        PRINT("t=" << t << ": Compute Time: " << (clock() - ilqr_begin_time) / (double) CLOCKS_PER_SEC);

        const Vector<CONTROL_DIM> ut = solver.compute_first_control(xt); 

        rollout_cost += TRUE_COST_t(xt, ut, t);
        const Vector<STATE_DIM> xt1 = dynamics(xt, ut);

        xt = xt1;
        states.push_back(xt);

        const Eigen::Vector2d robot_position(xt[State::POS_X], xt[State::POS_Y]);
        const double net_distance = (robot_position - obstacle_pos).norm() 
                                    - robot_radius - obs_radius;
        if (net_distance < 2)
        {
            WARN("OBSERVED OBSTACLE!, Distance: " << net_distance);
            obs_probability[0] = 1.0;
            obs_probability[1] = 0.0;
        }
        hindsight_solver.set_branch_probability(0, obs_probability[0]);
        hindsight_solver.set_branch_probability(1, obs_probability[1]);

        const int argmax_branch = get_argmax(obs_probability);
        const int other_branch = (argmax_branch == 0) ? 1 : 0;
        argmax_solver.set_branch_probability(argmax_branch, 1.0);
        argmax_solver.set_branch_probability(other_branch, 0.0);
    }
    rollout_cost += cT(xt);
    DEBUG(" x_rollout(" << T-1 << ")= " << xt.transpose());
    DEBUG(" Total cost rollout: " << rollout_cost);

    states_to_file(x0, xT, states, states_fname);
    obstacles_to_file(TRUE_WORLD, obstacles_fname);
}

int main()
{
    control_diffdrive("states.csv", "obstacles.csv");

    return 0;
}
