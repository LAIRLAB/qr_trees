
#include <experiments/simulators/circle_world.hh>
#include <experiments/simulators/simplecar.hh>
#include <filters/discrete_filter.hh>
#include <ilqr/ilqr_tree.hh>
#include <ilqr/ilqrtree_helpers.hh>
#include <ilqr/mpc_tree_policies.hh>
#include <utils/debug_utils.hh>
#include <utils/helpers.hh>
#include <utils/math_utils.hh>

#include <algorithm>
#include <fstream>
#include <memory>
#include <random>

namespace 
{

constexpr int STATE_DIM = simulators::simplecar::STATE_DIM;
constexpr int CONTROL_DIM = simulators::simplecar::CONTROL_DIM;

enum class PolicyTypes
{
    // Compute an iLQR policy from a tree that splits only at the first timestep.
    HINDSIGHT = 0,
    // Compute the iLQR chain policy under the true dynamics. This should be the best solution.
    TRUE_ILQR,
    // Compute iLQR chain using the argmax(probabilities_from_filter) dynamics.
    // This should perfectly when there is no noise in the observations.
    ARGMAX_ILQR,
    // Compute iLQR chain for each probabilistic split and take a weighted average of the controls.
    PROB_WEIGHTED_CONTROL,
};

std::string to_string(const PolicyTypes policy_type)
{
    switch(policy_type)
    {
        case PolicyTypes::HINDSIGHT:
            return "hindsight";
        case PolicyTypes::TRUE_ILQR:
            return "true_ilqr";
        case PolicyTypes::ARGMAX_ILQR:
            return "argmax_ilqr";
        case PolicyTypes::PROB_WEIGHTED_CONTROL:
            return "weighted_prob";
    };
    return "Unrecognized policy type. Error.";
}

double circle_world_cost(const CircleWorld &world, const Circle& robot)
{
    // We want to try to stay at least this far from obstacles, 
    // so we subtract this from the raw distance to get distance to buffer.
    constexpr double BUFFER_DIST = 0.5;

    std::vector<double> distances;
    world.distances(robot, distances);
    double cost = 0;
    for (double dist : distances)
    {
        double effective_dist = dist - BUFFER_DIST;
        double obstacle_cost = 0;
        if (effective_dist < 0)
        {
            obstacle_cost = -effective_dist;
        }
        cost += obstacle_cost;
    }
    return cost;
}

double goal_cost(const Circle& robot, const Circle &goal)
{
    // Try to keep this less than the obstacle cost so we don't go through obstacles.
    double distance;
    robot.distance(goal, distance);
    return distance;
}

ilqr::CostFunc make_cost_func(const CircleWorld &env, const double robot_radius, const Circle &goal, const std::vector<double> &cost_weights)
{
   // We have these many cost terms.
   IS_EQUAL(cost_weights.size(), 3);

   Eigen::MatrixXd R = Eigen::MatrixXd::Identity(CONTROL_DIM, CONTROL_DIM);
   R(0,0) = 1e-1;
   R(1,1) = 1e0;
   return [robot_radius, &env, goal, R, cost_weights](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
   {
        using State = simulators::simplecar::State;
        const Circle robot(robot_radius, x[State::POS_X], x[State::POS_Y]);
        const double c_world = circle_world_cost(env, robot);
        const double c_goal = goal_cost(robot, goal);
        const double c_control = u.transpose() * R * u;

        const double total_cost = cost_weights[0]*c_world 
                                + cost_weights[1]*c_goal 
                                + cost_weights[2]*c_control;
        return  total_cost;
    };
}


void print_to_csv(const std::vector<Eigen::VectorXd> &vecs, const std::string &fname)
{
    std::ofstream file(fname.c_str());
    IS_TRUE(file.is_open());
    for (const Eigen::VectorXd& vec : vecs)
    {
        file << vec.transpose() << std::endl;
    }
    file.close();
}


} // namespace


void single_obstacle_in_way(const PolicyTypes policy_type)
{
    std::mt19937 gen(1);

    constexpr double ROBOT_OBS_DIST = 1;

    const int T = 20;
    const double dt = 0.5;

    const double robot_radius = 1; 
    const Eigen::Vector2d start_pos(-10, 0);
    const Eigen::Vector2d end_pos = -start_pos;

    const double obs_radius = 2; 
    const Eigen::Vector2d obs_pos = (end_pos + start_pos) / 2.0;

    const ilqr::DynamicsFunc dynamics_func = simulators::simplecar::make_discrete_dynamics_func(dt);


    // Convert into Circle types.
    const Circle robot_goal(robot_radius, end_pos);
    const Circle obstacle(obs_radius, obs_pos);

    // Construct environments for the ilqr tree.
    CircleWorld env_with_obs, env_without_obs;
    env_with_obs.add_obstacle(obstacle);

    // Construct the true environment model.
    CircleWorld true_environment;

    const double prob_obstacle = 0.2;
    std::bernoulli_distribution obs_sample(prob_obstacle); 

    bool adding_obstacle = true;
    //bool adding_obstacle = obs_sample(gen);
    if (adding_obstacle)
    {
        true_environment.add_obstacle(obstacle);
    }

    // Create cost functions. On the final step, we weight the goal cost highly.
    const std::vector<double> REGULAR_COST_WEIGHTS = {1, 0, 1};
    const std::vector<double> FINAL_COST_WEIGHTS = {0, 10, 1};
    const ilqr::CostFunc cost_w_obs = make_cost_func(env_with_obs, robot_radius, robot_goal, REGULAR_COST_WEIGHTS);
    const ilqr::CostFunc final_cost_w_obs = make_cost_func(env_with_obs, robot_radius, robot_goal, FINAL_COST_WEIGHTS);
    const ilqr::CostFunc cost_without_obs = make_cost_func(env_without_obs, robot_radius, robot_goal, REGULAR_COST_WEIGHTS);
    const ilqr::CostFunc final_cost_without_obs = make_cost_func(env_without_obs, robot_radius, robot_goal, FINAL_COST_WEIGHTS);

    const ilqr::CostFunc true_cost_func = make_cost_func(true_environment, robot_radius, robot_goal, REGULAR_COST_WEIGHTS);
    const ilqr::CostFunc final_true_cost_func = make_cost_func(true_environment, robot_radius, robot_goal, FINAL_COST_WEIGHTS);

    using State = simulators::simplecar::State;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(STATE_DIM);
    x0[State::POS_X] = start_pos[0];
    x0[State::POS_Y] = start_pos[1];
    x0[State::VEL] = 0.0;
    x0[State::ANG] = 0.0;

    Eigen::VectorXd xT = Eigen::VectorXd::Zero(STATE_DIM);
    x0[State::POS_X] = end_pos[0];
    x0[State::POS_Y] = end_pos[1];
    x0[State::VEL] = 0.0;
    x0[State::ANG] = 0.0;

    // Initialize the belief for the policy optimization.
    std::vector<double> obstacle_beliefs = {prob_obstacle, 1-prob_obstacle};
    std::vector<ilqr::CostFunc> cost_funcs = {cost_w_obs, cost_without_obs};
    std::vector<ilqr::CostFunc> final_cost_funcs = {final_cost_w_obs, final_cost_without_obs};

    std::vector<Eigen::VectorXd> states;
    states.push_back(x0);

    Eigen::VectorXd xt = x0;
    // Run the control policy.
    double rollout_cost = 0;
    for (int t = 0; t < T; ++t)
    {
        Eigen::VectorXd ut;
        // Many policies require a ilqr_tree as an argument.
        ilqr::iLQRTree ilqr_tree(STATE_DIM, CONTROL_DIM);
        switch(policy_type)
        {
        case PolicyTypes::HINDSIGHT:
        {
            ut = policy::hindsight_tree_policy(t, xt, T, xT, Eigen::VectorXd::Zero(CONTROL_DIM), 
                  obstacle_beliefs, dynamics_func, cost_funcs, final_cost_funcs, ilqr_tree);
            break;
        }
        case PolicyTypes::TRUE_ILQR:
        {
            ut = policy::chain_policy(t, xt, T, xT, Eigen::VectorXd::Zero(CONTROL_DIM), 
              dynamics_func, true_cost_func, final_true_cost_func, ilqr_tree);
              break;
        }
        case PolicyTypes::ARGMAX_ILQR:
        {
            const int arg_max_prob = std::distance(obstacle_beliefs.begin(),
                    std::max_element(obstacle_beliefs.begin(), obstacle_beliefs.end()));
            ut = policy::chain_policy(t, xt, T, xT, Eigen::VectorXd::Zero(CONTROL_DIM), 
                  dynamics_func, cost_funcs[arg_max_prob], final_cost_funcs[arg_max_prob], 
                  ilqr_tree);
            break; 
        }
        case PolicyTypes::PROB_WEIGHTED_CONTROL:
        {
            ut = policy::probability_weighted_policy(t, xt, T, xT, Eigen::VectorXd::Zero(CONTROL_DIM), 
                  obstacle_beliefs, dynamics_func, cost_funcs, final_cost_funcs);
            break;
        }
        };

        rollout_cost += true_cost_func(xt, ut);

        const Eigen::VectorXd xt1 = dynamics_func(xt, ut);

        // Update belief if we are in the observation radius of the obstacle
        const Circle robot_t1(robot_radius, xt1[State::POS_X], xt1[State::POS_Y]);
        double dist_to_obs;
        obstacle.distance(robot_t1, dist_to_obs);
        if (dist_to_obs < ROBOT_OBS_DIST)
        {
            obstacle_beliefs[0] = 1.0;
            obstacle_beliefs[1] = 0.0;
        }

        xt = xt1;
        states.push_back(xt);
    }

    //PRINT(" x(" << T-1 << ")= " << xt.transpose());
    const std::string policy_name = to_string(policy_type);
    WARN(policy_name  << ": total cost: " << rollout_cost);

    constexpr char LQR_TREE_DIR[] = "~/Documents/CMU/lqr_trees/";
    print_to_csv(states, std::string(LQR_TREE_DIR) + "/src/python/vis/" + policy_name + "_states.csv");
}

int main()
{
    single_obstacle_in_way(PolicyTypes::TRUE_ILQR);
    //single_obstacle_in_way(PolicyTypes::HINDSIGHT);
    //single_obstacle_in_way(PolicyTypes::ARGMAX_ILQR);
    //single_obstacle_in_way(PolicyTypes::PROB_WEIGHTED_CONTROL);
    return 0;
}
