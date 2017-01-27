
//#include <experiments/shared_autonomy_circle.hh>
#include <experiments/shared_autonomy_circle_class.hh>
#include <utils/debug_utils.hh>

#include <array>

using StateVector = simulators::directdrive::StateVector;

int main()
{
    std::string state_output_fname, obstacle_output_fname;

    std::array<double, 4> world_dims = {{-30, 30, -30, 30}};
    circle_world::CircleWorld world(world_dims);
    world.add_obstacle(10, 0, 0);

    std::vector<double> goal_priors;
    std::vector<StateVector> goal_states;

    goal_states.push_back(StateVector(3, 25));
    goal_priors.push_back(0.5);
    goal_states.push_back(StateVector(-3, 25));
    goal_priors.push_back(0.5);

    int true_goal_ind = 0;
    int num_timesteps = 50;

    //cost_to_go = control_shared_autonomy(PolicyTypes::HINDSIGHT, world, goal_states, goal_priors, 0, state_output_fname, obstacle_output_fname);
    //PRINT("\n");
    using namespace experiments;
    SharedAutonomyCircle shared_auton_circle(PolicyTypes::HINDSIGHT, world, goal_states, goal_priors, true_goal_ind, num_timesteps);
    shared_auton_circle.run_control(shared_auton_circle.get_num_timesteps_remaining());
    double cost_to_go = shared_auton_circle.get_rollout_cost();
    PRINT("COST: " << cost_to_go);

    return 0;
}
