
#include <experiments/shared_autonomy_circle.hh>
#include <utils/debug_utils.hh>

#include <array>

int main()
{
    double cost_to_go = 0;

    std::array<double, 2> probs = {{0.1, 0.9}};

    std::string state_output_fname, obstacle_output_fname;

    std::array<double, 4> world_dims = {{-30, 30, -30, 30}};
    circle_world::CircleWorld w1(world_dims);
    circle_world::CircleWorld w2(world_dims);
    w1.add_obstacle(10, -2, 0);
    w2.add_obstacle(10, 0, 2);

    cost_to_go = control_shared_autonomy(PolicyTypes::TRUE_ILQR, w1, w2, 
            probs, state_output_fname, obstacle_output_fname);
    PRINT("\n");

    return 0;
}
