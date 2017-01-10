
#include <experiments/single_obs_diffdrive.hh>
#include <utils/debug_utils.hh>

#include <array>

int main()
{
    double cost_to_go = 0;

    std::array<double, 2> probs = {{0.1, 0.9}};

    std::string state_output_fname, obstacle_output_fname;

    cost_to_go = single_obs_control_diffdrive(PolicyTypes::TRUE_ILQR, true, probs, state_output_fname, obstacle_output_fname);
    PRINT("\n");
    cost_to_go = single_obs_control_diffdrive(PolicyTypes::TRUE_ILQR, false, probs, state_output_fname, obstacle_output_fname);
    PRINT("\n");

    // So that we can compile without complaining about unused variable...
    if (cost_to_go) {}

    //cost_to_go = control_diffdrive(PolicyTypes::PROB_WEIGHTED_CONTROL, true, probs);
    //PRINT("\n");
    //cost_to_go = control_diffdrive(PolicyTypes::PROB_WEIGHTED_CONTROL, false, probs);
    //PRINT("\n");

    //cost_to_go = control_diffdrive(PolicyTypes::HINDSIGHT, true, probs);
    //PRINT("\n");
    //cost_to_go = control_diffdrive(PolicyTypes::HINDSIGHT, false, probs);
    //PRINT("\n");

    return 0;
}
