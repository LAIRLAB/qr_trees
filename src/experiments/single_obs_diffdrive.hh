
#pragma once

#include <array>
#include <string>

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
            return "ilqr_true";
        case PolicyTypes::ARGMAX_ILQR:
            return "argmax";
        case PolicyTypes::PROB_WEIGHTED_CONTROL:
            return "weighted";
    };
    return "Unrecognized policy type. Error.";
}

// :param OBS_PRIOR - Prior that there is an obstacle, prior there is no obstacle.
double control_diffdrive(const PolicyTypes policy, 
        const bool true_world_with_obs, 
        const std::array<double, 2> &OBS_PRIOR,
        std::string &state_output_fname,
        std::string &obstacle_output_fname
        );

