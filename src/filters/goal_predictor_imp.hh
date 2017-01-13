#pragma once


#include <utils/debug_utils.hh>

#include <vector>
#include <algorithm>
#include <cmath>

namespace
{
    constexpr double TOL = 1e-5;
}


namespace filters
{

double LogSumExp(const std::vector<double>& vals)
{
    double max_exp = *std::max_element(vals.begin(), vals.end());

    double exp_sum = 0.0;
    for (const auto& val : vals)
    {
        exp_sum += exp(val - max_exp);
    }
    return (std::log(exp_sum) + max_exp);
}



GoalPredictor::GoalPredictor(const std::vector<double>& initial_goal_prob)
{
    initialize(initial_goal_prob);
}

void GoalPredictor::initialize(const std::vector<double>& initial_goal_prob)
{
    log_goal_distribution_.resize(initial_goal_prob.size());
    std::transform(initial_goal_prob.begin(), initial_goal_prob.end(), log_goal_distribution_.begin(), [](double d){return std::log(d);} );
//    for (size_t i=0; i < initial_goal_prob.size(); i++) {
//        log_goal_distribution_[i] = log(initial_goal_prob[i]);
//    }
//
}


std::vector<double> GoalPredictor::get_goal_distribution()
{
    std::vector<double> goal_distribution(log_goal_distribution_.size());
    std::transform(log_goal_distribution_.begin(), log_goal_distribution_.end(), goal_distribution.begin(), [](double d){return std::exp(d);} );
    return goal_distribution;
}

double GoalPredictor::get_prob_at_ind(const std::size_t i)
{
    return std::exp(log_goal_distribution_[i]);
}

void GoalPredictor::update_goal_distribution(const std::vector<double>& q_values, const std::vector<double>& v_values)
{
    IS_EQUAL(q_values.size(), v_values.size());
    IS_EQUAL(log_goal_distribution_.size(), v_values.size());

    for (size_t i=0; i < log_goal_distribution_.size(); i++)
    {
        double q_val = q_values[i];
        double v_val = v_values[i];
        log_goal_distribution_[i] -= q_val - v_val;
    }
    
    normalize_log_distribution();
}

void GoalPredictor::normalize_log_distribution()
{
    double log_normalization_val = LogSumExp(log_goal_distribution_);
    for(double& val : log_goal_distribution_)
    {
        val -= log_normalization_val;
    }
}



}//filters
