#pragma once

#include <filters/goal_predictor.hh>

#include <utils/debug_utils.hh>
#include <utils/print_helpers.hh>

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace
{
    constexpr double TOL = 1e-5;
    const double MAX_PROB_ANY_GOAL = 1.-1e-2;
    const double MAX_LOG_PROB_ANY_GOAL = std::log(MAX_PROB_ANY_GOAL);
}


namespace filters
{

double LogSumExp(const std::vector<double>& vals)
{
    double max_exp = *std::max_element(vals.begin(), vals.end());

    double exp_sum = 0.0;
    for (const auto& val : vals)
    {
        exp_sum += std::exp(val - max_exp);
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


std::vector<double> GoalPredictor::get_goal_distribution() const
{
    std::vector<double> goal_distribution(log_goal_distribution_.size());
    std::transform(log_goal_distribution_.begin(), log_goal_distribution_.end(), goal_distribution.begin(), [](double d){return std::exp(d);} );
    return goal_distribution;
}

double GoalPredictor::get_prob_at_ind(const std::size_t i) const
{
    return std::exp(log_goal_distribution_[i]);
}

void GoalPredictor::update_goal_distribution(const std::vector<double>& q_values, const std::vector<double>& v_values, const double scale_factor)
{
    IS_EQUAL(q_values.size(), v_values.size());
    IS_EQUAL(log_goal_distribution_.size(), v_values.size());

    for (size_t i=0; i < log_goal_distribution_.size(); i++)
    {
        double q_val = q_values[i];
        double v_val = v_values[i];
        log_goal_distribution_[i] -= (q_val - v_val) * scale_factor;
    }
    
    normalize_log_distribution();
    clip_prob_any_goal();
}

//normalize so probabilities sum to 1
//equivalently, so LogSumExp of log probabilities is zero
void GoalPredictor::normalize_log_distribution()
{
    double log_normalization_val = LogSumExp(log_goal_distribution_);
    for(double& val : log_goal_distribution_)
    {
        val -= log_normalization_val;
    }
}

//bound the max probability of any goal
//assumes that the log distribution has already been normalized
//TODO also bound the min?
void GoalPredictor::clip_prob_any_goal()
{
    //for some reason, overloading std::cout << directly with the vector log_goal_distribution_
    //causes the code to initialize a new GoalPredictor, and print that
    //instead, directly tell it to print the vector
    //print_vec(log_goal_distribution_); 

    if (get_num_goals() <= 1)
        return;

    //find max prob ind of any goal
    size_t max_prob_ind = std::distance(log_goal_distribution_.begin(), std::max_element(log_goal_distribution_.begin(), log_goal_distribution_.end()));
    if (log_goal_distribution_[max_prob_ind] > MAX_LOG_PROB_ANY_GOAL)
    {
      //see how much we will remove from probability
      double diff = std::exp(log_goal_distribution_[max_prob_ind]) - MAX_PROB_ANY_GOAL;
      //want to distribute this evenly among other goals
      double diff_per = diff/((double) get_num_goals()-1);

      //distribute this evenly in the probability space...this corresponds to doing so in log space
      //e^x_new = e^x_old + diff_per. Take log of both sides
      for (auto &v: log_goal_distribution_)
      {
          //v += std::log(1. + diff_per/(std::exp(v)));
          v = std::log(std::exp(v) + diff_per);
      }
      //set the max prob
      log_goal_distribution_[max_prob_ind] = MAX_LOG_PROB_ANY_GOAL;

    }

}

std::ostream& operator<<(std::ostream& os, const GoalPredictor &goal_predictor)
{  
    //os << goal_predictor.get_goal_distribution();
    //return os;  

    os << "{";
    for (size_t i=0; i < goal_predictor.get_num_goals(); i++)
    {
        if (i > 0)
        {
            os << ", ";
        }
        os << goal_predictor.get_prob_at_ind(i);
    }
    os << "}";
    return os;  


}



}//filters
