//
// Discrete prediction over goals, using MaxEnt IOC framework
//
//

#pragma once

#include <vector>
#include <iostream>

namespace filters
{


class GoalPredictor
{
public:
    GoalPredictor(const std::vector<double>& initial_goal_prob);
    
    void initialize(const std::vector<double>& initial_goal_prob);

    //get the current distribution over Goals
    std::vector<double> get_goal_distribution() const;

    double get_prob_at_ind (const std::size_t i) const;
    size_t get_num_goals() const {return log_goal_distribution_.size();}

    void update_goal_distribution(const std::vector<double>& q_values, const std::vector<double>& v_values, const double scale_factor=1.0);


private:
    void normalize_log_distribution();
    void clip_prob_any_goal();

    std::vector<double> log_goal_distribution_;
};


std::ostream& operator<<(std::ostream& os, const GoalPredictor &goal_predictor);

} //filters

#include <filters/goal_predictor_imp.hh>
