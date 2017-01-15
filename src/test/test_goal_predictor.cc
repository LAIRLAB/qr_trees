
#include <filters/goal_predictor.hh>
#include <utils/debug_utils.hh>

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>


void normalize_vector(std::vector<double>& v);
std::ostream& operator<<(std::ostream& os, const std::vector<double> &vec);

std::ostream& operator<<(std::ostream& os, const std::vector<double> &vec)
{
    os << "{";
    int i=0;
    for (const auto& val: vec)
    {
        if (i > 0)
        {
            os << ", ";
        }
        os << val;
        ++i;
    }
    os << "}";
    return os;  
}


void normalize_vector(std::vector<double>& v)
{
    double sum_v = std::accumulate(v.begin(), v.end(), 0.0);
    for (auto& val: v)
      val /= sum_v;

}

int main()
{
    const int NUM_GOALS = 3;
    std::vector<double> goal_probs(NUM_GOALS, 1./((double) NUM_GOALS));


    filters::GoalPredictor goal_predictor(goal_probs);

    PRINT("initial_goal_prob: " << goal_predictor);

    std::vector<double> q_values(NUM_GOALS, 1.0);
    std::vector<double> v_values(NUM_GOALS, 2.0);

    
    goal_predictor.update_goal_distribution(q_values, v_values);

    PRINT("after update 1 (should be unchanged): " << goal_predictor);

    q_values[0] = 0.5;
    goal_predictor.update_goal_distribution(q_values, v_values);

    for (size_t i=0; i < NUM_GOALS; i++){
      goal_probs[i] *= std::exp(v_values[i] - q_values[i]);
    }
    normalize_vector(goal_probs);

    PRINT("after update 2: " << goal_predictor);
    PRINT("after update 2 should be: " << goal_probs);

    q_values[0] = 0.5;
    v_values[2] = 3.0;
    goal_predictor.update_goal_distribution(q_values, v_values);

    for (size_t i=0; i < NUM_GOALS; i++){
      goal_probs[i] *= std::exp(v_values[i] - q_values[i]);
    }
    normalize_vector(goal_probs);

    PRINT("after update 3: " << goal_predictor);
    PRINT("after update 3 should be: " << goal_probs);


    return 0;
}
