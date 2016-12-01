//
// Discrete item filter.
//

#pragma once

#include <ilqr/ilqr_helpers.hh>
#include <utils/debug_utils.hh>

#include <functional>
#include <unordered_map>

namespace filters
{

double squared_dist(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2)
{
    return (x1-x2).squaredNorm();
}

template<typename T>
class DiscreteFilter 
{
public:
    // Observation function takes the belief state "item" of type T 
    // and returns a predicted next observation \hat{z}_{t}.
    using ObsFunc = std::function<Eigen::VectorXd(const T&)>;

    using DistFunc = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

    // The number of ranges is the number of filter dimension.
    DiscreteFilter(const std::unordered_map<T, double> &priors);

    // Updates the belief of all "item"s given the observation z_t. obs_func() synthesizes
    // observations given an item of type T.
    void update(const Eigen::VectorXd &z_t, const ObsFunc &obs_func, 
            const DistFunc &dist_func = squared_dist);
    
    const std::unordered_map<T, double> &beliefs() const { return beliefs_; };

private:
    // Normalizes the distribution using the partition function.
    void normalize();

    // Computes the partition function over the beliefs.
    double Z();

    std::unordered_map<T, double> beliefs_;
};


}// namespace filters

#include <filters/discrete_filter_imp.hh>
