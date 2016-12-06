#pragma once

#include <utils/debug_utils.hh>

#include <cmath>
#include <numeric>
#include <vector>

namespace
{
    constexpr double TOL = 1e-5;
}

namespace filters
{

template<typename T>
DiscreteFilter<T>::DiscreteFilter(const std::unordered_map<T, double> &priors)
{
      beliefs_ = priors;
      const double prob_sum = Z();
      IS_ALMOST_EQUAL(prob_sum, 1.0, TOL);
}

template <typename T>
void DiscreteFilter<T>::update(const Eigen::VectorXd &z_t, const ObsFunc& obs_func, 
        const DistFunc &dist_func)
{
    std::vector<double> exp_dists(beliefs_.size());
    double Z_obs = 0.0; // Partition for p(z|x)
    int i = 0; // counter to go along with range-based iterator
    for (auto &pair : beliefs_)
    {
        const Eigen::VectorXd hat_z_t = obs_func(pair.first);
        exp_dists[i] = std::exp(-dist_func(z_t, hat_z_t));
        Z_obs += exp_dists[i];
        ++i;
    }
    // Compute normalized probabilities from the exp(-dist) quantities.
    std::vector<double> p_obs(beliefs_.size());
    std::transform(exp_dists.begin(), exp_dists.end(), p_obs.begin(),
            [Z_obs](double e)
            {
                return e/Z_obs;
            }
            );
            
    i = 0;
    for (auto &pair : beliefs_)
    {
        // Update, p(x') = p(z|x) * p(x)
        pair.second *= p_obs[i];
        ++i;
    }
    normalize();
}

template<typename T>
void DiscreteFilter<T>::normalize()
{
    const double Z = this->Z();
    for (auto &pair : beliefs_)
    {
        pair.second /= Z;
    }
}

template<typename T>
double DiscreteFilter<T>::Z()
{
    const double z = std::accumulate(beliefs_.begin(), beliefs_.end(), 0.0, 
            [](double sum, const std::pair<T, double> &pair)
            {
                return sum + pair.second;
            });

    return z;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const DiscreteFilter<T> &filter)
{  
    os << "{";
    int i = 0;
    for (const auto& pair : filter.beliefs())
    {
        if (i > 0)
        {
            os << ", ";
        }
        os << pair.first << ": " << pair.second;
        ++i;
    }
    os << "}";
    return os;  
}

}// namespace filters


