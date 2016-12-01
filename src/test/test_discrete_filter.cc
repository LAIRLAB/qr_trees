

#include <experiments/simulators/pendulum.hh>
#include <filters/discrete_filter.hh>
#include <utils/debug_utils.hh>

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr int STATE_DIM = 2;
    constexpr int CONTROL_DIM = 1;
    constexpr double DT = 0.1;
    constexpr double LENGTH = 1.0;
    
    constexpr int T = 20;

    std::ostream& operator<<(std::ostream& os, const  std::unordered_map<double, double>& beliefs)  
    {  
        os << "{";
        int i = 0;
        for (const auto& pair : beliefs)
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

    double weighted_squared_norm(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
    {
        return 100.0*(x1-x2).squaredNorm();
    }



}

int main()
{
    std::random_device rd;
    std::mt19937 gen(1);

    const std::vector<double> damping_coeffs = {0.1, 0.5, 0.3, 1e-3};

    std::unordered_map<double, double> uniform_prior;
    for (const auto &d: damping_coeffs)
    {
        uniform_prior.emplace(d, 1.0/damping_coeffs.size());
    }

    filters::DiscreteFilter<double> filter(uniform_prior);


    std::uniform_int_distribution<> dis(0, damping_coeffs.size()-1);
    const double true_damping_coeff = damping_coeffs[dis(gen)];
    auto true_dynamics = simulators::pendulum::make_discrete_dynamics_func(DT, LENGTH, true_damping_coeff);

    // Simulate the pendulum.
    Eigen::VectorXd x0(STATE_DIM);
    x0 << M_PI/2.0, 1.0;

    Eigen::VectorXd xt = x0;
    Eigen::VectorXd ut = Eigen::VectorXd::Random(CONTROL_DIM);
    filters::DiscreteFilter<double>::ObsFunc obs_func = [&xt, &ut](double damping_coeff)
        {
            auto dynamics = simulators::pendulum::make_discrete_dynamics_func(DT, LENGTH, damping_coeff);
            return dynamics(xt, ut);
        };


    SUCCESS("True Damping: " << true_damping_coeff);
    for (int t = 0; t < T; ++t)
    {
        const auto beliefs = filter.beliefs();
        PRINT(" t=" << t << ": " <<  beliefs);

        ut = 10.*Eigen::VectorXd::Random(CONTROL_DIM);
        Eigen::VectorXd xt1 = true_dynamics(xt, ut);

        // Full state observation model.
        Eigen::VectorXd zt1 = xt1; 

        filter.update(zt1, obs_func, weighted_squared_norm);
        xt = xt1;
    }

    const auto beliefs = filter.beliefs();
    for (const double d: damping_coeffs)
    {
        IS_GREATER_EQUAL(beliefs.at(true_damping_coeff), beliefs.at(d));
    }
    return 0;
}
