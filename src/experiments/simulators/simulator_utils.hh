//
// Helper utilities for simulators.
//

#pragma once

#include <ilqr/ilqr_taylor_expansions.hh> // Definition of dynamics function.

#include <Eigen/Dense>

namespace simulators
{

constexpr double INTEGRATION_FREQUENCY = 5;
constexpr double MIN_INTEGRATION_DT = 1e-2;


Eigen::VectorXd step(const Eigen::VectorXd &state, const Eigen::VectorXd &control, 
                     const double dt, const ilqr::DynamicsFunc &dynamics, 
                     const double min_integration_dt = MIN_INTEGRATION_DT,
                     const double integration_frequency  = INTEGRATION_FREQUENCY
                     );

} // namespace simulators
