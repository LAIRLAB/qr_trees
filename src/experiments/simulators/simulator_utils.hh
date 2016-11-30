//
// Helper utilities for simulators.
//

#pragma once

#include <ilqr/ilqr_helpers.hh> // Definition of dynamics function.

#include <Eigen/Dense>

namespace simulators
{

Eigen::VectorXd step(const Eigen::VectorXd &state, const Eigen::VectorXd &control, 
                     const double dt, const ilqr::DynamicsFunc &dynamics);

} // namespace simulators
