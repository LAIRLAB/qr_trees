//
// Shervin Javdani (sjavdani@cs.cmu.edu)
// January 2017
//

#include <experiments/simulators/directdrive.hh>

#include <utils/debug_utils.hh>

namespace simulators
{
namespace directdrive
{

using namespace std::placeholders;

DirectDrive::DirectDrive(const double dt, 
                    const std::array<double, CONTROL_DIM> &control_lims,
                    const std::array<double, STATE_DIM*2> &world_lims)
    : dt_(dt), control_lims_(control_lims), world_lims_(world_lims)
{
    IS_GREATER(control_lims_[1], control_lims_[0]);
    IS_GREATER(world_lims_[1], world_lims_[0]);
    IS_GREATER(world_lims_[3], world_lims_[2]);
}

StateVector DirectDrive::operator()(const StateVector& x, const ControlVector& u)
{
    StateVector xt1 = x + u;

    xt1[POS_X] = std::min(xt1[POS_X], world_lims_[1]);
    xt1[POS_X] = std::max(xt1[POS_X], world_lims_[0]);
    xt1[POS_Y] = std::min(xt1[POS_Y], world_lims_[3]);
    xt1[POS_Y] = std::max(xt1[POS_Y], world_lims_[2]);
    return xt1;
}


} // namespace directdrive
} // namespace simulators

