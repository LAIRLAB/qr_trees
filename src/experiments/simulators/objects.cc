//
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// January 2017
//

#include <experiments/simulators/objects.hh>

#include <iomanip>
#include <sstream>

namespace objects
{

Circle::Circle(const double radius, const double x, double y)
    : radius_(radius)
{
    position_ << x, y;
}

Circle::Circle(const double radius, const Eigen::Vector2d &position)
    : radius_(radius), position_(position)
{
}


bool Circle::distance(const Circle &other, double &distance) const
{
    // Compute the difference in the centroid
    distance = centroid_distance(other) - other.radius() - radius_;
    const bool does_intersect = distance <= 0;
    return does_intersect;
}

double Circle::centroid_distance(const Circle &other) const
{
    return (other.position() - position_).norm();
}

std::string Circle::repr() const
{
    std::ostringstream oss;
    oss << "Circle(" << radius() 
    << ", [" << x() << ", " << y() << "])"; 
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Circle& o) 
{
    constexpr char DELIMITER[] = " ";
    const double x = o.position()[0];
    const double y = o.position()[1];
    os << std::left << std::setw(objects::PRINT_WIDTH) << x << DELIMITER 
       << std::left << std::setw(objects::PRINT_WIDTH) << y 
       << std::left << std::setw(objects::PRINT_WIDTH) << o.radius();
    return os;
}

} // namespace objects

