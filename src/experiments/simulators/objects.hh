//
// Collection os simple objets like Circles etc.
// Arun Venkatraman (arunvenk@cs.cmu.edu)
// January 2017
//

#pragma once

#include <Eigen/Core>

namespace objects
{

constexpr int PRINT_WIDTH = 13;

class Circle
{
public:
    Circle(const double radius, const double x, double y);
    Circle(const double radius, const Eigen::Vector2d &position);

    // Returns TRUE if the other circle intersects with this circle. 
    // The Euclidean distance between the edges of the circles is returned in distance. 
    // The distance will be 0 or negative if they intersect.
    bool distance(const Circle &other, double &distance) const;

    // Returns the Euclidean distance between the centroids of this circle and the other.
    double centroid_distance(const Circle &other) const;

    double radius() const { return radius_; }
    double& radius() { return radius_; }

    Eigen::Vector2d& position() { return position_; };
    const Eigen::Vector2d& position() const { return position_; };

    double x() const { return position_[0]; }
    double y() const { return position_[1]; }
    double& x() { return position_[0]; }
    double& y() { return position_[1]; }

    // Used for pretty printing or python binding.
    std::string repr() const;

private:
    double radius_ = 0;
    Eigen::Vector2d position_ = Eigen::Vector2d::Zero();
};

} // namespace objects

// Provides fixed width printing (as specified above).
std::ostream& operator<<(std::ostream& os, const objects::Circle& o);
