//
// Environment that supports circle obstacles.
//

#pragma once

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace circle_world 
{

constexpr int OBSTACLE_DIM = 2;

class Circle
{
public:
    Circle(const double radius, const double x, double y);
    Circle(const double radius, const Eigen::Vector2d &position);

    // Returns TRUE if the other circle intersects with this circle. 
    // The Euclidean distance between the edges of the circles is returned in distance. 
    // The distance will be negative if they intersect.
    bool distance(const Circle &other, double &distance) const;

    // Returns the Euclidean distance between the centroids of this circle and the other.
    double centroid_distance(const Circle &other) const;

    double radius() const { return radius_; }
    double& radius() { return radius_; }

    Eigen::Vector2d& position() { return position_; };
    const Eigen::Vector2d& position() const { return position_; };

private:
    double radius_ = 0;
    Eigen::Vector2d position_ = Eigen::Vector2d::Zero();
};

class CircleWorld
{
public:
    // Set the world boundary constraints to the default.
    CircleWorld() = default;

    // Set the world boundary constraints.
    CircleWorld(double min_x, double max_x, double min_y, double max_y); 
    // Same order as above, [min_x, max_x, min_y, max_y].
    CircleWorld(const std::array<double, 4> &world_dims);

    void add_obstacle(const double radius, double x, double y);
    void add_obstacle(const double radius, const Eigen::Vector2d& position);
    void add_obstacle(const Circle &obstacle);

    // Returns TRUE if circle intersects with any obstacle. 
    // The Euclidean distance from circle's edge to all the obstacles 
    // is returned in distance. 
    // The distance will be negative if they intersect.
    bool distances(const Circle& circle, std::vector<double> &distances) const;

    const std::vector<Circle> &obstacles() const { return obstacles_; }
    std::vector<Circle> &obstacles() { return obstacles_; }
    
    std::array<double, 4> dimensions() const { return {{min_x_, max_x_, min_y_, max_y_}}; }

private:
    std::vector<Circle> obstacles_;

    double min_x_ = -20;
    double max_x_ = 20;
    double min_y_ = -20;
    double max_y_ = 20;
};

std::ostream& operator<<(std::ostream& os, const Circle& o);

// First line is the dimensions of the world.
// Following lines are posx posy radius) for each Circle obstacle.
// Everything is set at 13 character width for each entry.
std::ostream& operator<<(std::ostream& os, const CircleWorld& world);

} // namespace circle_world 

