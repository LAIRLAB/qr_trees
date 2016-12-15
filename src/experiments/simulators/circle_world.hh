//
// Environment that supports circle obstacles.
//

#pragma once

#include <Eigen/Dense>

#include <vector>

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
    CircleWorld() = default;

    void add_obstacle(const double radius, const Eigen::Vector2d& position);
    void add_obstacle(const Circle &obstacle);

    // Returns TRUE if circle intersects with any obstacle. 
    // The Euclidean distance from circle's edge to all the obstacles 
    // is returned in distance. 
    // The distance will be negative if they intersect.
    bool distances(const Circle& circle, std::vector<double> &distances) const;

    const std::vector<Circle> &obstacles() const { return obstacles_; }
    std::vector<Circle> &obstacles() { return obstacles_; }

private:
    std::vector<Circle> obstacles_;

};
