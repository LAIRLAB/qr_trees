
#include <experiments/simulators/circle_world.hh>

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
    const bool does_intersect = distance < 0;
    return does_intersect;
}

double Circle::centroid_distance(const Circle &other) const
{
    return (other.position() - position_).norm();
}


void CircleWorld::add_obstacle(const double radius, const Eigen::Vector2d& position)
{
    obstacles_.emplace_back(radius, position);
}

void CircleWorld::add_obstacle(const Circle &obstacle)
{
    obstacles_.push_back(obstacle);
}

bool CircleWorld::distances(const Circle& circle, std::vector<double> &distances) const
{
    bool any_intersect = false;
    const int num_obstacles = obstacles_.size();
    distances.resize(num_obstacles);
    for(int i = 0; i < num_obstacles; ++i)
    {
        const Circle& obs = obstacles_[i];
        bool does_intersect = circle.distance(obs, distances[i]);
        any_intersect = any_intersect || does_intersect;
    }
    return any_intersect;
}
