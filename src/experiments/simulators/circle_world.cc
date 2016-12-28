
#include <experiments/simulators/circle_world.hh>

#include <utils/debug_utils.hh>

#include <iomanip>
#include <ostream>

namespace
{
    constexpr int PRINT_WIDTH = 13;
}

namespace circle_world
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
    const bool does_intersect = distance < 0;
    return does_intersect;
}

double Circle::centroid_distance(const Circle &other) const
{
    return (other.position() - position_).norm();
}


CircleWorld::CircleWorld(double min_x, double max_x, double min_y, double max_y)
    : min_x_(min_x), max_x_(max_x), min_y_(min_y), max_y_(max_y)
{
    IS_GREATER(max_x, min_x);
    IS_GREATER(max_y, min_y);
}

CircleWorld::CircleWorld(const std::array<double, 4> &world_dims)
    : CircleWorld(world_dims[0], world_dims[1], world_dims[2], world_dims[3])
{
}

void CircleWorld::add_obstacle(const double radius, double x, double y)

{
    // Confirm at least part of the circle is within bounds.
    IS_GREATER_EQUAL(x, min_x_);
    IS_LESS_EQUAL(x, max_x_);
    IS_GREATER_EQUAL(y, min_y_);
    IS_LESS_EQUAL(y, max_y_);

    obstacles_.emplace_back(radius, x, y);
}

void CircleWorld::add_obstacle(const double radius, const Eigen::Vector2d& position)
{
    add_obstacle(radius, position[0], position[1]);
}

void CircleWorld::add_obstacle(const Circle &obstacle)
{
    add_obstacle(obstacle.radius(), obstacle.position()[0], obstacle.position()[1]);
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

std::ostream& operator<<(std::ostream& os, const Circle& o) 
{
    constexpr char DELIMITER[] = " ";
    const double x = o.position()[0];
    const double y = o.position()[1];
    os << std::left << std::setw(PRINT_WIDTH) << x << DELIMITER 
       << std::left << std::setw(PRINT_WIDTH) << y 
       << std::left << std::setw(PRINT_WIDTH) << o.radius();
    return os;
}

std::ostream& operator<<(std::ostream& os, const CircleWorld& world) 
{
    const std::vector<Circle>& obstacles = world.obstacles();
    const std::array<double, 4> dimensions = world.dimensions();
    for (const double dim : dimensions)
    {
        os << std::left << std::setw(PRINT_WIDTH) << dim;
    }
    os << std::endl;

    for (const Circle &obs : obstacles)
    {
        os << obs << std::endl;
    }
    return os;
}

} // namespace circle_world 

