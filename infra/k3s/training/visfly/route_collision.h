#pragma once

#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace rl2 {

using RoutePoint = std::array<double, 3>;

struct CollisionHit {
  std::array<float, 3> position;
  bool outside;
};

// Match collision_points: float32 VisFly coordinates, float64 subtraction and
// norm, then a strict float32 distance comparison. Do not enable fast-math.
template <typename Query>
bool routePointCollides(const RoutePoint& point, double radius, Query&& query) {
  const std::array<float, 3> habitat{
      -static_cast<float>(point[1]), static_cast<float>(point[2]),
      -static_cast<float>(point[0])};
  const auto hit = query(habitat, static_cast<float>(radius));
  if (hit.outside) return true;
  const double dx = static_cast<double>(hit.position[0]) - habitat[0];
  const double dy = static_cast<double>(hit.position[1]) - habitat[1];
  const double dz = static_cast<double>(hit.position[2]) - habitat[2];
  const auto distance = static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz));
  return distance < static_cast<float>(radius);
}

// OMPL 2.0.1 DiscreteMotionValidator checks the end, then intermediate probes
// breadth-first. The start is checked by its caller. OMPL supplies segments.
template <typename Query>
bool routeMotionIsValid(const RoutePoint& start, const RoutePoint& end,
                        unsigned int segments, double radius, Query&& query) {
  if (routePointCollides(end, radius, query)) return false;
  if (segments < 2) return true;
  std::vector<std::pair<unsigned int, unsigned int>> pending{{1, segments - 1}};
  for (std::size_t index = 0; index < pending.size(); ++index) {
    const auto interval = pending[index];
    const auto middle = interval.first + (interval.second - interval.first) / 2;
    const double fraction = static_cast<double>(middle) / segments;
    RoutePoint point;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      point[axis] = start[axis] + (end[axis] - start[axis]) * fraction;
    }
    if (routePointCollides(point, radius, query)) return false;
    if (interval.first < middle) pending.emplace_back(interval.first, middle - 1);
    if (middle < interval.second) pending.emplace_back(middle + 1, interval.second);
  }
  return true;
}

}  // namespace rl2
