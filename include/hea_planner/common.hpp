#ifndef HEA_PLANNER_COMMON_HPP_
#define HEA_PLANNER_COMMON_HPP_

#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

namespace hea_planner {

constexpr double PI = 3.14159265358979323846;

using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using Path3D = std::vector<Vector3d>;

// Kinodynamic limits
struct KinodynamicLimits {
    double max_vel = 2.0;    ///< Maximum velocity (m/s)
    double max_acc = 3.0;    ///< Maximum acceleration (m/s^2)
    double max_jerk = 20.0;  ///< Maximum jerk (m/s^3)
    double max_tilt = 0.523; ///< Maximum tilt angle (rad)
};

// 3D trajectory state
struct TrajectoryState {
    double t = 0.0;
    Vector3d pos = Vector3d::Zero();
    Vector3d vel = Vector3d::Zero();
    Vector3d acc = Vector3d::Zero();
    Vector3d jerk = Vector3d::Zero();
    double yaw = 0.0;
};

// Axis-Aligned Bounding Box (AABB)
struct AABBBox {
    Vector3d min_point = Vector3d::Zero();
    Vector3d max_point = Vector3d::Zero();

    Vector3d center() const {
        return 0.5 * (min_point + max_point);
    }

    Vector3d radii() const {
        return 0.5 * (max_point - min_point);
    }

    bool contains(const Vector3d& pt, double margin = 0.0) const {
        return (pt.x() >= min_point.x() + margin && pt.x() <= max_point.x() - margin &&
                pt.y() >= min_point.y() + margin && pt.y() <= max_point.y() - margin &&
                pt.z() >= min_point.z() + margin && pt.z() <= max_point.z() - margin);
    }
};

// Sphere static obstacle
struct SphereObstacle {
    Vector3d center = Vector3d::Zero();
    double radius = 0.5;

    double signedDistance(const Vector3d& pt) const {
        return (pt - center).norm() - radius;
    }
};

// Vertical cylinder static obstacle
struct CylinderObstacle {
    Vector3d center = Vector3d::Zero();
    double radius = 0.5;
    double height = 3.0;

    double signedDistance(const Vector3d& pt) const {
        double d_xy = (pt.head<2>() - center.head<2>()).norm() - radius;
        double z_min = center.z();
        double z_max = center.z() + height;
        double d_z = std::max(z_min - pt.z(), pt.z() - z_max);
        if (d_xy > 0 && d_z > 0) return std::hypot(d_xy, d_z);
        if (d_xy > 0) return d_xy;
        if (d_z > 0) return d_z;
        return std::max(d_xy, d_z);
    }
};

// Axis-aligned cuboid static obstacle
struct CuboidObstacle {
    Vector3d center = Vector3d::Zero();
    Vector3d size = Vector3d(2.0, 1.0, 4.0);

    double signedDistance(const Vector3d& pt) const {
        Vector3d half_size = 0.5 * size;
        Vector3d diff = (pt - center).cwiseAbs() - half_size;
        double outside_dist = diff.cwiseMax(0.0).norm();
        double inside_dist = std::min(std::max({diff.x(), diff.y(), diff.z()}), 0.0);
        return outside_dist + inside_dist;
    }
};

// Tracked dynamic obstacle representation
struct DynamicObstacle {
    int id = 0;
    Vector3d pos = Vector3d::Zero();
    Vector3d vel = Vector3d::Zero();
    Vector3d size = Vector3d(0.8, 0.8, 1.8);
    double radius = 0.4;
    double height = 1.8;

    Vector3d predictPosition(double delta_t) const {
        return pos + vel * delta_t;
    }
};

} // namespace hea_planner

#endif // HEA_PLANNER_COMMON_HPP_
