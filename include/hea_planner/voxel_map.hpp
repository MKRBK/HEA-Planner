#ifndef HEA_PLANNER_VOXEL_MAP_HPP_
#define HEA_PLANNER_VOXEL_MAP_HPP_

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace hea_planner {
namespace voxel_map {

constexpr uint8_t Unoccupied = 0;
constexpr uint8_t Occupied   = 1;
constexpr uint8_t Dilated    = 2;

// 3D Voxel Grid Map with 26-connectivity dilation and fast query
class VoxelMap {
public:
    VoxelMap() = default;

    VoxelMap(const Eigen::Vector3i& size,
             const Eigen::Vector3d& origin,
             double vox_scale)
        : map_size_(size),
          origin_(origin),
          scale_(vox_scale),
          vox_num_(map_size_.prod()),
          step_(1, map_size_(0), map_size_(1) * map_size_(0)),
          origin_center_(origin_ + Eigen::Vector3d::Constant(0.5 * scale_)),
          bounds_((map_size_.array() - 1) * step_.array()),
          step_scale_(step_.cast<double>().cwiseInverse() * scale_),
          voxels_(vox_num_, Unoccupied) {}

    // Factory constructor automatically bounding an input point cloud
    static std::shared_ptr<VoxelMap> createFromPointCloud(
        const std::vector<Eigen::Vector3d>& pts,
        double vox_scale = 0.10,
        double dilate_radius = 0.20,
        const Eigen::Vector3d& margin = Eigen::Vector3d(1.0, 1.0, 1.0))
    {
        if (pts.empty()) {
            Eigen::Vector3i default_size(100, 50, 35);
            Eigen::Vector3d default_origin(-1.0, -2.5, 0.0);
            return std::make_shared<VoxelMap>(default_size, default_origin, vox_scale);
        }

        Eigen::Vector3d p_min = pts[0];
        Eigen::Vector3d p_max = pts[0];
        for (const auto& p : pts) {
            p_min = p_min.cwiseMin(p);
            p_max = p_max.cwiseMax(p);
        }

        Eigen::Vector3d origin = p_min - margin;
        Eigen::Vector3d corner = p_max + margin;
        Eigen::Vector3d span = corner - origin;

        Eigen::Vector3i size(
            std::max(1, static_cast<int>(std::ceil(span.x() / vox_scale))),
            std::max(1, static_cast<int>(std::ceil(span.y() / vox_scale))),
            std::max(1, static_cast<int>(std::ceil(span.z() / vox_scale)))
        );

        auto vmap = std::make_shared<VoxelMap>(size, origin, vox_scale);
        vmap->setPointCloud(pts);
        if (dilate_radius > 1e-4) {
            vmap->dilate(dilate_radius);
        }
        return vmap;
    }

    inline Eigen::Vector3i getSize() const { return map_size_; }
    inline double getScale() const { return scale_; }
    inline Eigen::Vector3d getOrigin() const { return origin_; }
    inline Eigen::Vector3d getCorner() const { return map_size_.cast<double>() * scale_ + origin_; }
    inline const std::vector<uint8_t>& getVoxels() const { return voxels_; }

    inline void setOccupied(const Eigen::Vector3d& pos) {
        const Eigen::Vector3i id = ((pos - origin_) / scale_).cast<int>();
        setOccupied(id);
    }

    inline void setOccupied(const Eigen::Vector3i& id) {
        if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 &&
            id(0) < map_size_(0) && id(1) < map_size_(1) && id(2) < map_size_(2)) {
            voxels_[id.dot(step_)] = Occupied;
        }
    }

    void setPointCloud(const std::vector<Eigen::Vector3d>& pts) {
        for (const auto& p : pts) {
            setOccupied(p);
        }
    }

    // Rasterize a solid 3D box into occupied voxels
    void setSolidBox(const Eigen::Vector3d& p_min, const Eigen::Vector3d& p_max) {
        Eigen::Vector3i id_min = ((p_min - origin_) / scale_).cast<int>().cwiseMax(Eigen::Vector3i::Zero());
        Eigen::Vector3i id_max = ((p_max - origin_) / scale_).cast<int>().cwiseMin(map_size_ - Eigen::Vector3i::Constant(1));
        for (int x = id_min.x(); x <= id_max.x(); ++x) {
            for (int y = id_min.y(); y <= id_max.y(); ++y) {
                for (int z = id_min.z(); z <= id_max.z(); ++z) {
                    voxels_[x + y * step_(1) + z * step_(2)] = Occupied;
                }
            }
        }
    }

    // Dilate occupied cells by safe radius
    void dilate(double radius) {
        int r = static_cast<int>(std::ceil(radius / scale_));
        if (r <= 0) return;
        dilateSteps(r);
    }

    void dilateSteps(int r) {
        if (r <= 0) return;

        std::vector<Eigen::Vector3i> lvec, cvec;
        lvec.reserve(vox_num_ / 8);
        cvec.reserve(vox_num_ / 8);

        const int bx = map_size_(0) - 1;
        const int by = map_size_(1) - 1;
        const int bz = map_size_(2) - 1;
        const int sy = step_(1);
        const int sz = step_(2);

        // 1. Initial pass: find all Occupied cells and expand one layer
        for (int x = 0; x <= bx; ++x) {
            for (int y = 0; y <= by; ++y) {
                for (int z = 0; z <= bz; ++z) {
                    int offset = x + y * sy + z * sz;
                    if (voxels_[offset] == Occupied) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nx = x + dx;
                            if (nx < 0 || nx > bx) continue;
                            for (int dy = -1; dy <= 1; ++dy) {
                                int ny = y + dy;
                                if (ny < 0 || ny > by) continue;
                                for (int dz = -1; dz <= 1; ++dz) {
                                    int nz = z + dz;
                                    if (nz < 0 || nz > bz) continue;
                                    int noff = nx + ny * sy + nz * sz;
                                    if (voxels_[noff] == Unoccupied) {
                                        voxels_[noff] = Dilated;
                                        cvec.emplace_back(nx, ny, nz);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Multi-pass dilation for remaining layers
        for (int loop = 1; loop < r; ++loop) {
            std::swap(cvec, lvec);
            cvec.clear();
            for (const auto& id : lvec) {
                int x = id(0), y = id(1), z = id(2);
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    if (nx < 0 || nx > bx) continue;
                    for (int dy = -1; dy <= 1; ++dy) {
                        int ny = y + dy;
                        if (ny < 0 || ny > by) continue;
                        for (int dz = -1; dz <= 1; ++dz) {
                            int nz = z + dz;
                            if (nz < 0 || nz > bz) continue;
                            int noff = nx + ny * sy + nz * sz;
                            if (voxels_[noff] == Unoccupied) {
                                voxels_[noff] = Dilated;
                                cvec.emplace_back(nx, ny, nz);
                            }
                        }
                    }
                }
            }
            lvec.clear();
        }

        surf_ = cvec;
    }

    inline uint8_t query(const Eigen::Vector3d& pos) const {
        const Eigen::Vector3i id = ((pos - origin_) / scale_).cast<int>();
        return query(id);
    }

    inline uint8_t query(const Eigen::Vector3i& id) const {
        if (id(0) >= 0 && id(1) >= 0 && id(2) >= 0 &&
            id(0) < map_size_(0) && id(1) < map_size_(1) && id(2) < map_size_(2)) {
            return voxels_[id.dot(step_)];
        }
        return Occupied; // Out of bounds is treated as wall/obstacle
    }

    // Estimate continuous clearance to nearest obstacle surface
    double getClearance(const Eigen::Vector3d& pos) const {
        uint8_t q = query(pos);
        if (q == Occupied) return 0.0;

        const Eigen::Vector3i id = ((pos - origin_) / scale_).cast<int>();
        if (id(0) < 0 || id(1) < 0 || id(2) < 0 ||
            id(0) >= map_size_(0) || id(1) >= map_size_(1) || id(2) >= map_size_(2)) {
            return 0.0;
        }

        // Local search in neighboring voxels to estimate distance to nearest occupied cell
        const int search_r = 4;
        double min_dist_sq = std::numeric_limits<double>::infinity();

        int x_min = std::max(0, id(0) - search_r);
        int x_max = std::min(map_size_(0) - 1, id(0) + search_r);
        int y_min = std::max(0, id(1) - search_r);
        int y_max = std::min(map_size_(1) - 1, id(1) + search_r);
        int z_min = std::max(0, id(2) - search_r);
        int z_max = std::min(map_size_(2) - 1, id(2) + search_r);

        for (int ix = x_min; ix <= x_max; ++ix) {
            for (int iy = y_min; iy <= y_max; ++iy) {
                for (int iz = z_min; iz <= z_max; ++iz) {
                    int off = ix + iy * step_(1) + iz * step_(2);
                    if (voxels_[off] == Occupied) {
                        Eigen::Vector3d v_center = posI2D(Eigen::Vector3i(ix, iy, iz));
                        double d_sq = (pos - v_center).squaredNorm();
                        if (d_sq < min_dist_sq) {
                            min_dist_sq = d_sq;
                        }
                    }
                }
            }
        }

        if (std::isfinite(min_dist_sq)) {
            return std::max(0.0, std::sqrt(min_dist_sq) - 0.5 * scale_);
        }

        return (q == Dilated) ? (0.5 * scale_) : (search_r * scale_);
    }

    inline Eigen::Vector3d posI2D(const Eigen::Vector3i& id) const {
        return id.cast<double>() * scale_ + origin_center_;
    }

    inline Eigen::Vector3i posD2I(const Eigen::Vector3d& pos) const {
        return ((pos - origin_) / scale_).cast<int>();
    }

    inline void getSurf(std::vector<Eigen::Vector3d>& points) const {
        points.clear();
        points.reserve(surf_.size());
        for (const auto& id : surf_) {
            points.push_back(posI2D(id));
        }
    }

private:
    Eigen::Vector3i map_size_ = Eigen::Vector3i::Zero();
    Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
    double scale_ = 0.10;
    int vox_num_ = 0;
    Eigen::Vector3i step_ = Eigen::Vector3i::Zero();
    Eigen::Vector3d origin_center_ = Eigen::Vector3d::Zero();
    Eigen::Vector3i bounds_ = Eigen::Vector3i::Zero();
    Eigen::Vector3d step_scale_ = Eigen::Vector3d::Zero();
    std::vector<uint8_t> voxels_;
    std::vector<Eigen::Vector3i> surf_;
};

} // namespace voxel_map
} // namespace hea_planner

#endif // HEA_PLANNER_VOXEL_MAP_HPP_
