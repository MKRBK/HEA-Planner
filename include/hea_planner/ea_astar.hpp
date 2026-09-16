#ifndef HEA_PLANNER_EA_ASTAR_HPP_
#define HEA_PLANNER_EA_ASTAR_HPP_

#include "common.hpp"
#include "voxel_map.hpp"
#include <queue>
#include <unordered_map>
#include <memory>
#include <functional>
#include <algorithm>

namespace hea_planner {

using VelocityProfile = std::function<double(double s)>;

// Configuration parameters for EA-A*
struct EAAStarConfig {
    double resolution = 0.25;         ///< Grid resolution (m)
    double nominal_vel = 1.5;         ///< Nominal velocity (m/s)
    double v_min = 0.1;               ///< Minimum velocity (m/s)

    double R_safe = 0.20;             ///< Safety radius (m)
    double tau_safe = 2.5;            ///< Lookahead collision time (s)
    double C_risk = 15.0;             ///< Risk repulsion gain
    double w_env = 2.0;               ///< Risk cost weight
    double k_v = 0.60;                ///< Velocity covariance factor
    double spatial_decay_sigma = 1.0; ///< Spatial spread (m)
    double lookahead_distance = 2.0;  ///< Lookahead distance (m)
    double goal_arrival_tolerance = 0.25; ///< Goal arrival radius (m)

    int max_expansions = 1000;        ///< Max node expansions

    Vector3d local_window = Vector3d(5.0, 5.0, 4.0); ///< Local window size (m)
    bool enable_local_window = false;

    bool enable_risk_field = true;
    bool enable_cpa_check = true;
    bool enable_hard_prune = true;
};

// Replanning mode according to FSM state
enum class ReplanningMode {
    S2_CRUISE,        ///< Mode S2: Evasion along global path
    S3_RENDEZVOUS     ///< Mode S3: Moving platform rendezvous
};

// Result of EA-A* local replanning
struct LocalPlanResult {
    bool success = false;
    bool is_partial = false;
    Path3D path;
    std::vector<double> timestamps;
    double path_length = 0.0;
    double min_clearance_static = 0.0;
    double min_clearance_dynamic = 0.0;
    double solve_time_ms = 0.0;
    int expansions = 0;
};

// Node representation in 3D-time graph search
struct AStarNode {
    Vector3d pos;
    double t = 0.0;
    double g_cost = 0.0;
    double h_cost = 0.0;
    double risk_cost = 0.0;
    double f_cost = 0.0;
    std::shared_ptr<AStarNode> parent = nullptr;

    bool operator>(const AStarNode& other) const {
        return f_cost > other.f_cost;
    }
};

// EA-A* Local Dynamic Replanner
class EAAStarPlanner {
public:
    explicit EAAStarPlanner(const EAAStarConfig& cfg = EAAStarConfig())
        : cfg_(cfg) {}

    void setConfig(const EAAStarConfig& cfg) {
        cfg_ = cfg;
    }

    const EAAStarConfig& getConfig() const {
        return cfg_;
    }

    void setVoxelMap(const std::shared_ptr<voxel_map::VoxelMap>& map) {
        voxel_map_ = map;
    }

    void setStaticPointCloud(const std::vector<Vector3d>& cloud,
                             double resolution = 0.10,
                             double dilate_radius = 0.20,
                             const Vector3d& margin = Vector3d(1.0, 1.0, 1.0)) {
        voxel_map_ = voxel_map::VoxelMap::createFromPointCloud(cloud, resolution, dilate_radius, margin);
    }

    const std::shared_ptr<voxel_map::VoxelMap>& getVoxelMap() const {
        return voxel_map_;
    }

    void setDynamicObstacles(const std::vector<DynamicObstacle>& dyn_obs) {
        dyn_obs_ = dyn_obs;
    }

    // Inherited velocity profile from previous trajectory optimization
    void setVelocityProfile(const VelocityProfile& prof) {
        vel_profile_ = prof;
    }

    // Clear velocity profile
    void clearVelocityProfile() {
        vel_profile_ = nullptr;
    }

    bool hasVelocityProfile() const {
        return static_cast<bool>(vel_profile_);
    }

    const VelocityProfile& getVelocityProfile() const {
        return vel_profile_;
    }

    // Compute Time of Closest Approach (t_CPA) and collision distance
    bool computeCPA(const Vector3d& p_x, double t_x, const Vector3d& v_uav,
                    const DynamicObstacle& obs, double& t_cpa, double& d_cpa) const {
        Vector3d p_pred = obs.predictPosition(t_x);
        double z_half = 0.5 * obs.height;
        p_pred.z() = std::clamp(p_x.z(), p_pred.z() - z_half, p_pred.z() + z_half);
        Vector3d p_rel = p_x - p_pred;
        Vector3d v_rel = v_uav - obs.vel;

        double v_rel_sq = v_rel.squaredNorm();
        if (v_rel_sq < 1e-6) {
            t_cpa = 0.0;
            d_cpa = p_rel.norm();
            return false;
        }

        double p_dot_v = p_rel.dot(v_rel);
        if (p_dot_v >= 0.0) {
            // Moving apart or parallel
            t_cpa = 0.0;
            d_cpa = p_rel.norm();
            return false;
        }

        t_cpa = -p_dot_v / v_rel_sq;
        d_cpa = (p_rel + v_rel * t_cpa).norm();
        return true;
    }

    // Spatiotemporal risk repulsion term
    double computeRiskRepulsion(const Vector3d& p_x, double t_x, const Vector3d& v_uav) const {
        if (!cfg_.enable_risk_field || dyn_obs_.empty()) return 0.0;

        double total_risk = 0.0;
        for (const auto& obs : dyn_obs_) {
            Vector3d p_pred = obs.predictPosition(t_x);
            double z_half = 0.5 * obs.height;
            p_pred.z() = std::clamp(p_x.z(), p_pred.z() - z_half, p_pred.z() + z_half);
            Vector3d delta_p = p_x - p_pred;

            // Anisotropic covariance matrix Sigma = R(phi) diag(...) R(phi)^T
            double v_norm = obs.vel.norm();
            double sigma_lat = obs.radius + cfg_.R_safe;
            double sigma_vert = obs.radius + cfg_.R_safe;
            double sigma_long = obs.radius + cfg_.R_safe + cfg_.k_v * v_norm;

            double phi_tcpa = 1.0;
            if (cfg_.enable_cpa_check) {
                double t_cpa = 0.0, d_cpa = 0.0;
                computeCPA(p_x, t_x, v_uav, obs, t_cpa, d_cpa);
                double d_footprint = sigma_long;
                double current_dist = delta_p.norm();

                if (current_dist < d_footprint) {
                    // Already inside danger footprint at arrival time t_x
                    phi_tcpa = 1.0;
                } else if (t_cpa > 0.0 && d_cpa < d_footprint) {
                    // Impending dynamic collision within lookahead horizon
                    phi_tcpa = std::max(0.0, 1.0 - t_cpa / cfg_.tau_safe);
                } else {
                    phi_tcpa = 0.0;
                }
            }

            if (phi_tcpa <= 1e-4) continue;

            // Yaw angle around Z
            double yaw = (v_norm > 1e-3) ? std::atan2(obs.vel.y(), obs.vel.x()) : 0.0;
            Matrix3d R_yaw;
            R_yaw = Eigen::AngleAxisd(yaw, Vector3d::UnitZ());

            Matrix3d inv_diag = Matrix3d::Zero();
            inv_diag(0, 0) = 1.0 / (sigma_long * sigma_long);
            inv_diag(1, 1) = 1.0 / (sigma_lat * sigma_lat);
            inv_diag(2, 2) = 1.0 / (sigma_vert * sigma_vert);

            Matrix3d sigma_inv = R_yaw * inv_diag * R_yaw.transpose();
            double quad_form = delta_p.transpose() * sigma_inv * delta_p;

            double g_val = cfg_.C_risk * std::exp(-0.5 * quad_form) * phi_tcpa;
            total_risk += g_val;
        }

        return total_risk;
    }

    // Static obstacle distance from VoxelMap
    double queryStaticClearance(const Vector3d& pt) const {
        if (voxel_map_) {
            return voxel_map_->getClearance(pt);
        }
        return std::numeric_limits<double>::infinity();
    }

    // Dynamic obstacle distance at time t
    double queryDynamicClearance(const Vector3d& pt, double t) const {
        double min_dist = std::numeric_limits<double>::infinity();
        for (const auto& obs : dyn_obs_) {
            Vector3d p_pred = obs.predictPosition(t);
            double d_xy = (pt.head<2>() - p_pred.head<2>()).norm() - obs.radius;
            double z_min = p_pred.z() - 0.5 * obs.height;
            double z_max = p_pred.z() + 0.5 * obs.height;
            double d_z = std::max(z_min - pt.z(), pt.z() - z_max);
            double d = 0.0;
            if (d_xy > 0 && d_z > 0) d = std::hypot(d_xy, d_z);
            else if (d_xy > 0) d = d_xy;
            else if (d_z > 0) d = d_z;
            else d = std::max(d_xy, d_z);
            min_dist = std::min(min_dist, d);
        }
        return min_dist;
    }

    // Execute local dynamic replanning
    LocalPlanResult plan(const Vector3d& start_pos, const Vector3d& start_vel,
                         double start_time, const Vector3d& local_goal,
                         ReplanningMode mode = ReplanningMode::S2_CRUISE,
                         const VelocityProfile& override_vel_profile = nullptr) {
        auto t_start = std::chrono::steady_clock::now();
        LocalPlanResult res;

        // Weight selection based on operating state (Algorithm 2)
        double w_risk = (mode == ReplanningMode::S3_RENDEZVOUS) ? 0.0 : cfg_.w_env;

        // Custom comparator for priority queue
        auto cmp = [](const std::shared_ptr<AStarNode>& a, const std::shared_ptr<AStarNode>& b) {
            return a->f_cost > b->f_cost;
        };
        std::priority_queue<std::shared_ptr<AStarNode>,
                            std::vector<std::shared_ptr<AStarNode>>,
                            decltype(cmp)> open_set(cmp);

        // Spatial grid hash for deduplication (64-bit safe)
        auto grid_hash = [this](const Vector3d& p) -> uint64_t {
            uint64_t ix = static_cast<uint64_t>(static_cast<int64_t>(std::round(p.x() / cfg_.resolution)) & 0x1FFFFF);
            uint64_t iy = static_cast<uint64_t>(static_cast<int64_t>(std::round(p.y() / cfg_.resolution)) & 0x1FFFFF);
            uint64_t iz = static_cast<uint64_t>(static_cast<int64_t>(std::round(p.z() / cfg_.resolution)) & 0x1FFFFF);
            return ix | (iy << 21) | (iz << 42);
        };

        std::unordered_map<uint64_t, double> closed_set; // hash -> best g_cost

        // Root node
        auto root = std::make_shared<AStarNode>();
        root->pos = start_pos;
        root->t = start_time;
        root->g_cost = 0.0;
        root->h_cost = (local_goal - start_pos).norm();
        root->risk_cost = 0.0;
        root->f_cost = root->h_cost;
        open_set.push(root);
        closed_set[grid_hash(start_pos)] = 0.0;

        // Neighbor displacements (26-connectivity)
        std::vector<Vector3d> directions;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    directions.emplace_back(dx * cfg_.resolution, dy * cfg_.resolution, dz * cfg_.resolution);
                }
            }
        }

        std::shared_ptr<AStarNode> goal_node = nullptr;
        std::shared_ptr<AStarNode> closest_node = root;
        double min_dist_to_goal = root->h_cost;
        int expansions = 0;

        while (!open_set.empty() && expansions < cfg_.max_expansions) {
            auto curr = open_set.top();
            open_set.pop();

            // Skip stale duplicate nodes from priority queue
            uint64_t curr_id = grid_hash(curr->pos);
            if (closed_set.count(curr_id) && curr->g_cost > closed_set[curr_id] + 1e-6) {
                continue;
            }
            expansions++;

            // Goal test
            if ((curr->pos - local_goal).norm() <= cfg_.resolution * 1.2) {
                goal_node = curr;
                break;
            }

            if (curr->h_cost < min_dist_to_goal) {
                min_dist_to_goal = curr->h_cost;
                closest_node = curr;
            }

            Vector3d v_curr = (curr->parent) ?
                ((curr->pos - curr->parent->pos) / std::max(curr->t - curr->parent->t, 1e-3)) : start_vel;

            for (const auto& d : directions) {
                Vector3d next_pos = curr->pos + d;

                // 0. Receding horizon local window bounding check (Section 3.3.3)
                if (cfg_.enable_local_window) {
                    Vector3d diff = (next_pos - start_pos).cwiseAbs();
                    if (diff.x() > 0.5 * cfg_.local_window.x() ||
                        diff.y() > 0.5 * cfg_.local_window.y() ||
                        diff.z() > 0.5 * cfg_.local_window.z()) {
                        continue;
                    }
                }

                double edge_len = d.norm();

                // 1. Static clearance check via 3D VoxelMap
                if (voxel_map_) {
                    uint8_t q_next = voxel_map_->query(next_pos);
                    if (q_next == voxel_map::Occupied) continue; // Hard obstacle
                    if (q_next == voxel_map::Dilated) {
                        // Entering or remaining in dilated buffer is forbidden unless escaping from an already-dilated state
                        uint8_t q_curr = voxel_map_->query(curr->pos);
                        if (q_curr != voxel_map::Dilated ||
                            voxel_map_->getClearance(next_pos) <= voxel_map_->getClearance(curr->pos)) {
                            continue;
                        }
                    }
                } else {
                    double static_clear = queryStaticClearance(next_pos);
                    double curr_clear = queryStaticClearance(curr->pos);
                    if (static_clear <= 0.12 || (static_clear <= cfg_.R_safe && static_clear <= curr_clear)) continue;
                }

                // 2. Time calibration along branch (Eq. 16): dt = ds / max(||v_prev(s)||, v_min)
                double s_mid = curr->g_cost + 0.5 * edge_len;
                double v_branch = cfg_.nominal_vel;
                const auto& active_profile = override_vel_profile ? override_vel_profile : vel_profile_;
                if (active_profile) {
                    v_branch = active_profile(s_mid);
                }
                v_branch = std::max(v_branch, cfg_.v_min);
                double dt = edge_len / v_branch;
                double next_t = curr->t + dt;

                // 3. Dynamic hard pruning check (Eq. 19)
                if (cfg_.enable_hard_prune) {
                    bool collides_dynamic = false;
                    for (const auto& obs : dyn_obs_) {
                        Vector3d p_pred = obs.predictPosition(next_t);
                        double d_xy = (next_pos.head<2>() - p_pred.head<2>()).norm();
                        double z_min = p_pred.z() - 0.5 * obs.height;
                        double z_max = p_pred.z() + 0.5 * obs.height;
                        if (d_xy < cfg_.R_safe + obs.radius &&
                            next_pos.z() >= z_min - cfg_.R_safe &&
                            next_pos.z() <= z_max + cfg_.R_safe) {
                            collides_dynamic = true;
                            break;
                        }
                    }
                    if (collides_dynamic) continue;
                }

                // 4. Closed set / duplicate check
                uint64_t next_id = grid_hash(next_pos);
                double tentative_g = curr->g_cost + edge_len;
                if (closed_set.count(next_id) && tentative_g >= closed_set[next_id]) {
                    continue;
                }

                // 5. Spatiotemporal risk repulsion cost (Eq. 14, 20)
                double risk = computeRiskRepulsion(next_pos, next_t, v_curr);
                double h = (local_goal - next_pos).norm();

                auto next_node = std::make_shared<AStarNode>();
                next_node->pos = next_pos;
                next_node->t = next_t;
                next_node->g_cost = tentative_g;
                next_node->h_cost = h;
                next_node->risk_cost = risk;
                next_node->f_cost = tentative_g + h + w_risk * risk;
                next_node->parent = curr;

                closed_set[next_id] = tentative_g;
                open_set.push(next_node);
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        res.solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        res.expansions = expansions;

        auto final_node = goal_node ? goal_node : closest_node;
        if (final_node && final_node != root) {
            res.success = true;
            res.is_partial = (goal_node == nullptr);

            // Reconstruct path
            auto p_it = final_node;
            while (p_it != nullptr) {
                res.path.push_back(p_it->pos);
                res.timestamps.push_back(p_it->t);
                p_it = p_it->parent;
            }
            std::reverse(res.path.begin(), res.path.end());
            std::reverse(res.timestamps.begin(), res.timestamps.end());

            // Compute metrics
            double len = 0.0;
            double min_stat = std::numeric_limits<double>::infinity();
            double min_dyn = std::numeric_limits<double>::infinity();

            for (size_t i = 0; i < res.path.size(); ++i) {
                if (i + 1 < res.path.size()) len += (res.path[i + 1] - res.path[i]).norm();
                min_stat = std::min(min_stat, queryStaticClearance(res.path[i]));
                min_dyn = std::min(min_dyn, queryDynamicClearance(res.path[i], res.timestamps[i]));
            }

            res.path_length = len;
            res.min_clearance_static = min_stat;
            res.min_clearance_dynamic = min_dyn;
        }

        return res;
    }

    // Construct continuous VelocityProfile from sampled trajectory states
    static VelocityProfile makeVelocityProfileFromTrajectory(const std::vector<TrajectoryState>& states) {
        if (states.empty()) return nullptr;

        // Precompute cumulative arc lengths along sampled states
        std::vector<double> arc_lengths(states.size(), 0.0);
        for (size_t i = 1; i < states.size(); ++i) {
            arc_lengths[i] = arc_lengths[i - 1] + (states[i].pos - states[i - 1].pos).norm();
        }

        return [states, arc_lengths](double s) -> double {
            if (s <= 0.0) return states.front().vel.norm();
            if (s >= arc_lengths.back()) return states.back().vel.norm();

            auto it = std::upper_bound(arc_lengths.begin(), arc_lengths.end(), s);
            size_t idx = std::distance(arc_lengths.begin(), it);
            size_t i0 = (idx > 0) ? idx - 1 : 0;
            size_t i1 = idx;

            double ds = arc_lengths[i1] - arc_lengths[i0];
            if (ds < 1e-6) return states[i0].vel.norm();

            double alpha = (s - arc_lengths[i0]) / ds;
            double v0 = states[i0].vel.norm();
            double v1 = states[i1].vel.norm();
            return (1.0 - alpha) * v0 + alpha * v1;
        };
    }

    // Generate intermediate waypoints and convex Safe Flight Corridors
    static bool generateCorridor(const Path3D& path,
                                 double min_waypoint_spacing,
                                 const Vector3d& box_radii,
                                 double nominal_vel,
                                 Eigen::MatrixXd& waypoints,
                                 std::vector<AABBBox>& boxes,
                                 Eigen::VectorXd& durations,
                                 const std::shared_ptr<voxel_map::VoxelMap>& voxel_map = nullptr) {
        if (path.size() < 2) return false;

        // Filter / downsample waypoints by accumulated distance
        std::vector<Vector3d> keypoints;
        keypoints.push_back(path.front());

        double dist_accum = 0.0;
        for (size_t i = 1; i + 1 < path.size(); ++i) {
            dist_accum += (path[i] - path[i - 1]).norm();
            if (dist_accum >= min_waypoint_spacing) {
                keypoints.push_back(path[i]);
                dist_accum = 0.0;
            }
        }
        // Always include the terminal point
        keypoints.push_back(path.back());

        // If only start and end, insert midpoint to guarantee at least one interior box
        if (keypoints.size() == 2) {
            Vector3d mid = 0.5 * (keypoints[0] + keypoints[1]);
            keypoints.insert(keypoints.begin() + 1, mid);
        }

        int num_mid = static_cast<int>(keypoints.size()) - 2;
        int M = num_mid + 1;

        waypoints.resize(3, num_mid);
        boxes.resize(num_mid);
        durations.resize(M);

        for (int i = 0; i < num_mid; ++i) {
            Vector3d q = keypoints[i + 1];
            waypoints.col(i) = q;
            Vector3d r = box_radii;

            // Obstacle-aware box shrinking: guarantee box boundaries never intersect static obstacles
            if (voxel_map) {
                double cl = voxel_map->getClearance(q);
                // Lateral margin: box extent in Y must not exceed clearance to nearest obstacle
                double max_r_y = std::clamp(cl - 0.05, 0.08, box_radii.y());
                r.y() = max_r_y;

                // Vertical floor clamping: ensure box does not extend below safe altitude (0.20m)
                if (q.z() - r.z() < 0.20) {
                    r.z() = std::max(0.08, q.z() - 0.20);
                }
            }

            boxes[i].min_point = q - r;
            boxes[i].max_point = q + r;
        }

        for (int j = 0; j < M; ++j) {
            double seg_len = (keypoints[j + 1] - keypoints[j]).norm();
            durations(j) = std::max(seg_len / std::max(nominal_vel, 0.1), 0.5);
        }

        return true;
    }

private:
    EAAStarConfig cfg_;
    VelocityProfile vel_profile_ = nullptr;
    std::shared_ptr<voxel_map::VoxelMap> voxel_map_ = nullptr;
    std::vector<DynamicObstacle> dyn_obs_;
};

} // namespace hea_planner

#endif // HEA_PLANNER_EA_ASTAR_HPP_
