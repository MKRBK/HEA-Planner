#ifndef HEA_PLANNER_EA_ACO_HPP_
#define HEA_PLANNER_EA_ACO_HPP_

#include "common.hpp"
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

namespace hea_planner {

// Configuration parameters for EA-ACO
struct EAACOConfig {
    double alpha = 2.0;            ///< Pheromone weight
    double beta = 4.0;             ///< Heuristic distance weight
    double mu = 4.0;               ///< Environmental factor weight
    double rho = 0.30;             ///< Evaporation rate
    double Q = 1.0;                ///< Pheromone deposit scale
    int num_ants = 30;             ///< Number of ants
    int max_iterations = 200;      ///< Max iterations
    int step_max = 500;            ///< Max steps per ant

    double R1 = 0.20;              ///< Hard obstacle clearance (m)
    double R2 = 0.70;              ///< Soft clearance buffer (m)
    double psi_max = M_PI / 2.0;   ///< Max turn angle (rad)

    bool enable_obstacle_factor = true;
    bool enable_turning_factor = true;
    bool enable_env_pheromone_weight = true;

    double resolution = 0.10;      ///< Grid resolution (m)

    double initial_pheromone = 1.0;
    double min_pheromone = 0.01;
    double max_pheromone = 20.0;

    double keypoint_angle_cos = 0.99;
    double keypoint_min_distance = 0.80;

    Vector3d map_min = Vector3d(-6.0, -6.0, 0.0);
    Vector3d map_max = Vector3d(18.0, 22.0, 4.0);
};

// Result of global path planning
struct GlobalPlanResult {
    bool success = false;
    Path3D path;
    double path_length = 0.0;
    double min_clearance = 0.0;
    int num_turns = 0;
    double solve_time_ms = 0.0;
};

// EA-ACO 3D Global Planner
class EAACOPlanner {
public:
    explicit EAACOPlanner(const EAACOConfig& cfg = EAACOConfig())
        : cfg_(cfg), rng_(20260324) {}

    void setSeed(unsigned int seed) {
        rng_.seed(seed);
    }

    void setConfig(const EAACOConfig& cfg) {
        cfg_ = cfg;
    }

    const EAACOConfig& getConfig() const {
        return cfg_;
    }

    /**
     * @brief Set obstacles in the environment.
     */
    void setObstacles(const std::vector<SphereObstacle>& spheres,
                      const std::vector<CylinderObstacle>& cylinders,
                      const std::vector<CuboidObstacle>& cuboids = {}) {
        spheres_ = spheres;
        cylinders_ = cylinders;
        cuboids_ = cuboids;
    }

    void setCuboidObstacles(const std::vector<CuboidObstacle>& cuboids) {
        cuboids_ = cuboids;
    }

    /**
     * @brief Query minimal distance from a 3D point to all static obstacles.
     */
    double computeObstacleClearance(const Vector3d& pt) const {
        double min_dist = std::numeric_limits<double>::infinity();
        for (const auto& s : spheres_) {
            min_dist = std::min(min_dist, s.signedDistance(pt));
        }
        for (const auto& c : cylinders_) {
            min_dist = std::min(min_dist, c.signedDistance(pt));
        }
        for (const auto& cb : cuboids_) {
            min_dist = std::min(min_dist, cb.signedDistance(pt));
        }
        return min_dist;
    }

    // Obstacle clearance heuristic factor
    double computeObstacleFactor(double dist_obs) const {
        if (!cfg_.enable_obstacle_factor) return 1.0;
        if (dist_obs <= cfg_.R1) return 0.0;
        if (dist_obs >= cfg_.R2) return 1.0;
        return (dist_obs - cfg_.R1) / (cfg_.R2 - cfg_.R1);
    }

    // Path turning factor
    double computeTurningFactor(const Vector3d& prev_vec, const Vector3d& next_vec) const {
        if (!cfg_.enable_turning_factor) return 1.0;
        double norm_p = prev_vec.norm();
        double norm_n = next_vec.norm();
        if (norm_p < 1e-6 || norm_n < 1e-6) return 1.0;

        double cos_theta = prev_vec.dot(next_vec) / (norm_p * norm_n);
        cos_theta = std::clamp(cos_theta, -1.0, 1.0);
        double delta_psi = std::acos(cos_theta);

        if (delta_psi > cfg_.psi_max) return 0.0;
        return std::max(0.0, 1.0 - delta_psi / cfg_.psi_max);
    }

    // Edge-level environment-weighted scaling factor
    double computePheromoneEnvWeight(double gamma_val) const {
        if (!cfg_.enable_env_pheromone_weight) return 1.0;
        if (gamma_val <= 1e-6) return 0.0;
        return std::exp(1.0 - 1.0 / gamma_val);
    }

    // Execute global path planning
    GlobalPlanResult plan(const Vector3d& start, const Vector3d& goal) {
        auto t_start = std::chrono::steady_clock::now();
        GlobalPlanResult res;

        // Discretized lattice directions (26-connectivity)
        std::vector<Vector3d> directions;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    directions.emplace_back(dx * cfg_.resolution, dy * cfg_.resolution, dz * cfg_.resolution);
                }
            }
        }

        // Pheromone table: hash key -> pheromone level
        std::unordered_map<uint64_t, double> pheromones;
        auto edge_hash = [](uint64_t from, uint64_t to) -> uint64_t {
            uint64_t h = from;
            h ^= to + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        };
        auto pt_to_id = [this](const Vector3d& p) -> uint64_t {
            uint64_t ix = static_cast<uint64_t>(static_cast<int64_t>(std::round(p.x() / cfg_.resolution)) & 0x1FFFFF);
            uint64_t iy = static_cast<uint64_t>(static_cast<int64_t>(std::round(p.y() / cfg_.resolution)) & 0x1FFFFF);
            uint64_t iz = static_cast<uint64_t>(static_cast<int64_t>(std::round(p.z() / cfg_.resolution)) & 0x1FFFFF);
            return ix | (iy << 21) | (iz << 42);
        };

        double best_length = std::numeric_limits<double>::infinity();
        Path3D best_path;

        for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
            std::vector<Path3D> ant_paths(cfg_.num_ants);
            std::vector<double> ant_lengths(cfg_.num_ants, 0.0);
            std::vector<double> ant_gammas(cfg_.num_ants, 1.0);
            std::vector<bool> ant_success(cfg_.num_ants, false);

            for (int k = 0; k < cfg_.num_ants; ++k) {
                Vector3d curr = start;
                Vector3d prev_dir = (goal - start).normalized();
                Path3D path = {curr};
                std::unordered_set<uint64_t> visited;
                visited.insert(pt_to_id(curr));
                double length = 0.0;
                double avg_gamma = 0.0;
                int step = 0;

                while ((curr - goal).norm() > cfg_.resolution && step < cfg_.step_max) {
                    step++;
                    std::vector<Vector3d> allowed_nodes;
                    std::vector<double> probabilities;

                    for (const auto& d : directions) {
                        Vector3d next_pt = curr + d;
                        if (next_pt.x() < cfg_.map_min.x() || next_pt.x() > cfg_.map_max.x() ||
                            next_pt.y() < cfg_.map_min.y() || next_pt.y() > cfg_.map_max.y() ||
                            next_pt.z() < cfg_.map_min.z() || next_pt.z() > cfg_.map_max.z()) {
                            continue;
                        }

                        uint64_t next_id = pt_to_id(next_pt);
                        if (visited.count(next_id)) continue;

                        double d_obs = computeObstacleClearance(next_pt);
                        if (d_obs <= cfg_.R1) continue;

                        double f_d = computeObstacleFactor(d_obs);
                        double f_psi = computeTurningFactor(prev_dir, d);
                        double gamma = f_d * f_psi;
                        if (gamma < 1e-4) continue;

                        double dist_to_goal = (next_pt - goal).norm();
                        double eta = 1.0 / std::max(dist_to_goal, 1e-2);

                        uint64_t e_id = edge_hash(pt_to_id(curr), next_id);
                        double tau = pheromones.count(e_id) ? pheromones[e_id] : cfg_.initial_pheromone;

                        double prob = std::pow(tau, cfg_.alpha) *
                                      std::pow(eta, cfg_.beta) *
                                      std::pow(gamma, cfg_.mu);

                        allowed_nodes.push_back(next_pt);
                        probabilities.push_back(prob);
                    }

                    if (allowed_nodes.empty()) break;

                    // Roulette wheel selection
                    double sum_prob = 0.0;
                    for (double p : probabilities) sum_prob += p;
                    std::uniform_real_distribution<double> dist_roll(0.0, sum_prob);
                    double pick = dist_roll(rng_);

                    Vector3d chosen = allowed_nodes.back();
                    double acc = 0.0;
                    for (size_t idx = 0; idx < allowed_nodes.size(); ++idx) {
                        acc += probabilities[idx];
                        if (pick <= acc) {
                            chosen = allowed_nodes[idx];
                            break;
                        }
                    }

                    prev_dir = (chosen - curr).normalized();
                    length += (chosen - curr).norm();
                    curr = chosen;
                    path.push_back(curr);
                    visited.insert(pt_to_id(curr));
                }

                if ((curr - goal).norm() <= cfg_.resolution * 1.5) {
                    path.push_back(goal);
                    length += (goal - curr).norm();
                    ant_success[k] = true;
                    ant_paths[k] = path;
                    ant_lengths[k] = length;

                    if (length < best_length) {
                        best_length = length;
                        best_path = path;
                    }
                }
            }

            // Pheromone evaporation
            for (auto& pair : pheromones) {
                pair.second *= (1.0 - cfg_.rho);
                pair.second = std::clamp(pair.second, cfg_.min_pheromone, cfg_.max_pheromone);
            }

            // Pheromone deposit with environmental weighting
            for (int k = 0; k < cfg_.num_ants; ++k) {
                if (!ant_success[k]) continue;
                double delta_tau = cfg_.Q / ant_lengths[k];
                const auto& path = ant_paths[k];

                for (size_t i = 0; i + 1 < path.size(); ++i) {
                    Vector3d p1 = path[i];
                    Vector3d p2 = path[i + 1];
                    double d_obs = computeObstacleClearance(0.5 * (p1 + p2));
                    double f_d = computeObstacleFactor(d_obs);
                    Vector3d prev_vec = (i == 0) ? (p2 - p1) : (p1 - path[i - 1]);
                    double f_psi = computeTurningFactor(prev_vec, p2 - p1);
                    double gamma = f_d * f_psi;

                    double f_gamma = computePheromoneEnvWeight(gamma);
                    uint64_t e_id = edge_hash(pt_to_id(p1), pt_to_id(p2));
                    pheromones[e_id] += f_gamma * delta_tau;
                }
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        res.solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (!best_path.empty()) {
            res.success = true;
            res.path = best_path;
            res.path_length = best_length;

            // Metrics evaluation
            double min_c = std::numeric_limits<double>::infinity();
            for (const auto& pt : best_path) {
                min_c = std::min(min_c, computeObstacleClearance(pt));
            }
            res.min_clearance = min_c;

            int turns = 0;
            for (size_t i = 1; i + 1 < best_path.size(); ++i) {
                Vector3d v1 = (best_path[i] - best_path[i - 1]).normalized();
                Vector3d v2 = (best_path[i + 1] - best_path[i]).normalized();
                if (v1.dot(v2) < 0.95) turns++;
            }
            res.num_turns = turns;
        }

        return res;
    }

private:
    EAACOConfig cfg_;
    std::mt19937 rng_;
    std::vector<SphereObstacle> spheres_;
    std::vector<CylinderObstacle> cylinders_;
    std::vector<CuboidObstacle> cuboids_;
};

} // namespace hea_planner

#endif // HEA_PLANNER_EA_ACO_HPP_
