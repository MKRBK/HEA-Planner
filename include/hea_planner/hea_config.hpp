#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include <cmath>

#include "hea_planner/common.hpp"
#include "hea_planner/ea_aco.hpp"
#include "hea_planner/ea_astar.hpp"
#include "hea_planner/sg_minco.hpp"
#include "hea_planner/stc_minco.hpp"
#include "hea_planner/lbfgs_optimizer.hpp"

namespace hea_planner {

// Quadrotor physical limitations and mission parameters
struct PhysicalLimitsConfig {
    double max_velocity = 2.0;              ///< Maximum velocity (m/s)
    double max_acceleration = 3.0;          ///< Maximum acceleration (m/s^2)
    double max_thrust_to_weight = 11.96;    ///< Maximum thrust-to-weight ratio (m/s^2)
    double max_tilt_angle_deg = 35.0;       ///< Maximum tilt angle (deg)
    double conservative_safe_radius = 0.20; ///< Conservative safe radius (m)
    double nominal_replan_frequency = 20.0; ///< Replan frequency (Hz)
    double control_loop_dt = 0.05;          ///< Control loop dt (s)
};

// Numerical optimization parameters for L-BFGS
struct NumericalSolverConfig {
    int lbfgs_history_size = 8;
    double gradient_tolerance = 1.0e-4;
    double step_tolerance = 1.0e-6;
    double armijo_c1 = 1.0e-4;
    double backtrack_factor = 0.50;
    int max_line_search_iters = 20;

    double feasibility_tolerance = 1.0e-3;
    double penalty_inflation_ratio = 2.50;
    double violation_reduction_ratio = 0.25;
    int max_outer_iterations = 2;
    double time_variable_tau_min = -4.0;
    double time_variable_tau_max = 3.5;
};

// Unified configuration manager for HEA-Planner
class HEAConfig {
public:
    PhysicalLimitsConfig physical;
    EAACOConfig ea_aco;
    EAAStarConfig ea_astar;
    SGMincoConfig sg_minco;
    STCMincoConfig stc_minco;
    NumericalSolverConfig solver;

    HEAConfig() {
        synchronizePaperDefaults();
    }

    // Synchronize default parameter values
    void synchronizePaperDefaults() {
        // 1. Physical limits (Table 1 & Section 2.3)
        physical.max_velocity = 2.0;
        physical.max_acceleration = 3.0;
        physical.max_thrust_to_weight = 11.96;
        physical.max_tilt_angle_deg = 35.0;
        physical.conservative_safe_radius = 0.20;
        physical.nominal_replan_frequency = 20.0;
        physical.control_loop_dt = 0.05;

        // 2. State S1: EA-ACO (Table 1 & Section 3.2)
        ea_aco.alpha = 2.0;
        ea_aco.beta = 4.0;
        ea_aco.mu = 4.0;
        ea_aco.rho = 0.30;
        ea_aco.num_ants = 30;
        ea_aco.max_iterations = 200;
        ea_aco.Q = 1.0;
        ea_aco.step_max = 500;
        ea_aco.R1 = 0.20;
        ea_aco.R2 = 0.70;
        ea_aco.psi_max = M_PI / 2.0;
        ea_aco.resolution = 0.10;
        ea_aco.enable_obstacle_factor = true;
        ea_aco.enable_turning_factor = true;
        ea_aco.enable_env_pheromone_weight = true;
        ea_aco.initial_pheromone = 1.0;
        ea_aco.min_pheromone = 0.01;
        ea_aco.max_pheromone = 20.0;
        ea_aco.keypoint_angle_cos = 0.99;
        ea_aco.keypoint_min_distance = 0.80;
        ea_aco.map_min = Vector3d(-6.0, -6.0, 0.0);
        ea_aco.map_max = Vector3d(18.0, 22.0, 4.0);

        // 3. State S2: EA-A* (Table 1 & Section 3.3)
        ea_astar.resolution = 0.25;
        ea_astar.nominal_vel = 1.5;
        ea_astar.w_env = 2.0;
        ea_astar.C_risk = 15.0;
        ea_astar.tau_safe = 2.5;
        ea_astar.k_v = 0.60;
        ea_astar.R_safe = 0.20;
        ea_astar.spatial_decay_sigma = 1.0;
        ea_astar.lookahead_distance = 2.0;
        ea_astar.goal_arrival_tolerance = 0.25;
        ea_astar.max_expansions = 1000;
        ea_astar.local_window = Vector3d(5.0, 5.0, 4.0);
        ea_astar.enable_risk_field = true;
        ea_astar.enable_cpa_check = true;
        ea_astar.enable_hard_prune = true;

        // 4. State S1/S2: SG-MINCO (Table 1, Section 3.4 & Appendix A.1)
        sg_minco.w_guide = 30.0;
        sg_minco.rho_time = 20.0;
        sg_minco.max_vel = 2.0;
        sg_minco.max_acc = 3.0;
        sg_minco.r_safe = 0.28; // Tuned safe margin (0.28 m)
        sg_minco.quadrature_points = 12;
        sg_minco.w_corridor = 5000.0;
        sg_minco.w_dyn = 80.0;
        sg_minco.max_iters = 80;
        sg_minco.min_segment_time = 0.02;
        sg_minco.max_segment_time = 2.5;
        sg_minco.grad_tolerance = 1.0e-4;
        sg_minco.enable_spatial_guidance = true;
        sg_minco.enable_corridor_penalty = true;

        // 5. State S3: STC-MINCO (Table 1, Section 3.4 & Appendix A.2)
        stc_minco.w_time = 10.0; // w_t in Table 1
        stc_minco.w_guide = 10.0;
        stc_minco.w_dyn = 80.0;
        stc_minco.max_vel = 2.0;
        stc_minco.max_acc = 3.0;
        stc_minco.max_iters = 80;
        stc_minco.quadrature_points = 6;
        stc_minco.grad_tolerance = 1.0e-4;
        stc_minco.min_segment_time = 0.05;
        stc_minco.max_segment_time = 2.0;
        stc_minco.terminal_position_tolerance = 0.20;
        stc_minco.terminal_velocity_tolerance = 0.10;
        stc_minco.handover_relative_altitude = 0.50;
        stc_minco.enable_total_derivative = true;
        stc_minco.fix_arrival_time = false;

        // 6. Solver & ALM
        solver.lbfgs_history_size = 8;
        solver.gradient_tolerance = 1.0e-4;
        solver.step_tolerance = 1.0e-6;
        solver.armijo_c1 = 1.0e-4;
        solver.backtrack_factor = 0.50;
        solver.max_line_search_iters = 20;
        solver.feasibility_tolerance = 1.0e-3;
        solver.penalty_inflation_ratio = 2.50;
        solver.violation_reduction_ratio = 0.25;
        solver.max_outer_iterations = 2;
        solver.time_variable_tau_min = -4.0;
        solver.time_variable_tau_max = 3.5;
    }

    // Parse key-value formatted YAML parameter file
    bool loadFromFile(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            // Attempt fallback candidate paths
            std::vector<std::string> fallbacks = {
                "config/" + filepath,
                "open_source/" + filepath,
                "open_source/config/" + filepath,
                "../" + filepath,
                "../config/hea_planner_params.yaml",
                "config/hea_planner_params.yaml"
            };
            for (const auto& alt : fallbacks) {
                f.open(alt);
                if (f.is_open()) break;
            }
        }
        if (!f.is_open()) {
            std::cerr << "[HEAConfig] Warning: Could not open config file: " << filepath 
                      << ". Using synchronized paper defaults." << std::endl;
            return false;
        }

        std::string current_section = "";
        std::string line;
        while (std::getline(f, line)) {
            // Trim leading/trailing whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            line = line.substr(start);

            // Ignore comment lines
            if (line[0] == '#') continue;

            // Strip inline comments
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }

            // Check if section header (e.g., "ea_aco:")
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = line.substr(0, colon_pos);
                // trim key
                key.erase(key.find_last_not_of(" \t") + 1);

                std::string val = line.substr(colon_pos + 1);
                size_t val_start = val.find_first_not_of(" \t");
                if (val_start == std::string::npos) {
                    current_section = key;
                    continue;
                }
                val = val.substr(val_start);
                val.erase(val.find_last_not_of(" \t\r\n") + 1);

                applyParameter(current_section, key, val);
            }
        }
        return true;
    }

    void printSummary() const {
        std::cout << "========================================================================\n"
                  << "                  HEA-Planner Configuration Summary                     \n"
                  << "========================================================================\n"
                  << std::fixed << std::setprecision(3)
                  << "[Physical Limits]\n"
                  << "  Max Velocity (v_max):           " << physical.max_velocity << " m/s\n"
                  << "  Max Acceleration (a_max):       " << physical.max_acceleration << " m/s^2\n"
                  << "  Max Thrust/Weight (f_max/m):    " << physical.max_thrust_to_weight << " m/s^2\n"
                  << "  Conservative Safe Radius:       " << physical.conservative_safe_radius << " m\n"
                  << "  Nominal Replan Freq:            " << physical.nominal_replan_frequency << " Hz (" << physical.control_loop_dt << " s)\n"
                  << "[State S1: EA-ACO Global Planner]\n"
                  << "  Pheromone / Dist / Env Factors: alpha=" << ea_aco.alpha << ", beta=" << ea_aco.beta << ", mu=" << ea_aco.mu << "\n"
                  << "  Evaporation (rho) / Constant Q: rho=" << ea_aco.rho << ", Q=" << ea_aco.Q << "\n"
                  << "  Ants (K) / Max Iterations (N):  K=" << ea_aco.num_ants << ", N=" << ea_aco.max_iterations << " (step_max=" << ea_aco.step_max << ")\n"
                  << "  Clearance Thresholds (R1, R2):  R1=" << ea_aco.R1 << " m, R2=" << ea_aco.R2 << " m\n"
                  << "  Max Turn Angle (psi_max):       " << ea_aco.psi_max * 180.0 / M_PI << " deg\n"
                  << "  Map Bounds:                     [" << ea_aco.map_min.transpose() << "] to [" << ea_aco.map_max.transpose() << "]\n"
                  << "[State S2: EA-A* Local Replanner]\n"
                  << "  Env Risk Weight (w_env):        " << ea_astar.w_env << "\n"
                  << "  Risk Repulsion Gain (C_risk):   " << ea_astar.C_risk << "\n"
                  << "  Time Horizon (tau_safe) / kv:   tau=" << ea_astar.tau_safe << " s, kv=" << ea_astar.k_v << "\n"
                  << "  Lookahead / Arrival Tol:        L_ahead=" << ea_astar.lookahead_distance << " m, tol=" << ea_astar.goal_arrival_tolerance << " m\n"
                  << "  Search Budget (N_max):          " << ea_astar.max_expansions << " expansions\n"
                  << "  Local Horizon Window:           [" << ea_astar.local_window.transpose() << "] m\n"
                  << "[State S1 & S2: SG-MINCO Trajectory Optimization]\n"
                  << "  Spatial Guidance Weight:        " << sg_minco.w_guide << "\n"
                  << "  Time Regularizer (rho_time):    " << sg_minco.rho_time << "\n"
                  << "  Corridor Clearance Margin:      " << sg_minco.r_safe << " m (w_corr=" << sg_minco.w_corridor << ")\n"
                  << "  Quadrature Points (K_quad):     " << sg_minco.quadrature_points << " pts/seg\n"
                  << "  Segment Time Bounds:            [" << sg_minco.min_segment_time << ", " << sg_minco.max_segment_time << "] s\n"
                  << "[State S3: STC-MINCO Moving-Platform Landing]\n"
                  << "  Arrival Time Weight (w_t):      " << stc_minco.w_time << "\n"
                  << "  Handover Tolerances:            pos_tol=" << stc_minco.terminal_position_tolerance << " m, vel_tol=" << stc_minco.terminal_velocity_tolerance << " m/s\n"
                  << "  Handover Altitude (Delta z):    " << stc_minco.handover_relative_altitude << " m\n"
                  << "  Analytical Total Derivative:    " << (stc_minco.enable_total_derivative ? "ENABLED (Eq. 27)" : "DISABLED") << "\n"
                  << "========================================================================" << std::endl;
    }

private:
    void applyParameter(const std::string& section, const std::string& key, const std::string& val) {
        auto parseDouble = [](const std::string& s) -> double {
            try { return std::stod(s); } catch (...) { return 0.0; }
        };
        auto parseInt = [](const std::string& s) -> int {
            try { return std::stoi(s); } catch (...) { return 0; }
        };
        auto parseBool = [](const std::string& s) -> bool {
            return (s == "true" || s == "1" || s == "True" || s == "TRUE");
        };
        auto parseVector3d = [](const std::string& s) -> Vector3d {
            Vector3d vec = Vector3d::Zero();
            std::string cleaned = s;
            for (char& c : cleaned) {
                if (c == '[' || c == ']' || c == ',') c = ' ';
            }
            std::stringstream ss(cleaned);
            double x = 0.0, y = 0.0, z = 0.0;
            if (ss >> x) vec.x() = x;
            if (ss >> y) vec.y() = y;
            if (ss >> z) vec.z() = z;
            return vec;
        };

        if (section == "physical_limits") {
            if (key == "max_velocity") physical.max_velocity = parseDouble(val);
            else if (key == "max_acceleration") physical.max_acceleration = parseDouble(val);
            else if (key == "max_thrust_to_weight") physical.max_thrust_to_weight = parseDouble(val);
            else if (key == "max_tilt_angle_deg") physical.max_tilt_angle_deg = parseDouble(val);
            else if (key == "conservative_safe_radius") physical.conservative_safe_radius = parseDouble(val);
            else if (key == "nominal_replan_frequency") physical.nominal_replan_frequency = parseDouble(val);
            else if (key == "control_loop_dt") physical.control_loop_dt = parseDouble(val);
        } else if (section == "ea_aco") {
            if (key == "alpha") ea_aco.alpha = parseDouble(val);
            else if (key == "beta") ea_aco.beta = parseDouble(val);
            else if (key == "mu") ea_aco.mu = parseDouble(val);
            else if (key == "evaporation_rate") ea_aco.rho = parseDouble(val);
            else if (key == "num_ants") ea_aco.num_ants = parseInt(val);
            else if (key == "max_iterations") ea_aco.max_iterations = parseInt(val);
            else if (key == "pheromone_constant") ea_aco.Q = parseDouble(val);
            else if (key == "max_steps") ea_aco.step_max = parseInt(val);
            else if (key == "hard_safety_distance") ea_aco.R1 = parseDouble(val);
            else if (key == "soft_safety_distance") ea_aco.R2 = parseDouble(val);
            else if (key == "max_turn_angle_deg") ea_aco.psi_max = parseDouble(val) * M_PI / 180.0;
            else if (key == "grid_resolution") ea_aco.resolution = parseDouble(val);
            else if (key == "initial_pheromone") ea_aco.initial_pheromone = parseDouble(val);
            else if (key == "min_pheromone") ea_aco.min_pheromone = parseDouble(val);
            else if (key == "max_pheromone") ea_aco.max_pheromone = parseDouble(val);
            else if (key == "keypoint_angle_cos") ea_aco.keypoint_angle_cos = parseDouble(val);
            else if (key == "keypoint_min_distance") ea_aco.keypoint_min_distance = parseDouble(val);
            else if (key == "map_bound_min") ea_aco.map_min = parseVector3d(val);
            else if (key == "map_bound_max") ea_aco.map_max = parseVector3d(val);
        } else if (section == "ea_astar") {
            if (key == "env_risk_weight") ea_astar.w_env = parseDouble(val);
            else if (key == "risk_repulsion_gain") ea_astar.C_risk = parseDouble(val);
            else if (key == "spatial_decay_sigma") ea_astar.spatial_decay_sigma = parseDouble(val);
            else if (key == "velocity_expansion_kv") ea_astar.k_v = parseDouble(val);
            else if (key == "collision_time_horizon") ea_astar.tau_safe = parseDouble(val);
            else if (key == "lookahead_distance") ea_astar.lookahead_distance = parseDouble(val);
            else if (key == "max_expansions") ea_astar.max_expansions = parseInt(val);
            else if (key == "local_window_size") ea_astar.local_window = parseVector3d(val);
            else if (key == "nominal_velocity") ea_astar.nominal_vel = parseDouble(val);
            else if (key == "goal_arrival_tolerance") ea_astar.goal_arrival_tolerance = parseDouble(val);
            else if (key == "enable_hard_prune") ea_astar.enable_hard_prune = parseBool(val);
            else if (key == "enable_cpa_check") ea_astar.enable_cpa_check = parseBool(val);
            else if (key == "local_grid_resolution") ea_astar.resolution = parseDouble(val);
        } else if (section == "sg_minco") {
            if (key == "spatial_guidance_weight") sg_minco.w_guide = parseDouble(val);
            else if (key == "time_regularization_weight") sg_minco.rho_time = parseDouble(val);
            else if (key == "corridor_safety_radius") sg_minco.r_safe = parseDouble(val);
            else if (key == "quadrature_points") sg_minco.quadrature_points = parseInt(val);
            else if (key == "continuous_corridor_penalty_weight") sg_minco.w_corridor = parseDouble(val);
            else if (key == "dynamic_penalty_weight") sg_minco.w_dyn = parseDouble(val);
            else if (key == "max_iterations") sg_minco.max_iters = parseInt(val);
            else if (key == "min_segment_time") sg_minco.min_segment_time = parseDouble(val);
            else if (key == "max_segment_time") sg_minco.max_segment_time = parseDouble(val);
            else if (key == "grad_tolerance") sg_minco.grad_tolerance = parseDouble(val);
        } else if (section == "stc_minco") {
            if (key == "arrival_time_weight") stc_minco.w_time = parseDouble(val);
            else if (key == "enable_total_derivative") stc_minco.enable_total_derivative = parseBool(val);
            else if (key == "terminal_position_tolerance") stc_minco.terminal_position_tolerance = parseDouble(val);
            else if (key == "terminal_velocity_tolerance") stc_minco.terminal_velocity_tolerance = parseDouble(val);
            else if (key == "handover_relative_altitude") stc_minco.handover_relative_altitude = parseDouble(val);
            else if (key == "spatial_guidance_weight") stc_minco.w_guide = parseDouble(val);
            else if (key == "dynamic_penalty_weight") stc_minco.w_dyn = parseDouble(val);
            else if (key == "max_iterations") stc_minco.max_iters = parseInt(val);
            else if (key == "max_segment_time") stc_minco.max_segment_time = parseDouble(val);
            else if (key == "quadrature_points") stc_minco.quadrature_points = parseInt(val);
            else if (key == "grad_tolerance") stc_minco.grad_tolerance = parseDouble(val);
        } else if (section == "numerical_solver") {
            if (key == "lbfgs_history_size") solver.lbfgs_history_size = parseInt(val);
            else if (key == "gradient_tolerance") solver.gradient_tolerance = parseDouble(val);
            else if (key == "step_tolerance") solver.step_tolerance = parseDouble(val);
            else if (key == "armijo_c1") solver.armijo_c1 = parseDouble(val);
            else if (key == "backtrack_factor") solver.backtrack_factor = parseDouble(val);
            else if (key == "max_line_search_iters") solver.max_line_search_iters = parseInt(val);
            else if (key == "feasibility_tolerance") solver.feasibility_tolerance = parseDouble(val);
            else if (key == "penalty_inflation_ratio") solver.penalty_inflation_ratio = parseDouble(val);
            else if (key == "violation_reduction_ratio") solver.violation_reduction_ratio = parseDouble(val);
            else if (key == "max_outer_iterations") solver.max_outer_iterations = parseInt(val);
            else if (key == "time_variable_tau_min") solver.time_variable_tau_min = parseDouble(val);
            else if (key == "time_variable_tau_max") solver.time_variable_tau_max = parseDouble(val);
        }
    }
};

} // namespace hea_planner
