#ifndef HEA_PLANNER_STC_MINCO_HPP_
#define HEA_PLANNER_STC_MINCO_HPP_

#include "minco_s3nu.hpp"
#include "lbfgs_optimizer.hpp"
#include <vector>
#include <cmath>

namespace hea_planner {

// Dynamic target motion model (e.g. moving platform)
struct TargetMotionModel {
    Vector3d p0 = Vector3d::Zero();
    Vector3d v0 = Vector3d(0.5, 0.0, 0.0);
    Vector3d a0 = Vector3d::Zero();
    Vector3d hover_offset = Vector3d(0.0, 0.0, 0.5);

    // Compute target PVA state at arrival time tau
    Eigen::Matrix3d evaluateState(double tau) const {
        Eigen::Matrix3d s;
        s.row(0) = p0 + v0 * tau + 0.5 * a0 * tau * tau + hover_offset;
        s.row(1) = v0 + a0 * tau;
        s.row(2) = a0;
        return s;
    }

    // Time derivative of the target state
    Eigen::Matrix3d evaluateStateDerivative(double tau) const {
        Eigen::Matrix3d s_dot;
        s_dot.row(0) = v0 + a0 * tau;
        s_dot.row(1) = a0;
        s_dot.row(2) = Vector3d::Zero();
        return s_dot;
    }
};

struct STCMincoConfig {
    double w_time = 10.0;             ///< Weight for arrival time
    double w_guide = 10.0;            ///< Weight for spatial guidance penalty
    double w_dyn = 80.0;              ///< Weight for velocity and acceleration penalties
    double max_vel = 2.0;             ///< Maximum permissible velocity (m/s)
    double max_acc = 3.0;             ///< Maximum permissible acceleration (m/s^2)
    int max_iters = 80;               ///< Maximum L-BFGS iterations
    int quadrature_points = 6;        ///< Numerical integration points per segment
    double grad_tolerance = 1e-4;     ///< L-BFGS gradient tolerance
    double min_segment_time = 0.05;   ///< Lower limit on segment flight time (s)
    double max_segment_time = 2.0;    ///< Upper limit on segment flight time (s)
    double terminal_position_tolerance = 0.20; ///< Position error tolerance (m)
    double terminal_velocity_tolerance = 0.10; ///< Velocity error tolerance (m/s)
    double handover_relative_altitude = 0.50;  ///< Hovering height offset (m)

    bool fix_arrival_time = false;         ///< Fixed arrival time toggle
    bool enable_total_derivative = true;   ///< Enable total derivative coupling
};

struct LandingHandoverResult {
    bool success = false;
    double total_duration = 0.0;
    double handover_pos_error = 0.0;   ///< e_p = ||p_uav - p_target|| at T_Sigma (m)
    double handover_vel_error = 0.0;   ///< e_v = ||v_uav - v_target|| at T_Sigma (m/s)
    double max_vel = 0.0;
    double max_acc = 0.0;
    double solve_time_ms = 0.0;
    std::vector<TrajectoryState> sampled_states;

    Eigen::MatrixXd coeffs;
    Eigen::VectorXd durations;

    Vector3d evalPos(double t) const {
        if (durations.size() == 0) return Vector3d::Zero();
        if (t <= 0.0) return coeffs.block<1, 3>(0, 0).transpose();
        double t_accum = 0.0;
        int M = durations.size();
        for (int j = 0; j < M; ++j) {
            double T = durations(j);
            if (t <= t_accum + T || j == M - 1) {
                double tau = std::clamp(t - t_accum, 0.0, T);
                double tau2 = tau * tau, tau3 = tau2 * tau, tau4 = tau3 * tau, tau5 = tau4 * tau;
                Eigen::RowVectorXd basis(6);
                basis << 1.0, tau, tau2, tau3, tau4, tau5;
                return (basis * coeffs.block<6, 3>(6 * j, 0)).transpose();
            }
            t_accum += T;
        }
        return coeffs.block<1, 3>(6 * M - 6, 0).transpose();
    }

    Vector3d evalVel(double t) const {
        if (durations.size() == 0) return Vector3d::Zero();
        double t_accum = 0.0;
        int M = durations.size();
        for (int j = 0; j < M; ++j) {
            double T = durations(j);
            if (t <= t_accum + T || j == M - 1) {
                double tau = std::clamp(t - t_accum, 0.0, T);
                double tau2 = tau * tau, tau3 = tau2 * tau, tau4 = tau3 * tau;
                Eigen::RowVectorXd basis(6);
                basis << 0.0, 1.0, 2.0*tau, 3.0*tau2, 4.0*tau3, 5.0*tau4;
                return (basis * coeffs.block<6, 3>(6 * j, 0)).transpose();
            }
            t_accum += T;
        }
        return Vector3d::Zero();
    }

    Vector3d evalAcc(double t) const {
        if (durations.size() == 0) return Vector3d::Zero();
        double t_accum = 0.0;
        int M = durations.size();
        for (int j = 0; j < M; ++j) {
            double T = durations(j);
            if (t <= t_accum + T || j == M - 1) {
                double tau = std::clamp(t - t_accum, 0.0, T);
                double tau2 = tau * tau;
                Eigen::RowVectorXd basis(6);
                basis << 0.0, 0.0, 2.0, 6.0*tau, 12.0*tau2, 20.0*(tau2*tau);
                return (basis * coeffs.block<6, 3>(6 * j, 0)).transpose();
            }
            t_accum += T;
        }
        return Vector3d::Zero();
    }

    Vector3d evalJerk(double t) const {
        if (durations.size() == 0) return Vector3d::Zero();
        double t_accum = 0.0;
        int M = durations.size();
        for (int j = 0; j < M; ++j) {
            double T = durations(j);
            if (t <= t_accum + T || j == M - 1) {
                double tau = std::clamp(t - t_accum, 0.0, T);
                Eigen::RowVectorXd basis(6);
                basis << 0.0, 0.0, 0.0, 6.0, 24.0*tau, 60.0*(tau*tau);
                return (basis * coeffs.block<6, 3>(6 * j, 0)).transpose();
            }
            t_accum += T;
        }
        return Vector3d::Zero();
    }
};

class STCMincoOptimizer {
public:
    explicit STCMincoOptimizer(const STCMincoConfig& cfg = STCMincoConfig())
        : cfg_(cfg) {}

    void setConfig(const STCMincoConfig& cfg) {
        cfg_ = cfg;
    }

    const STCMincoConfig& getConfig() const {
        return cfg_;
    }

    // Optimize landing trajectory towards a moving target platform
    LandingHandoverResult optimize(const Eigen::Matrix3d& s0,
                                  const TargetMotionModel& target_model,
                                  const Eigen::MatrixXd& ref_waypoints,
                                  const std::vector<AABBBox>& corridor_boxes,
                                  const Eigen::VectorXd& initial_durations) {
        auto t_start = std::chrono::steady_clock::now();
        LandingHandoverResult res;

        int M = initial_durations.size();
        int num_mid_pts = M - 1;
        int n_vars = cfg_.fix_arrival_time ? (3 * num_mid_pts) : (3 * num_mid_pts + M);

        Eigen::MatrixXd d_box(3, num_mid_pts);
        Eigen::MatrixXd P_ref(3, num_mid_pts);
        for (int i = 0; i < num_mid_pts; ++i) {
            d_box.col(i) = corridor_boxes[i].radii().cwiseMax(0.2);
            P_ref.col(i) = corridor_boxes[i].center();
        }

        // Initialize x = [xi; tau]
        Eigen::VectorXd x(n_vars);
        for (int i = 0; i < num_mid_pts; ++i) {
            Vector3d delta = (ref_waypoints.col(i) - P_ref.col(i)).cwiseQuotient(d_box.col(i));
            for (int dim = 0; dim < 3; ++dim) {
                delta(dim) = std::clamp(delta(dim), -0.95, 0.95);
                x(3 * i + dim) = 0.5 * std::log((1.0 + delta(dim)) / (1.0 - delta(dim)));
            }
        }
        if (!cfg_.fix_arrival_time) {
            for (int j = 0; j < M; ++j) {
                x(3 * num_mid_pts + j) = std::log(std::max(initial_durations(j), 0.1));
            }
        }

        MincoS3NU minco;
        minco.reset(M);

        auto cost_function = [&](const Eigen::VectorXd& cur_x, Eigen::VectorXd& cur_grad) -> double {
            cur_grad.resize(n_vars);
            cur_grad.setZero();

            // 1. Recover waypoints q
            Eigen::MatrixXd q(3, num_mid_pts);
            Eigen::MatrixXd dq_dxi(3, num_mid_pts);
            for (int i = 0; i < num_mid_pts; ++i) {
                for (int dim = 0; dim < 3; ++dim) {
                    double xi = cur_x(3 * i + dim);
                    double th = std::tanh(xi);
                    q(dim, i) = P_ref(dim, i) + d_box(dim, i) * th;
                    dq_dxi(dim, i) = d_box(dim, i) * (1.0 - th * th);
                }
            }

            // 2. Recover durations T
            Eigen::VectorXd T(M);
            Eigen::VectorXd dT_dtau(M);
            if (cfg_.fix_arrival_time) {
                T = initial_durations;
            } else {
                double tau_min = std::log(std::max(1e-4, cfg_.min_segment_time));
                double tau_max = std::log(std::max(0.5, cfg_.max_segment_time));
                for (int j = 0; j < M; ++j) {
                    double tau = cur_x(3 * num_mid_pts + j);
                    tau = std::clamp(tau, tau_min, tau_max);
                    T(j) = std::exp(tau);
                    dT_dtau(j) = T(j);
                }
            }

            double T_sigma = T.sum();

            // 3. Evaluate time-coupled target rendezvous state s_f(T_Sigma)
            Eigen::Matrix3d sf = target_model.evaluateState(T_sigma);
            Eigen::Matrix3d s_dot = target_model.evaluateStateDerivative(T_sigma);

            // 4. Generate MINCO trajectory
            minco.generate(s0, sf, q, T);

            double J_jerk = minco.computeJerkEnergy();
            double J_time = cfg_.w_time * T_sigma;

            Eigen::MatrixXd grad_q;
            Eigen::VectorXd grad_T;
            Eigen::Matrix3d grad_sf;
            minco.computeJerkGradients(grad_q, grad_T, grad_sf);

            // 5. Spatial guidance penalty
            double J_guide = 0.0;
            if (cfg_.w_guide > 1e-6) {
                for (int i = 0; i < num_mid_pts; ++i) {
                    Vector3d err = q.col(i) - P_ref.col(i);
                    J_guide += cfg_.w_guide * err.squaredNorm();
                    grad_q.col(i) += 2.0 * cfg_.w_guide * err;
                }
            }

            // 6. Kinodynamic penalty and analytical duration gradient
            double J_dyn = 0.0;
            Eigen::VectorXd grad_T_dyn = Eigen::VectorXd::Zero(M);
            double v_max_sq = cfg_.max_vel * cfg_.max_vel;
            double a_max_sq = cfg_.max_acc * cfg_.max_acc;
            int K = cfg_.quadrature_points;
            for (int j = 0; j < M; ++j) {
                double Tj = std::max(T(j), 1e-3);
                double dt = Tj / static_cast<double>(K);
                for (int k = 0; k <= K; ++k) {
                    double tau = k * dt;
                    Eigen::RowVectorXd bv(6), ba(6);
                    bv << 0.0, 1.0, 2.0*tau, 3.0*tau*tau, 4.0*tau*tau*tau, 5.0*tau*tau*tau*tau;
                    ba << 0.0, 0.0, 2.0, 6.0*tau, 12.0*tau*tau, 20.0*tau*tau*tau;
                    Eigen::Matrix<double, 6, 3> Cj = minco.getCoeffs().block<6, 3>(6 * j, 0);
                    Vector3d vel = (bv * Cj).transpose();
                    Vector3d acc = (ba * Cj).transpose();

                    double v_sq = vel.squaredNorm();
                    double a_sq = acc.squaredNorm();

                    double v_viol = v_sq - v_max_sq;
                    if (v_viol > 0.0) {
                        double pen_v = cfg_.w_dyn * v_viol * v_viol;
                        J_dyn += pen_v * dt;
                        // Time dilation derivative: dt component + scale sensitivity
                        grad_T_dyn(j) += (pen_v / static_cast<double>(K)) - (4.0 * cfg_.w_dyn * dt / Tj) * v_viol * v_sq;
                    }

                    double a_viol = a_sq - a_max_sq;
                    if (a_viol > 0.0) {
                        double pen_a = cfg_.w_dyn * a_viol * a_viol;
                        J_dyn += pen_a * dt;
                        // Time dilation derivative: dt component + quadratic acceleration expansion damping
                        grad_T_dyn(j) += (pen_a / static_cast<double>(K)) - (8.0 * cfg_.w_dyn * dt / Tj) * a_viol * a_sq;
                    }
                }
                // Numerical stability: softly clamp duration gradient to prevent L-BFGS step explosion
                grad_T_dyn(j) = std::clamp(grad_T_dyn(j), -100.0, 100.0);
            }

            // 7. TOTAL DERIVATIVE CHAIN RULE (Eq. 27 & Appendix B)
            // dL/dT_i = native_grad_T + (grad_sf : s_dot) + w_time + grad_T_dyn
            if (!cfg_.fix_arrival_time) {
                double target_coupling = 0.0;
                if (cfg_.enable_total_derivative) {
                    // Inner Frobenius product: sum_m (dL/ds_f,m * ds_f,m/dT_Sigma)
                    target_coupling = (grad_sf.array() * s_dot.array()).sum();
                }

                for (int j = 0; j < M; ++j) {
                    double total_dJ_dT = grad_T(j) + target_coupling + cfg_.w_time + grad_T_dyn(j);
                    cur_grad(3 * num_mid_pts + j) = total_dJ_dT * dT_dtau(j);
                }
            }

            // Waypoint gradients
            for (int i = 0; i < num_mid_pts; ++i) {
                for (int dim = 0; dim < 3; ++dim) {
                    cur_grad(3 * i + dim) = grad_q(dim, i) * dq_dxi(dim, i);
                }
            }

            return J_jerk + J_time + J_guide + J_dyn;
        };

        LBFGSConfig lbfgs_cfg;
        lbfgs_cfg.max_iterations = cfg_.max_iters;
        lbfgs_cfg.grad_tolerance = cfg_.grad_tolerance;

        double final_cost = 0.0;
        LBFGSOptimizer::optimize(cost_function, x, final_cost, lbfgs_cfg);

        // Re-generate final optimal MINCO trajectory at optimal solution x
        Eigen::MatrixXd opt_q(3, num_mid_pts);
        for (int i = 0; i < num_mid_pts; ++i) {
            for (int dim = 0; dim < 3; ++dim) {
                double xi = x(3 * i + dim);
                double th = std::tanh(xi);
                opt_q(dim, i) = P_ref(dim, i) + d_box(dim, i) * th;
            }
        }
        Eigen::VectorXd opt_T(M);
        if (cfg_.fix_arrival_time) {
            opt_T = initial_durations;
        } else {
            double tau_max = std::log(std::max(0.5, cfg_.max_segment_time));
            for (int j = 0; j < M; ++j) {
                double tau = x(3 * num_mid_pts + j);
                tau = std::clamp(tau, -1.6, tau_max);
                opt_T(j) = std::exp(tau);
            }
        }
        double opt_T_sigma = opt_T.sum();
        Eigen::Matrix3d opt_sf = target_model.evaluateState(opt_T_sigma);
        minco.generate(s0, opt_sf, opt_q, opt_T);

        auto t_end = std::chrono::steady_clock::now();
        res.solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        double total_T = minco.getTotalDuration();
        res.total_duration = total_T;
        res.success = true;

        // Terminal handover accuracy evaluation
        Vector3d final_p_uav = minco.evaluatePosition(total_T);
        Vector3d final_v_uav = minco.evaluateVelocity(total_T);

        Eigen::Matrix3d target_final = target_model.evaluateState(total_T);
        Vector3d final_p_target = target_final.row(0).transpose();
        Vector3d final_v_target = target_final.row(1).transpose();

        res.handover_pos_error = (final_p_uav - final_p_target).norm();
        res.handover_vel_error = (final_v_uav - final_v_target).norm();

        // Sample trajectory
        double max_v = 0.0, max_a = 0.0;
        for (double t = 0.0; t <= total_T; t += 0.05) {
            TrajectoryState state;
            state.t = t;
            state.pos = minco.evaluatePosition(t);
            state.vel = minco.evaluateVelocity(t);
            state.acc = minco.evaluateAcceleration(t);
            max_v = std::max(max_v, state.vel.norm());
            max_a = std::max(max_a, state.acc.norm());
            res.sampled_states.push_back(state);
        }
        res.max_vel = max_v;
        res.max_acc = max_a;
        res.coeffs = minco.getCoeffs();
        res.durations = minco.getDurations();

        return res;
    }

private:
    STCMincoConfig cfg_;
};

} // namespace hea_planner

#endif // HEA_PLANNER_STC_MINCO_HPP_
