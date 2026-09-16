#ifndef HEA_PLANNER_SG_MINCO_HPP_
#define HEA_PLANNER_SG_MINCO_HPP_

#include "minco_s3nu.hpp"
#include "lbfgs_optimizer.hpp"
#include <vector>
#include <cmath>

namespace hea_planner {

struct SGMincoConfig {
    double rho_time = 20.0;           ///< Weight for total flight duration
    double w_guide = 30.0;            ///< Weight for spatial guidance penalty
    double w_dyn = 80.0;              ///< Weight for dynamic constraint penalty
    double max_vel = 2.0;             ///< Maximum permissible velocity (m/s)
    double max_acc = 3.0;             ///< Maximum permissible acceleration (m/s^2)
    int max_iters = 80;               ///< Maximum L-BFGS iterations
    int quadrature_points = 12;       ///< Quadrature points per segment

    bool enable_spatial_guidance = true; ///< Enable spatial guidance penalty
    bool enable_corridor_penalty = true; ///< Enable continuous corridor penalty
    double w_corridor = 5000.0;          ///< Weight for corridor penetration penalty
    double r_safe = 0.28;                ///< Safety margin threshold (m)

    double min_segment_time = 0.02;      ///< Lower bound on segment duration (s)
    double max_segment_time = 2.5;       ///< Upper bound on segment duration (s)
    double grad_tolerance = 1e-4;        ///< L-BFGS gradient tolerance
};

struct TrajectoryResult {
    bool success = false;
    double total_duration = 0.0;
    double jerk_energy = 0.0;
    double max_vel = 0.0;
    double max_acc = 0.0;
    double max_dist_to_center = 0.0;
    double solve_time_ms = 0.0;
    int iterations = 0;
    double final_cost = 0.0;
    std::vector<TrajectoryState> sampled_states;
};

class SGMincoOptimizer {
public:
    using ClearanceFunction = std::function<double(const Vector3d&)>;
    using ClearanceGradFunction = std::function<Vector3d(const Vector3d&)>;

    explicit SGMincoOptimizer(const SGMincoConfig& cfg = SGMincoConfig())
        : cfg_(cfg) {}

    void setConfig(const SGMincoConfig& cfg) {
        cfg_ = cfg;
    }

    const SGMincoConfig& getConfig() const {
        return cfg_;
    }

    // Optimize 3D trajectory within corridor boxes
    TrajectoryResult optimize(const Eigen::Matrix3d& s0,
                              const Eigen::Matrix3d& sf,
                              const Eigen::MatrixXd& ref_waypoints,
                              const std::vector<AABBBox>& corridor_boxes,
                              const Eigen::VectorXd& initial_durations,
                              const ClearanceFunction& clearance_fn = nullptr,
                              const ClearanceGradFunction& clearance_grad_fn = nullptr) {
        auto t_start = std::chrono::steady_clock::now();
        TrajectoryResult res;

        int M = initial_durations.size();
        int num_mid_pts = M - 1;
        int n_vars = 3 * num_mid_pts + M;

        // Extract AABB half-lengths d_box and reference centers
        Eigen::MatrixXd d_box(3, num_mid_pts);
        Eigen::MatrixXd P_ref(3, num_mid_pts);
        for (int i = 0; i < num_mid_pts; ++i) {
            d_box.col(i) = corridor_boxes[i].radii().cwiseMax(0.1);
            P_ref.col(i) = corridor_boxes[i].center();
        }

        // Initialize unconstrained optimization variables x = [xi; tau]
        // xi = atanh((q - P_ref) / d_box)
        Eigen::VectorXd x(n_vars);
        for (int i = 0; i < num_mid_pts; ++i) {
            Vector3d delta = (ref_waypoints.col(i) - P_ref.col(i)).cwiseQuotient(d_box.col(i));
            for (int dim = 0; dim < 3; ++dim) {
                delta(dim) = std::clamp(delta(dim), -0.95, 0.95);
                x(3 * i + dim) = 0.5 * std::log((1.0 + delta(dim)) / (1.0 - delta(dim)));
            }
        }
        for (int j = 0; j < M; ++j) {
            x(3 * num_mid_pts + j) = std::log(std::max(initial_durations(j), 0.1));
        }

        MincoS3NU minco;
        minco.reset(M);

        auto eval_grad = [&](const Vector3d& pos) -> Vector3d {
            if (clearance_grad_fn) return clearance_grad_fn(pos);
            Vector3d grad;
            const double eps = 1e-4;
            for (int dim = 0; dim < 3; ++dim) {
                Vector3d p_plus = pos, p_minus = pos;
                p_plus(dim) += eps;
                p_minus(dim) -= eps;
                grad(dim) = (clearance_fn(p_plus) - clearance_fn(p_minus)) / (2.0 * eps);
            }
            return grad;
        };

        // Define cost function and analytic gradient evaluation
        auto cost_function = [&](const Eigen::VectorXd& cur_x, Eigen::VectorXd& cur_grad) -> double {
            cur_grad.resize(n_vars);
            cur_grad.setZero();

            // 1. Recover waypoints q and durations T via diffeomorphisms
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

            Eigen::VectorXd T(M);
            Eigen::VectorXd dT_dtau(M);
            double tau_min = std::log(std::max(1e-4, cfg_.min_segment_time));
            double tau_max = std::log(std::max(0.5, cfg_.max_segment_time));
            for (int j = 0; j < M; ++j) {
                double tau = std::clamp(cur_x(3 * num_mid_pts + j), tau_min, tau_max);
                T(j) = std::exp(tau);
                dT_dtau(j) = T(j);
            }

            // 2. Solve linear system A*C = B
            minco.generate(s0, sf, q, T);

            // Jerk energy and time regularization
            double J_jerk = minco.computeJerkEnergy();
            double J_time = cfg_.rho_time * T.sum();

            // Spatial guidance penalty J_guide
            double J_guide = 0.0;
            Eigen::MatrixXd grad_q_guide = Eigen::MatrixXd::Zero(3, num_mid_pts);
            if (cfg_.enable_spatial_guidance && cfg_.w_guide > 1e-6) {
                for (int i = 0; i < num_mid_pts; ++i) {
                    Vector3d err = q.col(i) - P_ref.col(i);
                    J_guide += cfg_.w_guide * err.squaredNorm();
                    grad_q_guide.col(i) += 2.0 * cfg_.w_guide * err;
                }
            }

            // 5. GCOPTER Continuous Quadrature Corridor Penetration Penalty & Kinodynamics
            double J_corr = 0.0;
            double J_dyn = 0.0;
            Eigen::MatrixXd extra_dJ_dC = Eigen::MatrixXd::Zero(6 * M, 3);
            Eigen::VectorXd extra_dJ_dT = Eigen::VectorXd::Zero(M);

            double v_max_sq = cfg_.max_vel * cfg_.max_vel;
            double a_max_sq = cfg_.max_acc * cfg_.max_acc;
            int K = cfg_.quadrature_points;

            for (int j = 0; j < M; ++j) {
                double dt = T(j) / static_cast<double>(K);
                Eigen::Matrix<double, 6, 3> Cj = minco.getCoeffs().block<6, 3>(6 * j, 0);

                for (int k = 0; k <= K; ++k) {
                    double tau = k * dt;
                    double tau2 = tau * tau, tau3 = tau2 * tau, tau4 = tau3 * tau, tau5 = tau4 * tau;
                    double weight = (k == 0 || k == K) ? 0.5 : 1.0;

                    Eigen::RowVectorXd bp(6), bv(6), ba(6);
                    bp << 1.0, tau, tau2, tau3, tau4, tau5;
                    bv << 0.0, 1.0, 2.0*tau, 3.0*tau2, 4.0*tau3, 5.0*tau4;
                    ba << 0.0, 0.0, 2.0, 6.0*tau, 12.0*tau2, 20.0*tau3;

                    Vector3d pos = (bp * Cj).transpose();
                    Vector3d vel = (bv * Cj).transpose();
                    Vector3d acc = (ba * Cj).transpose();

                    // Continuous corridor penetration penalty on interior curve points (GCOPTER)
                    if (cfg_.enable_corridor_penalty && clearance_fn) {
                        double cl = clearance_fn(pos);
                        double viol = cfg_.r_safe - cl;
                        if (viol > 0.0) {
                            double p_viol = viol * viol * viol; // cubic penalty functional
                            J_corr += cfg_.w_corridor * p_viol * weight * dt;

                            Vector3d grad_cl = eval_grad(pos);
                            Vector3d dJ_dpos = -3.0 * cfg_.w_corridor * (viol * viol) * grad_cl * weight * dt;

                            extra_dJ_dC.block<6, 3>(6 * j, 0) += bp.transpose() * dJ_dpos.transpose();
                            extra_dJ_dT(j) += cfg_.w_corridor * p_viol * weight * (1.0 / static_cast<double>(K));
                        }
                    }

                    // Kinodynamic penalties (vel, acc)
                    if (cfg_.w_dyn > 1e-6) {
                        double v_viol = vel.squaredNorm() - v_max_sq;
                        if (v_viol > 0.0) {
                            double pen_v = v_viol * v_viol * v_viol;
                            J_dyn += cfg_.w_dyn * pen_v * weight * dt;
                            Vector3d dJ_dvel = 6.0 * cfg_.w_dyn * (v_viol * v_viol) * vel * weight * dt;
                            extra_dJ_dC.block<6, 3>(6 * j, 0) += bv.transpose() * dJ_dvel.transpose();
                            extra_dJ_dT(j) += cfg_.w_dyn * pen_v * weight * (1.0 / static_cast<double>(K));
                        }

                        double a_viol = acc.squaredNorm() - a_max_sq;
                        if (a_viol > 0.0) {
                            double pen_a = a_viol * a_viol * a_viol;
                            J_dyn += cfg_.w_dyn * pen_a * weight * dt;
                            Vector3d dJ_dacc = 6.0 * cfg_.w_dyn * (a_viol * a_viol) * acc * weight * dt;
                            extra_dJ_dC.block<6, 3>(6 * j, 0) += ba.transpose() * dJ_dacc.transpose();
                            extra_dJ_dT(j) += cfg_.w_dyn * pen_a * weight * (1.0 / static_cast<double>(K));
                        }
                    }
                }
            }

            // 6. MINCO exact adjoint sensitivity solve
            Eigen::MatrixXd grad_q;
            Eigen::VectorXd grad_T;
            Eigen::Matrix3d grad_sf;
            minco.computeGradients(grad_q, grad_T, grad_sf, &extra_dJ_dC, &extra_dJ_dT);

            grad_q += grad_q_guide;
            grad_T.array() += cfg_.rho_time;

            // 7. Chain rule backward to x = [xi; tau]
            for (int i = 0; i < num_mid_pts; ++i) {
                for (int dim = 0; dim < 3; ++dim) {
                    cur_grad(3 * i + dim) = grad_q(dim, i) * dq_dxi(dim, i);
                }
            }
            for (int j = 0; j < M; ++j) {
                cur_grad(3 * num_mid_pts + j) = grad_T(j) * dT_dtau(j);
            }

            return J_jerk + J_time + J_guide + J_corr + J_dyn;
        };

        // Solve unconstrained optimization using L-BFGS
        LBFGSConfig lbfgs_cfg;
        lbfgs_cfg.max_iterations = cfg_.max_iters;
        lbfgs_cfg.grad_tolerance = cfg_.grad_tolerance;

        double final_cost = 0.0;
        res.iterations = LBFGSOptimizer::optimize(cost_function, x, final_cost, lbfgs_cfg);
        res.final_cost = final_cost;

        auto t_end = std::chrono::steady_clock::now();
        res.solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // Sample resulting trajectory
        double total_T = std::clamp(minco.getTotalDuration(), 0.05, 300.0);
        res.total_duration = total_T;
        res.jerk_energy = minco.computeJerkEnergy();
        res.success = (total_T > 0.05 && total_T < 290.0);

        double max_v = 0.0, max_a = 0.0, max_dev = 0.0;
        double dt_sample = 0.05;
        for (double t = 0.0; t <= total_T; t += dt_sample) {
            TrajectoryState state;
            state.t = t;
            state.pos = minco.evaluatePosition(t);
            state.vel = minco.evaluateVelocity(t);
            state.acc = minco.evaluateAcceleration(t);

            max_v = std::max(max_v, state.vel.norm());
            max_a = std::max(max_a, state.acc.norm());
            res.sampled_states.push_back(state);
        }

        // Measure corner-cutting / deviation from reference centers
        for (int i = 0; i < num_mid_pts; ++i) {
            // Find trajectory closest point to waypoint i
            Vector3d q_i = corridor_boxes[i].center();
            double d_min_traj = std::numeric_limits<double>::infinity();
            for (const auto& s : res.sampled_states) {
                d_min_traj = std::min(d_min_traj, (s.pos - q_i).norm());
            }
            max_dev = std::max(max_dev, d_min_traj);
        }

        res.max_vel = max_v;
        res.max_acc = max_a;
        res.max_dist_to_center = max_dev;
        return res;
    }

private:
    SGMincoConfig cfg_;
};

} // namespace hea_planner

#endif // HEA_PLANNER_SG_MINCO_HPP_
