#ifndef HEA_PLANNER_LBFGS_OPTIMIZER_HPP_
#define HEA_PLANNER_LBFGS_OPTIMIZER_HPP_

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <cmath>
#include <iostream>

namespace hea_planner {

struct LBFGSConfig {
    int max_iterations = 100;
    int history_size = 8;
    double grad_tolerance = 1e-4;
    double step_tolerance = 1e-6;
    double armijo_c1 = 1e-4;
    double backtrack_factor = 0.5;
    int max_linesearch_iters = 20;
    bool verbose = false;
};

class LBFGSOptimizer {
public:
    using CostFunction = std::function<double(const Eigen::VectorXd& x, Eigen::VectorXd& grad)>;

    static int optimize(const CostFunction& cost_func,
                        Eigen::VectorXd& x,
                        double& final_cost,
                        const LBFGSConfig& cfg = LBFGSConfig()) {
        int n = x.size();
        Eigen::VectorXd grad(n);
        double f = cost_func(x, grad);

        if (grad.norm() <= cfg.grad_tolerance) {
            final_cost = f;
            return 0;
        }

        std::vector<Eigen::VectorXd> s_history;
        std::vector<Eigen::VectorXd> y_history;
        std::vector<double> rho_history;

        int iter = 0;
        for (; iter < cfg.max_iterations; ++iter) {
            if (grad.norm() <= cfg.grad_tolerance) {
                break;
            }

            // Two-loop recursion to compute search direction d = - H_k * grad
            Eigen::VectorXd q = grad;
            int k = s_history.size();
            std::vector<double> alpha(k);

            for (int i = k - 1; i >= 0; --i) {
                alpha[i] = rho_history[i] * s_history[i].dot(q);
                q -= alpha[i] * y_history[i];
            }

            // Initial Hessian scaling H_0 = gamma * I
            double gamma = 1.0;
            if (k > 0) {
                double y_dot_y = y_history.back().dot(y_history.back());
                double s_dot_y = s_history.back().dot(y_history.back());
                if (std::abs(y_dot_y) > 1e-12) {
                    gamma = s_dot_y / y_dot_y;
                }
            } else {
                double g_inf = grad.lpNorm<Eigen::Infinity>();
                if (g_inf > 1.0) {
                    gamma = 0.5 / g_inf;
                }
            }
            Eigen::VectorXd r = gamma * q;

            for (int i = 0; i < k; ++i) {
                double beta = rho_history[i] * y_history[i].dot(r);
                r += s_history[i] * (alpha[i] - beta);
            }

            Eigen::VectorXd dir = -r;

            // Ensure descent direction
            double dir_dot_grad = dir.dot(grad);
            if (dir_dot_grad >= 0.0) {
                double g_inf = grad.lpNorm<Eigen::Infinity>();
                double scale = (g_inf > 1.0) ? (0.5 / g_inf) : 1.0;
                dir = -grad * scale;
                dir_dot_grad = dir.dot(grad);
                s_history.clear();
                y_history.clear();
                rho_history.clear();
            }

            // Armijo backtracking line search
            double step = 1.0;
            double f_new = f;
            Eigen::VectorXd x_new = x;
            Eigen::VectorXd grad_new = grad;

            bool line_search_ok = false;
            for (int ls = 0; ls < cfg.max_linesearch_iters; ++ls) {
                x_new = x + step * dir;
                f_new = cost_func(x_new, grad_new);

                if (f_new <= f + cfg.armijo_c1 * step * dir_dot_grad) {
                    line_search_ok = true;
                    break;
                }
                step *= cfg.backtrack_factor;
            }

            if (!line_search_ok) {
                // Line search failed: do not accept uphill point!
                break;
            }

            // Update L-BFGS history
            Eigen::VectorXd s = x_new - x;
            Eigen::VectorXd y = grad_new - grad;
            double s_dot_y = s.dot(y);

            if (s_dot_y > 1e-10) {
                if (static_cast<int>(s_history.size()) >= cfg.history_size) {
                    s_history.erase(s_history.begin());
                    y_history.erase(y_history.begin());
                    rho_history.erase(rho_history.begin());
                }
                s_history.push_back(s);
                y_history.push_back(y);
                rho_history.push_back(1.0 / s_dot_y);
            }

            x = x_new;
            f = f_new;
            grad = grad_new;

            if (s.norm() < cfg.step_tolerance) break;
        }

        final_cost = f;
        return iter;
    }
};

} // namespace hea_planner

#endif // HEA_PLANNER_LBFGS_OPTIMIZER_HPP_
