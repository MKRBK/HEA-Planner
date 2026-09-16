#ifndef HEA_PLANNER_MINCO_S3NU_HPP_
#define HEA_PLANNER_MINCO_S3NU_HPP_

#include "common.hpp"
#include <Eigen/Dense>
#include <Eigen/LU>

namespace hea_planner {

// MINCO S3NU Solver for 3D piecewise 5th-order minimum-jerk polynomials
class MincoS3NU {
public:
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    using Vec6 = Eigen::Matrix<double, 6, 1>;

    MincoS3NU() = default;

    /**
     * @brief Reset problem dimensions.
     * @param num_segments Number of trajectory segments M.
     */
    void reset(int num_segments) {
        M_ = num_segments;
        N_ = 6 * M_;
        A_.resize(N_, N_);
        B_.resize(N_, 3);
        C_.resize(N_, 3);
        adj_lambda_.resize(N_, 3);
    }

    // Compute 6x6 jerk cost Hessian Q(T) for a single piece
    static Mat6 computeQ(double T) {
        Mat6 Q = Mat6::Zero();
        double T2 = T * T;
        double T3 = T2 * T;
        double T4 = T3 * T;
        double T5 = T4 * T;

        Q(3, 3) = 36.0 * T;
        Q(3, 4) = 72.0 * T2;   Q(4, 3) = Q(3, 4);
        Q(3, 5) = 120.0 * T3;  Q(5, 3) = Q(3, 5);
        Q(4, 4) = 192.0 * T3;
        Q(4, 5) = 360.0 * T4;  Q(5, 4) = Q(4, 5);
        Q(5, 5) = 720.0 * T5;
        return Q;
    }

    // Compute derivative dQ/dT
    static Mat6 computeQDerivative(double T) {
        Mat6 Qd = Mat6::Zero();
        double T2 = T * T;
        double T3 = T2 * T;
        double T4 = T3 * T;

        Qd(3, 3) = 36.0;
        Qd(3, 4) = 144.0 * T;   Qd(4, 3) = Qd(3, 4);
        Qd(3, 5) = 360.0 * T2;  Qd(5, 3) = Qd(3, 5);
        Qd(4, 4) = 576.0 * T2;
        Qd(4, 5) = 1440.0 * T3; Qd(5, 4) = Qd(4, 5);
        Qd(5, 5) = 3600.0 * T4;
        return Qd;
    }

    // Solve linear system A*C = B for polynomial coefficients C
    bool generate(const Eigen::Matrix3d& s0,
                  const Eigen::Matrix3d& sf,
                  const Eigen::MatrixXd& waypoints,
                  const Eigen::VectorXd& durations) {
        M_ = durations.size();
        N_ = 6 * M_;
        if (A_.rows() != N_) reset(M_);

        durations_ = durations;
        waypoints_ = waypoints;
        s0_ = s0;
        sf_ = sf;

        A_.setZero();
        B_.setZero();

        // 1. Initial conditions at t=0 of segment 1 (p, v, a)
        A_(0, 0) = 1.0;
        A_(1, 1) = 1.0;
        A_(2, 2) = 2.0;
        B_.row(0) = s0.row(0);
        B_.row(1) = s0.row(1);
        B_.row(2) = s0.row(2);

        // 2. Intermediate conditions (j = 1 to M-1)
        for (int j = 1; j < M_; ++j) {
            double T = durations(j - 1);
            double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
            int row = 3 + 6 * (j - 1);
            int col_prev = 6 * (j - 1);
            int col_curr = 6 * j;

            // Intermediate position waypoint condition: p_j(T_j) = q_j
            A_(row, col_prev + 0) = 1.0;
            A_(row, col_prev + 1) = T;
            A_(row, col_prev + 2) = T2;
            A_(row, col_prev + 3) = T3;
            A_(row, col_prev + 4) = T4;
            A_(row, col_prev + 5) = T5;
            B_.row(row) = waypoints.col(j - 1).transpose();

            // Position continuity: p_{j+1}(0) - p_j(T_j) = 0
            A_(row + 1, col_curr + 0) = 1.0;
            A_(row + 1, col_prev + 0) = -1.0;
            A_(row + 1, col_prev + 1) = -T;
            A_(row + 1, col_prev + 2) = -T2;
            A_(row + 1, col_prev + 3) = -T3;
            A_(row + 1, col_prev + 4) = -T4;
            A_(row + 1, col_prev + 5) = -T5;

            // Velocity continuity: v_{j+1}(0) - v_j(T_j) = 0
            A_(row + 2, col_curr + 1) = 1.0;
            A_(row + 2, col_prev + 1) = -1.0;
            A_(row + 2, col_prev + 2) = -2.0 * T;
            A_(row + 2, col_prev + 3) = -3.0 * T2;
            A_(row + 2, col_prev + 4) = -4.0 * T3;
            A_(row + 2, col_prev + 5) = -5.0 * T4;

            // Acceleration continuity: a_{j+1}(0) - a_j(T_j) = 0
            A_(row + 3, col_curr + 2) = 2.0;
            A_(row + 3, col_prev + 2) = -2.0;
            A_(row + 3, col_prev + 3) = -6.0 * T;
            A_(row + 3, col_prev + 4) = -12.0 * T2;
            A_(row + 3, col_prev + 5) = -20.0 * T3;

            // Jerk continuity: j_{j+1}(0) - j_j(T_j) = 0
            A_(row + 4, col_curr + 3) = 6.0;
            A_(row + 4, col_prev + 3) = -6.0;
            A_(row + 4, col_prev + 4) = -24.0 * T;
            A_(row + 4, col_prev + 5) = -60.0 * T2;

            // Snap continuity: s_{j+1}(0) - s_j(T_j) = 0
            A_(row + 5, col_curr + 4) = 24.0;
            A_(row + 5, col_prev + 4) = -24.0;
            A_(row + 5, col_prev + 5) = -120.0 * T;
        }

        // 3. Terminal conditions at t=T_M of segment M (p, v, a)
        {
            double T = durations(M_ - 1);
            double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
            int row = N_ - 3;
            int col = 6 * (M_ - 1);

            A_(row + 0, col + 0) = 1.0;
            A_(row + 0, col + 1) = T;
            A_(row + 0, col + 2) = T2;
            A_(row + 0, col + 3) = T3;
            A_(row + 0, col + 4) = T4;
            A_(row + 0, col + 5) = T5;
            B_.row(row + 0) = sf.row(0);

            A_(row + 1, col + 1) = 1.0;
            A_(row + 1, col + 2) = 2.0 * T;
            A_(row + 1, col + 3) = 3.0 * T2;
            A_(row + 1, col + 4) = 4.0 * T3;
            A_(row + 1, col + 5) = 5.0 * T4;
            B_.row(row + 1) = sf.row(1);

            A_(row + 2, col + 2) = 2.0;
            A_(row + 2, col + 3) = 6.0 * T;
            A_(row + 2, col + 4) = 12.0 * T2;
            A_(row + 2, col + 5) = 20.0 * T3;
            B_.row(row + 2) = sf.row(2);
        }

        // Solve A * C = B
        lu_solver_.compute(A_);
        C_ = lu_solver_.solve(B_);
        return true;
    }

    // Compute total jerk energy
    double computeJerkEnergy() const {
        double energy = 0.0;
        for (int j = 0; j < M_; ++j) {
            Mat6 Q = computeQ(durations_(j));
            Eigen::Matrix<double, 6, 3> Cj = C_.block<6, 3>(6 * j, 0);
            for (int dim = 0; dim < 3; ++dim) {
                energy += Cj.col(dim).dot(Q * Cj.col(dim));
            }
        }
        return energy;
    }

    // Compute adjoint sensitivities w.r.t waypoints q, durations T, and sf
    void computeGradients(Eigen::MatrixXd& grad_q,
                          Eigen::VectorXd& grad_T,
                          Eigen::Matrix3d& grad_sf,
                          const Eigen::MatrixXd* extra_dJ_dC = nullptr,
                          const Eigen::VectorXd* extra_dJ_dT = nullptr) {
        grad_q.resize(3, M_ - 1);
        grad_q.setZero();
        grad_T.resize(M_);
        grad_T.setZero();
        grad_sf.setZero();

        // 1. Partial derivative dJ/dC
        Eigen::MatrixXd dJ_dC = Eigen::MatrixXd::Zero(N_, 3);
        for (int j = 0; j < M_; ++j) {
            Mat6 Q = computeQ(durations_(j));
            Eigen::Matrix<double, 6, 3> Cj = C_.block<6, 3>(6 * j, 0);
            dJ_dC.block<6, 3>(6 * j, 0) = 2.0 * Q * Cj;
        }
        if (extra_dJ_dC != nullptr && extra_dJ_dC->rows() == N_ && extra_dJ_dC->cols() == 3) {
            dJ_dC += *extra_dJ_dC;
        }

        // 2. Solve adjoint system: A^T * lambda = dJ/dC
        adj_lambda_ = lu_solver_.transpose().solve(dJ_dC);

        // 3. Extract gradient w.r.t waypoints q_j (from row 3 + 6*(j-1) of B)
        for (int j = 1; j < M_; ++j) {
            int row = 3 + 6 * (j - 1);
            grad_q.col(j - 1) = adj_lambda_.row(row).transpose();
        }

        // 4. Extract adjoint gradient w.r.t terminal state sf
        int row_sf = N_ - 3;
        grad_sf.row(0) = adj_lambda_.row(row_sf + 0);
        grad_sf.row(1) = adj_lambda_.row(row_sf + 1);
        grad_sf.row(2) = adj_lambda_.row(row_sf + 2);

        // 5. Compute gradient w.r.t durations T_j:
        // dJ/dT_j = c_j^T (dQ_j/dT_j) c_j - lambda^T (dA/dT_j) C
        for (int j = 0; j < M_; ++j) {
            double T = durations_(j);
            Mat6 Qd = computeQDerivative(T);
            Eigen::Matrix<double, 6, 3> Cj = C_.block<6, 3>(6 * j, 0);

            double dJ_direct = 0.0;
            for (int dim = 0; dim < 3; ++dim) {
                dJ_direct += Cj.col(dim).dot(Qd * Cj.col(dim));
            }

            // Adjoint sensitivity term: - lambda^T * (dA / dT_j) * C
            double adj_term = 0.0;
            if (j < M_ - 1) {
                int row = 3 + 6 * j;
                int col = 6 * j;
                double T2 = T * T, T3 = T2 * T, T4 = T3 * T;
                Eigen::RowVectorXd dA_row(6);

                dA_row << 0.0, 1.0, 2.0*T, 3.0*T2, 4.0*T3, 5.0*T4;
                for (int dim = 0; dim < 3; ++dim) {
                    double c_val = dA_row.dot(C_.block<6, 1>(col, dim));
                    adj_term -= adj_lambda_(row, dim) * c_val;
                    adj_term += adj_lambda_(row + 1, dim) * c_val;
                }

                dA_row << 0.0, 0.0, 2.0, 6.0*T, 12.0*T2, 20.0*T3;
                for (int dim = 0; dim < 3; ++dim) {
                    double c_val = dA_row.dot(C_.block<6, 1>(col, dim));
                    adj_term += adj_lambda_(row + 2, dim) * c_val;
                }

                dA_row << 0.0, 0.0, 0.0, 6.0, 24.0*T, 60.0*T2;
                for (int dim = 0; dim < 3; ++dim) {
                    double c_val = dA_row.dot(C_.block<6, 1>(col, dim));
                    adj_term += adj_lambda_(row + 3, dim) * c_val;
                }

                dA_row << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0*T;
                for (int dim = 0; dim < 3; ++dim) {
                    double c_val = dA_row.dot(C_.block<6, 1>(col, dim));
                    adj_term += adj_lambda_(row + 4, dim) * c_val;
                }

                dA_row << 0.0, 0.0, 0.0, 0.0, 0.0, 120.0;
                for (int dim = 0; dim < 3; ++dim) {
                    double c_val = dA_row.dot(C_.block<6, 1>(col, dim));
                    adj_term += adj_lambda_(row + 5, dim) * c_val;
                }
            } else {
                // Segment M terminal conditions
                int row = N_ - 3;
                int col = 6 * (M_ - 1);
                double T2 = T * T, T3 = T2 * T, T4 = T3 * T;
                Eigen::RowVectorXd dA_row(6);

                dA_row << 0.0, 1.0, 2.0*T, 3.0*T2, 4.0*T3, 5.0*T4;
                for (int dim = 0; dim < 3; ++dim) {
                    double c_val = dA_row.dot(C_.block<6, 1>(col, dim));
                    adj_term -= adj_lambda_(row + 0, dim) * c_val;
                }

                dA_row << 0.0, 0.0, 2.0, 6.0*T, 12.0*T2, 20.0*T3;
                for (int dim = 0; dim < 3; ++dim) {
                    double c_val = dA_row.dot(C_.block<6, 1>(col, dim));
                    adj_term -= adj_lambda_(row + 1, dim) * c_val;
                }

                dA_row << 0.0, 0.0, 0.0, 6.0, 24.0*T, 60.0*T2;
                for (int dim = 0; dim < 3; ++dim) {
                    double c_val = dA_row.dot(C_.block<6, 1>(col, dim));
                    adj_term -= adj_lambda_(row + 2, dim) * c_val;
                }
            }

            grad_T(j) = dJ_direct + adj_term;
            if (extra_dJ_dT != nullptr && extra_dJ_dT->size() == M_) {
                grad_T(j) += (*extra_dJ_dT)(j);
            }
        }
    }

    /**
     * @brief Compute standard minimum-jerk gradients (convenience wrapper).
     */
    void computeJerkGradients(Eigen::MatrixXd& grad_q,
                              Eigen::VectorXd& grad_T,
                              Eigen::Matrix3d& grad_sf) {
        computeGradients(grad_q, grad_T, grad_sf, nullptr, nullptr);
    }

    /**
     * @brief Sample trajectory position at time t.
     */
    Vector3d evaluatePosition(double t) const {
        if (t <= 0.0) return C_.block<1, 3>(0, 0).transpose();

        double t_accum = 0.0;
        for (int j = 0; j < M_; ++j) {
            double T = durations_(j);
            if (t <= t_accum + T || j == M_ - 1) {
                double tau = std::clamp(t - t_accum, 0.0, T);
                double tau2 = tau * tau, tau3 = tau2 * tau, tau4 = tau3 * tau, tau5 = tau4 * tau;
                Eigen::RowVectorXd basis(6);
                basis << 1.0, tau, tau2, tau3, tau4, tau5;
                return (basis * C_.block<6, 3>(6 * j, 0)).transpose();
            }
            t_accum += T;
        }
        return C_.block<1, 3>(N_ - 6, 0).transpose();
    }

    /**
     * @brief Sample trajectory velocity at time t.
     */
    Vector3d evaluateVelocity(double t) const {
        double t_accum = 0.0;
        for (int j = 0; j < M_; ++j) {
            double T = durations_(j);
            if (t <= t_accum + T || j == M_ - 1) {
                double tau = std::clamp(t - t_accum, 0.0, T);
                double tau2 = tau * tau, tau3 = tau2 * tau, tau4 = tau3 * tau;
                Eigen::RowVectorXd basis(6);
                basis << 0.0, 1.0, 2.0*tau, 3.0*tau2, 4.0*tau3, 5.0*tau4;
                return (basis * C_.block<6, 3>(6 * j, 0)).transpose();
            }
            t_accum += T;
        }
        return Vector3d::Zero();
    }

    /**
     * @brief Sample trajectory acceleration at time t.
     */
    Vector3d evaluateAcceleration(double t) const {
        double t_accum = 0.0;
        for (int j = 0; j < M_; ++j) {
            double T = durations_(j);
            if (t <= t_accum + T || j == M_ - 1) {
                double tau = std::clamp(t - t_accum, 0.0, T);
                double tau2 = tau * tau, tau3 = tau2 * tau;
                Eigen::RowVectorXd basis(6);
                basis << 0.0, 0.0, 2.0, 6.0*tau, 12.0*tau2, 20.0*tau3;
                return (basis * C_.block<6, 3>(6 * j, 0)).transpose();
            }
            t_accum += T;
        }
        return Vector3d::Zero();
    }

    double getTotalDuration() const {
        return durations_.sum();
    }

    const Eigen::MatrixXd& getCoeffs() const {
        return C_;
    }

    const Eigen::VectorXd& getDurations() const {
        return durations_;
    }

private:
    int M_ = 0;
    int N_ = 0;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd B_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd adj_lambda_;
    Eigen::PartialPivLU<Eigen::MatrixXd> lu_solver_;

    Eigen::VectorXd durations_;
    Eigen::MatrixXd waypoints_;
    Eigen::Matrix3d s0_;
    Eigen::Matrix3d sf_;
};

} // namespace hea_planner

#endif // HEA_PLANNER_MINCO_S3NU_HPP_
