#include "preview_controller.h"
using namespace std;

PreviewController::PreviewController(double dt, double time_horizon)
    : dt_(dt), time_horizon_(time_horizon)
{
    NL_ = static_cast<int>(time_horizon_);
    
    // Initialize state and error integral
    state_.setZero();
    error_integral_.setZero();
}

PreviewController::~PreviewController()
{
}

void PreviewController::init(float vrp_height)
{
    // System matrices A, B, C
    A_ <<   1, dt_, pow(dt_, 2) / 2,
            0, 1, dt_,
            0, 0, 1;

    B_ <<   pow(dt_, 3) / 6,
            pow(dt_, 2) / 2,
            dt_;

    C_ << 1, 0, -vrp_height / 9.81;

    // Augmented system matrices (MATLAB: A_bar = [I_bar, F_bar])
    // A_bar (4x4): col 0 = [1;0;0;0], cols 1-3 = [C*A; A]
    Eigen::Vector4d I_bar;
    I_bar << 1, 0, 0, 0;

    Eigen::Matrix4d A_bar;
    A_bar.col(0) = I_bar;
    A_bar.block<1, 3>(0, 1) = C_ * A_;
    A_bar.block<3, 3>(1, 1) = A_;

    // B_bar (4x1): [C*B; B]
    Eigen::Vector4d B_bar;
    B_bar(0) = (C_ * B_)(0);
    B_bar.tail<3>() = B_;

    // Cost matrices: Q_bar has Q_e=1 at (0,0), R=1e-6
    Eigen::Matrix4d Q_bar = Eigen::Matrix4d::Zero();
    Q_bar(0, 0) = 1.0;
    const double R = 1e-6;

    // Solve DARE iteratively: K = A'KA - A'KB(R+B'KB)^{-1}B'KA + Q
    Eigen::Matrix4d K = Q_bar;
    for (int iter = 0; iter < 200000; ++iter)
    {
        double S_iter = R + (B_bar.transpose() * K * B_bar)(0, 0);
        Eigen::Matrix4d K_new = A_bar.transpose() * K * A_bar
                              - (A_bar.transpose() * K * B_bar) * (1.0 / S_iter) * (B_bar.transpose() * K * A_bar)
                              + Q_bar;
        double err = (K_new - K).norm();
        K = K_new;
        if (err < 1e-10)
            break;
    }
    K_dare_ = K;

    // Optimal gain: G = (R + B'KB)^{-1} * B' * K * A_bar  (1x4)
    double S = R + (B_bar.transpose() * K * B_bar)(0, 0);
    Eigen::RowVector4d G = (1.0 / S) * (B_bar.transpose() * K * A_bar);

    G_i_dare_ = G(0);
    G_x_dare_ = G.segment(1, 3);  // 1x3

    // Ac_bar_T = (A_bar - B_bar * G)'
    Eigen::Matrix4d Ac_bar_T = (A_bar - B_bar * G).transpose();

    // RBT = (R + B'KB)^{-1} * B'  (1x4)
    Eigen::RowVector4d RBT = (1.0 / S) * B_bar.transpose();

    // X_dare_[0] = -Ac_bar_T * K * I_bar
    X_dare_.setZero(NL_, 4);
    X_dare_.row(0) = (-Ac_bar_T * K * I_bar).transpose();

    G_d_dare_.setZero(NL_);
    G_d_dare_(0) = -G_i_dare_;

    for (int l = 1; l < NL_; ++l)
    {
        X_dare_.row(l) = (Ac_bar_T * X_dare_.row(l - 1).transpose()).transpose();
        G_d_dare_(l) = RBT * X_dare_.row(l - 1).transpose();
    }
}

Eigen::MatrixXd PreviewController::compute_target_state(const Eigen::MatrixXd &vrp_ref)
{
    Eigen::RowVector3d current_output = C_ * state_; 
    error_integral_ += (current_output - vrp_ref.row(0)).transpose();

    Eigen::RowVector3d term1 = -G_i_dare_ * error_integral_.transpose();
    Eigen::RowVector3d term2 = -G_x_dare_ * state_;
    Eigen::RowVector3d term3 = Eigen::RowVector3d::Zero();
    for (int i = 1; i < NL_; ++i)
        term3 -= G_d_dare_(i) * vrp_ref.row(i);

    Eigen::RowVector3d input = term1 + term2 + term3;
    Eigen::Matrix3d next_state = A_ * state_ + B_ * input;
    state_ = next_state;
    return state_;
}

void PreviewController::update_state(const Eigen::MatrixXd &next_state)
{
    state_ = next_state;
}
