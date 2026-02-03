#ifndef PREVIEW_CONTROLLER_H
#define PREVIEW_CONTROLLER_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

class PreviewController
{
public:
    PreviewController(double dt, double time_horizon);
    ~PreviewController();

    void init();
    Eigen::MatrixXd compute_target_state(const Eigen::MatrixXd &vrp_ref);
    void update_state(const Eigen::MatrixXd &next_state);

    double time_horizon_;
    double dt_;
    int NL_;

    Eigen::Vector3d error_integral_;
    Eigen::Matrix3d state_;

private:
    Eigen::MatrixXd X_dare_;   // NL x 4
    Eigen::VectorXd G_d_dare_; // NL

    Eigen::MatrixXd K_dare_; // 4x4
    double G_i_dare_;
    Eigen::MatrixXd G_x_dare_; // 1x3

    Eigen::Matrix3d A_;
    Eigen::Vector3d B_;
    Eigen::RowVector3d C_;
};

#endif
