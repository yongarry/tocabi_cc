#include "preview_controller.h"

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

void PreviewController::init()
{
    // System matrices A, B, C
    A_ <<   1, dt_, pow(dt_, 2) / 2,
            0, 1, dt_,
            0, 0, 1;

    B_ <<   pow(dt_, 3) / 6,
            pow(dt_, 2) / 2,
            dt_;

    C_ << 1, 0, -0.728 / 9.81;

    X_dare_.setZero(NL_, 4);
    Eigen::Vector4d x_dare_0;
    x_dare_0 << -55.285035334563560,
                -1.555860083640699e+03,
                -4.318355308321815e+02,
                -2.184808186673521;
    X_dare_.row(0) = x_dare_0;

    G_d_dare_.setZero(NL_);

    K_dare_.resize(4, 4);
    K_dare_ <<  56.285035334559723, 1.555860083640587e+03, 4.318355308321502e+02, 2.184808186673372,
                1.555860083640587e+03, 4.438808670021533e+04, 1.232544343284430e+04, 63.789558477062045,
                4.318355308321502e+02, 1.232544343284430e+04, 3.422554848344056e+03, 17.738596936652250,
                2.184808186673372, 63.789558477062045, 17.738596936652250, 0.099079098311263;

    G_i_dare_ = 5.156153390660927e+02;
    G_x_dare_.resize(1, 3);
    G_x_dare_ << 2.902142757843520e+04, 8.312467521458724e+03, 1.144927293595159e+02;

    G_d_dare_(0) = -G_i_dare_;

    Eigen::Matrix4d Ac_bar_T;
    Ac_bar_T << 1.382552154670966,	-8.593588984435812e-05,	-0.025780766953308,	-5.156153390661508,
                22.531961543010048,	0.995163095403594,	-1.451071378921841,	-2.902142757843676e+02,
                6.177295889074953,	0.008614588746423,	0.584376623927040,	-83.124675214591917,    
                0.010785970858796,	3.091787844008025e-05,	0.004275363532024,	-0.144927293595184;

    Eigen::RowVector4d RBT;
    RBT << -1.972497589419503e+02,	0.044309862980228,	13.292958894068470,	2.658591778813694e+03;

    for (int l = 1; l < NL_; ++l)
    {
        // X_dare[l] = Ac_bar_T * X_dare[l-1]
        X_dare_.row(l) = (Ac_bar_T * X_dare_.row(l - 1).transpose()).transpose();
        // G_d_dare[l] = RBT * X_dare[l-1]
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
