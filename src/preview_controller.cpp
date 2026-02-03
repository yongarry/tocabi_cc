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
    // Initialize matrices based on Python code
    X_dare_.setZero(NL_, 4);
    
    // Initial value for X_dare[0]
    Eigen::Vector4d x_dare_0;
    x_dare_0 << -69.0813009391229,
                -2420.65372019103,
                -669.387885399942,
                -2.72127899669319;
    X_dare_.row(0) = x_dare_0;

    // G_d_dare initialization
    G_d_dare_.setZero(NL_);

    // K_dare initialization
    K_dare_.resize(4, 4);
    K_dare_ << 70.0813009391222, 2420.65372019101, 669.387885399937, 2.72127899669319,
               2420.65372019101, 85969.0761591658, 23782.1340369413, 99.0855302680049,
               669.387885399937, 23782.1340369413, 6579.12894080612, 27.4486400273521,
               2.72127899669319, 99.0855302680049, 27.4486400273521, 0.125016968799296;

    G_i_dare_ = 556.382091536873;
    
    G_x_dare_.resize(1, 3);
    G_x_dare_ << 38991.9807941560, 11086.4028841705, 130.234568101964;

    G_d_dare_(0) = -G_i_dare_;

    // Ac_bar_T and RBT for calculating G_d_dare sequence
    Eigen::Matrix4d Ac_bar_T;
    Ac_bar_T << 1.33026539679257, -4.74779384778249e-05, -0.0178042269291843, -4.45105673229607,
                24.1454286624088, 0.996672684305565, -1.24774338541315, -311.935846353287,
                6.58882872047239, 0.00705396028721734, 0.645235107706503, -88.6912230733744,
                0.00312854461408964, 2.08866501886319e-05, 0.00383249382073698, -0.0418765448157545;

    Eigen::RowVector4d RBT;
    RBT << -183.753752229875, 0.0264158747121758, 9.90595301706591, 2476.48825426648;

    for (int l = 1; l < NL_; ++l)
    {
        // X_dare[l] = Ac_bar_T * X_dare[l-1]
        X_dare_.row(l) = (Ac_bar_T * X_dare_.row(l - 1).transpose()).transpose();
        // G_d_dare[l] = RBT * X_dare[l-1]
        G_d_dare_(l) = RBT * X_dare_.row(l - 1).transpose();
    }

    // System matrices A, B, C
    A_ << 1, dt_, pow(dt_, 2) / 2,
          0, 1, dt_,
          0, 0, 1;

    B_ << pow(dt_, 3) / 6,
          pow(dt_, 2) / 2,
          dt_;

    // Note: com_height is hardcoded as 0.728 in Python code (0.728/9.81)
    C_ << 1, 0, -0.728 / 9.81;
}

Eigen::MatrixXd PreviewController::compute_target_state(const Eigen::MatrixXd &vrp_ref)
{
    // vrp_ref shape: (NL, 3) -> 3 columns for x, y, z (or yaw)
    // In python code: vrp_ref_permute is (NL, num_envs, 3). For single env: (NL, 1, 3) or effectively (NL, 3)
    // Input vrp_ref is expected to be (NL, 3)
    
    // Ensure vrp_ref has correct size
    // if (vrp_ref.rows() != NL_) { ... handle error ... }

    // Python: self.error_integral += torch.bmm(self.C.repeat(...), self.state).squeeze() - vrp_ref_permute[0, :]
    // C * state_ (1x3 * 3x3 = 1x3 row vector)
    Eigen::RowVector3d current_output = C_ * state_; 
    
    // vrp_ref.row(0) is 1x3
    error_integral_ += (current_output - vrp_ref.row(0)).transpose();

    // Python: input = -self.G_i_dare * self.error_integral - G_x_dare * state - Sum(G_d_dare * vrp_ref)
    // G_i_dare * error_integral -> 1x3 row vector (if we transpose error_integral)
    Eigen::RowVector3d term1 = -G_i_dare_ * error_integral_.transpose();
    
    // G_x_dare * state -> 1x3 * 3x3 = 1x3
    Eigen::RowVector3d term2 = -G_x_dare_ * state_;

    // Sum calculation
    Eigen::RowVector3d term3 = Eigen::RowVector3d::Zero();
    
    // Vectorized operation for term3 might be faster, but loop is clearer
    // G_d_dare[1:NL] * vrp_ref[1:NL]
    // Python code: (self.G_d_dare[1:self.NL].reshape(-1, 1, 1).repeat(1, self.num_envs, 3) * vrp_ref_permute[1:self.NL]).sum(dim=0)
    // This is effectively dot product of G_d_dare vector and vrp_ref columns (for indices 1 to NL-1)
    
    // Using Eigen map or block operations
    // term3 = (vrp_ref.bottomRows(NL_ - 1).array().colwise() * G_d_dare_.tail(NL_ - 1).array()).colwise().sum();
    // But dimensions must match. G_d_dare_.tail is vector.
    
    for (int i = 1; i < NL_; ++i)
    {
        term3 -= G_d_dare_(i) * vrp_ref.row(i); // term3 is negative sum in python formula, so we subtract
    }

    // Input u (1x3)
    Eigen::RowVector3d input = term1 + term2 + term3;

    // Next state: A * state + B * input
    // A: 3x3, state: 3x3
    // B: 3x1, input: 1x3 -> B * input = 3x3 (outer product-like, but input is row vector so it broadcasts)
    // In Python: torch.bmm(B, input.unsqueeze(dim=1)) -> (3,1) x (1,3) = (3,3)
    
    Eigen::Matrix3d next_state = A_ * state_ + B_ * input;
    
    state_ = next_state;
    
    return state_;
}

void PreviewController::update_state(const Eigen::MatrixXd &next_state)
{
    state_ = next_state;
}
