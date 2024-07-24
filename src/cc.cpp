#include "cc.h"

using namespace TOCABI;

CustomController::CustomController(RobotData &rd) : rd_(rd) //, wbc_(dc.wbc_)
{
    ControlVal_.setZero();

    if (is_write_file_)
    {
        if (is_on_robot_)
        {
            writeFile.open("/home/dyros/catkin_ws/src/tocabi_cc/result/data.csv", std::ofstream::out | std::ofstream::app);
        }
        else
        {
            // writeFile.open("/home/yong20/ros_ws/ros/tocabi_ws/src/tocabi_cc/result/data.csv", std::ofstream::out | std::ofstream::app);
            writeFile.open("/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/result/obs.txt", std::ofstream::out | std::ofstream::app);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    initVariable();
    loadNetwork();

    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &CustomController::joyCallback, this);
}

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}

void CustomController::loadNetwork()
{
    state_.setZero();
    rl_action_.setZero();


    string cur_path = "/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
    }
    std::ifstream file[18];

    file[0].open(cur_path+"weight/a2c_network_actor_mlp_0_weight.txt", std::ios::in);
    file[1].open(cur_path+"weight/a2c_network_actor_mlp_0_bias.txt", std::ios::in);
    file[2].open(cur_path+"weight/a2c_network_actor_mlp_2_weight.txt", std::ios::in);
    file[3].open(cur_path+"weight/a2c_network_actor_mlp_2_bias.txt", std::ios::in);
    file[4].open(cur_path+"weight/a2c_network_mu_weight.txt", std::ios::in);
    file[5].open(cur_path+"weight/a2c_network_mu_bias.txt", std::ios::in);
    file[6].open(cur_path+"weight/running_mean_std_running_mean.txt", std::ios::in);
    file[7].open(cur_path+"weight/running_mean_std_running_var.txt", std::ios::in);
    file[8].open(cur_path+"weight/a2c_network_critic_mlp_0_weight.txt", std::ios::in);
    file[9].open(cur_path+"weight/a2c_network_critic_mlp_0_bias.txt", std::ios::in);
    file[10].open(cur_path+"weight/a2c_network_critic_mlp_2_weight.txt", std::ios::in);
    file[11].open(cur_path+"weight/a2c_network_critic_mlp_2_bias.txt", std::ios::in);
    file[12].open(cur_path+"weight/a2c_network_value_weight.txt", std::ios::in);
    file[13].open(cur_path+"weight/a2c_network_value_bias.txt", std::ios::in);


    if(!file[0].is_open())
    {
        std::cout<<"Can not find the weight file"<<std::endl;
    }

    float temp;
    auto loadMatrix = [&](std::ifstream& file, Eigen::MatrixXd& matrix) {
        int row = 0, col = 0;
        while (!file.eof() && row != matrix.rows()) {
            file >> temp;
            if (file.fail()) break; // Ensure we don't read past the end of file
            matrix(row, col) = temp;
            col++;
            if (col == matrix.cols()) {
                col = 0;
                row++;
            }
        }
    };
    loadMatrix(file[0], policy_net_w0_);
    loadMatrix(file[1], policy_net_b0_);
    loadMatrix(file[2], policy_net_w2_);
    loadMatrix(file[3], policy_net_b2_);
    loadMatrix(file[4], action_net_w_);
    loadMatrix(file[5], action_net_b_);
    loadMatrix(file[6], state_mean_);
    loadMatrix(file[7], state_var_);
    loadMatrix(file[8], value_net_w0_);
    loadMatrix(file[9], value_net_b0_);
    loadMatrix(file[10], value_net_w2_);
    loadMatrix(file[11], value_net_b2_);
    loadMatrix(file[12], value_net_w_);
    loadMatrix(file[13], value_net_b_);  
}

void CustomController::initVariable()
{    
    policy_net_w0_.resize(num_hidden1, num_state);
    policy_net_b0_.resize(num_hidden1, 1);
    policy_net_w2_.resize(num_hidden2, num_hidden1);
    policy_net_b2_.resize(num_hidden2, 1);
    action_net_w_.resize(num_action, num_hidden2);
    action_net_b_.resize(num_action, 1);

    hidden_layer1_.resize(num_hidden1, 1);
    hidden_layer2_.resize(num_hidden2, 1);
    rl_action_.resize(num_action, 1);

    value_net_w0_.resize(num_hidden1, num_state);
    value_net_b0_.resize(num_hidden1, 1);
    value_net_w2_.resize(num_hidden2, num_hidden1);
    value_net_b2_.resize(num_hidden2, 1);
    value_net_w_.resize(1, num_hidden2);
    value_net_b_.resize(1, 1);

    value_hidden_layer1_.resize(num_hidden1, 1);
    value_hidden_layer2_.resize(num_hidden2, 1);
    
    state_cur_.resize(num_cur_state, 1);
    state_.resize(num_state, 1);
    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 1);
    state_mean_.resize(num_state, 1);
    state_var_.resize(num_state, 1);

    q_dot_lpf_.setZero();

    torque_bound_ << 333, 232, 263, 289, 222, 166,
                    333, 232, 263, 289, 222, 166,
                    303, 303, 303, 
                    64, 64, 64, 64, 23, 23, 10, 10,
                    10, 10,
                    64, 64, 64, 64, 23, 23, 10, 10;  
                    
    q_init_ << 0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, 0.0,
                0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                0.0, 0.0,
                -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0;

    kp_.setZero();
    kv_.setZero();
    kp_.diagonal() <<   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        6000.0, 10000.0, 10000.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
                        100.0, 100.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0;
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;
}

Eigen::Vector3d CustomController::mat2euler(Eigen::Matrix3d mat)
{
    Eigen::Vector3d euler;

    double cy = std::sqrt(mat(2, 2) * mat(2, 2) + mat(1, 2) * mat(1, 2));
    if (cy > std::numeric_limits<double>::epsilon())
    {
        euler(2) = -atan2(mat(0, 1), mat(0, 0));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = -atan2(mat(1, 2), mat(2, 2));
    }
    else
    {
        euler(2) = -atan2(-mat(1, 0), mat(1, 1));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = 0.0;
    }
    return euler;
}

void CustomController::processNoise()
{
    time_cur_ = rd_cc_.control_time_us_ / 1e6;
    if (is_on_robot_)
    {
        q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
        q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_dot_lpf_ = q_dot_lpf_;
        }
    }
    else
    {
        std::random_device rd;  
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.00001, 0.00001);
        for (int i = 0; i < MODEL_DOF; i++) {
            q_noise_(i) = rd_cc_.q_virtual_(6+i) + dis(gen);
        }
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_vel_noise_ = (q_noise_ - q_noise_pre_) / (time_cur_ - time_pre_);
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_vel_noise_ = q_vel_noise_;
            q_dot_lpf_ = q_dot_lpf_;
        }
        q_noise_pre_ = q_noise_;
    }
    time_pre_ = time_cur_;
}

void CustomController::processObservation()
{
    /*
    obs 
        1) root_h: root height (z)                  (1)     0
        2) target_h: target height (z)              (1)     1
        3) root_rot: root rotation                  (6)     2:8
        4) root_vel: root linear velocity           (3)     8:11
        5) root_ang_vel: root angular velocity      (3)     11:14
        6) commands: x, y, yaw                      (3)     14:17
        7) dof_pos: dof position                    (12)    17:29
        8) dof_vel: dof velocity                    (12)    29:41
        9) key pos: local key position              (6)     41:47
        10) action: action                           (12)    47:59
    */

    int data_idx = 0;
    
    // 1) root_h: root height (z)                  (1)     0 
    state_cur_(data_idx) = rd_cc_.q_virtual_(2);
    data_idx++;
    // 2) target_h: target height (z)              (1)     1
    state_cur_(data_idx) = 0.8282;
    data_idx++;

    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);    

    // 3) root_rot: root rotation                  (3)     2:8
    euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());

    state_cur_(data_idx) = euler_angle_(0);
    data_idx++;

    state_cur_(data_idx) = euler_angle_(1);
    data_idx++;

    state_cur_(data_idx) = euler_angle_(2);
    data_idx++;

    // quatToTanNorm(q, tan_vec, nor_vec);
    // state_cur_(data_idx) = tan_vec(0);
    // data_idx++;
    // state_cur_(data_idx) = tan_vec(1);
    // data_idx++;
    // state_cur_(data_idx) = tan_vec(2);
    // data_idx++;

    // state_cur_(data_idx) = nor_vec(0);
    // data_idx++;
    // state_cur_(data_idx) = nor_vec(1);
    // data_idx++;
    // state_cur_(data_idx) = nor_vec(2);
    // data_idx++;

    // 4) root_vel: root linear velocity           (3)     8:11
    // 5) root_ang_vel: root angular velocity      (3)     11:14    
    Vector3d local_root_vel = quatRotateInverse(q, rd_cc_.q_dot_virtual_.head(3));
    Vector3d local_root_ang_vel = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(3,3));
    for (int i=0; i<6; i++)
    {
        // state_cur_(data_idx) = rd_cc_.q_dot_virtual_(i);
        if (i < 3)
        {
            state_cur_(data_idx) = local_root_vel(i);
        }
        else
        {
            state_cur_(data_idx) = local_root_ang_vel(i-3);
        }
        data_idx++;
    }

    // 6) commands: x, y, yaw                      (3)     14:17
    state_cur_(data_idx) = 0.4;
    data_idx++;
    state_cur_(data_idx) = 0.0;
    data_idx++;
    state_cur_(data_idx) = 0.0;
    data_idx++;

    // 7) dof_pos: dof position                    (12)    17:29
    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_(data_idx) = q_noise_(i);
        data_idx++;
    }

    // 8) dof_vel: dof velocity                    (12)    29:41
    for (int i = 0; i < num_actuator_action; i++)
    {
        if (is_on_robot_)
        {
            state_cur_(data_idx) = q_vel_noise_(i);
        }
        else
        {
            state_cur_(data_idx) = q_vel_noise_(i); //rd_cc_.q_dot_virtual_(i+6); //q_vel_noise_(i);
        }
        data_idx++;
    }

    // Eigen::Isometry3d pelv;
    // pelv.linear() = q.toRotationMatrix();
    // pelv.translation() = rd_cc_.q_virtual_.head(3);
    // Eigen::Vector3d local_left = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(pelv), rd_cc_.link_[Left_Foot].xpos);
    // Eigen::Vector3d local_right = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(pelv), rd_cc_.link_[Right_Foot].xpos);
    // // 9) key pos: local key position              (6)     41:47
    // state_cur_(data_idx) = local_left(0);
    // data_idx++;
    // state_cur_(data_idx) = local_left(1);
    // data_idx++;
    // state_cur_(data_idx) = local_left(2);
    // data_idx++;
    // state_cur_(data_idx) = local_right(0);
    // data_idx++;
    // state_cur_(data_idx) = local_right(1);
    // data_idx++;
    // state_cur_(data_idx) = local_right(2);
    // data_idx++;

    // float squat_duration = 1.7995;
    // phase_ = std::fmod((rd_cc_.control_time_us_-start_time_)/1e6 + action_dt_accumulate_, squat_duration) / squat_duration;

    // state_cur_(data_idx) = sin(2*M_PI*phase_);
    // data_idx++;
    // state_cur_(data_idx) = cos(2*M_PI*phase_);
    // data_idx++;

    // // state_cur_(data_idx) = 0.5;//target_vel_x_;
    // state_cur_(data_idx) = target_vel_x_;
    // data_idx++;

    // state_cur_(data_idx) = 0.0;//target_vel_y_;
    // state_cur_(data_idx) = target_vel_y_;
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.LF_FT(2);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.RF_FT(2);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.LF_FT(3);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.RF_FT(3);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.LF_FT(4);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.RF_FT(4);
    // data_idx++;
    // 10) action: action                           (12)    47:59
    for (int i = 0; i <num_actuator_action; i++) 
    {
        state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
        data_idx++;
    }
    
    // state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(num_actuator_action), 0.0, 1.0);
    // data_idx++;
    
    state_buffer_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist-1),1) = state_buffer_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist-1),1);
    // state_buffer_.block(num_cur_state*(num_state_skip*num_state_hist-1), 0, num_cur_state,1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();

    // Internal State First
    for (int i = 0; i < num_state_hist; i++)
    {
        state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)-1), 0, num_cur_internal_state, 1);
    }
    // Action History Second
    for (int i = 0; i < num_state_hist-1; i++)
    {
        state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)) + num_cur_internal_state, 0, num_action, 1);
    }
    state_ = (state_ - state_mean_).array() / state_var_.cwiseSqrt().array();

}

void CustomController::feedforwardPolicy()
{
    hidden_layer1_ = policy_net_w0_ * state_ + policy_net_b0_;
    for (int i = 0; i < num_hidden1; i++) 
    {
        if (hidden_layer1_(i) < 0)
            hidden_layer1_(i) = 0.0;
    }

    hidden_layer2_ = policy_net_w2_ * hidden_layer1_ + policy_net_b2_;
    for (int i = 0; i < num_hidden2; i++) 
    {
        if (hidden_layer2_(i) < 0)
            hidden_layer2_(i) = 0.0;
    }

    rl_action_ = action_net_w_ * hidden_layer2_ + action_net_b_;
    // cout << rl_action_.transpose() << endl;
    value_hidden_layer1_ = value_net_w0_ * state_ + value_net_b0_;
    for (int i = 0; i < num_hidden1; i++) 
    {
        if (value_hidden_layer1_(i) < 0)
            value_hidden_layer1_(i) = 0.0;
    }

    value_hidden_layer2_ = value_net_w2_ * value_hidden_layer1_ + value_net_b2_;
    for (int i = 0; i < num_hidden2; i++) 
    {
        if (value_hidden_layer2_(i) < 0)
            value_hidden_layer2_(i) = 0.0;
    }

    value_ = (value_net_w_ * value_hidden_layer2_ + value_net_b_)(0);
    
}

void CustomController::computeSlow()
{
    copyRobotData(rd_);
    if (rd_cc_.tc_.mode == 7)
    {
        if (rd_cc_.tc_init)
        {
            //Initialize settings for Task Control! 
            start_time_ = rd_cc_.control_time_us_;
            q_noise_pre_ = q_noise_ = q_init_ = rd_cc_.q_virtual_.segment(6,MODEL_DOF);
            time_cur_ = start_time_ / 1e6;
            time_pre_ = time_cur_ - 0.005;
            time_inference_pre_ = rd_cc_.control_time_us_ - (1/249.9)*1e6;
            // ft_left_init_ = abs(rd_cc_.LF_FT(2));
            // ft_right_init_ = abs(rd_cc_.RF_FT(2));

            rd_.tc_init = false;
            std::cout<<"cc mode 7"<<std::endl;
            torque_init_ = rd_cc_.torque_desired;

            processNoise();
            processObservation();
            for (int i = 0; i < num_state_skip*num_state_hist; i++) 
            {
                // state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();
                state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1).setZero();
            }
        }

        processNoise();

        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/250.0 - 1/10000.0)
        {
            processObservation();
            feedforwardPolicy();
            
            // action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*1/250.0, 0.0, 1/250.0);

            // if (value_ < 50.0)
            // {
            //     if (stop_by_value_thres_ == false)
            //     {
            //         stop_by_value_thres_ = true;
            //         stop_start_time_ = rd_cc_.control_time_us_;
            //         q_stop_ = q_noise_;
            //         std::cout << "Stop by Value Function" << std::endl;
            //     }
            // }

            if (is_write_file_)
            {
                // writeFile << (rd_cc_.control_time_us_ - time_inference_pre_)/1e6 << "\t";
                // writeFile << phase_ << "\t";
                // writeFile << DyrosMath::minmax_cut(rl_action_(num_action-1)*1/250.0, 0.0, 1/250.0) << "\t";

                // writeFile << rd_cc_.LF_FT.transpose() << "\t";
                // writeFile << rd_cc_.RF_FT.transpose() << "\t";
                // writeFile << rd_cc_.LF_CF_FT.transpose() << "\t";
                // writeFile << rd_cc_.RF_CF_FT.transpose() << "\t";

                // writeFile << rd_cc_.torque_desired.transpose()  << "\t";
                // writeFile << q_noise_.transpose() << "\t";
                // writeFile << q_dot_lpf_.transpose() << "\t";
                // writeFile << rd_cc_.q_dot_virtual_.transpose() << "\t";
                // writeFile << rd_cc_.q_virtual_.transpose() << "\t";

                // writeFile << value_ << "\t" << stop_by_value_thres_;
            
                // writeFile << std::endl;

                // time_write_pre_ = rd_cc_.control_time_us_;
            }
            
            time_inference_pre_ = rd_cc_.control_time_us_;
        }
        for (int i = 0; i < num_actuator_action; i++)
        {
            torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i)*torque_bound_(i), -torque_bound_(i), torque_bound_(i));
        }
        for (int i = num_actuator_action; i < MODEL_DOF; i++)
        {
            torque_rl_(i) = kp_(i,i) * (q_init_(i) - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i);
        }
        
        if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
        {
            for (int i = 0; i <MODEL_DOF; i++)
            {
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.1e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);
            }
            rd_.torque_desired = torque_spline_;
        }
        else
        {
            rd_.torque_desired = torque_rl_;
        }

        if (stop_by_value_thres_)
        {
            rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_*q_vel_noise_;
        }


    }
}

void CustomController::computeFast()
{
    // if (tc.mode == 10)
    // {
    // }
    // else if (tc.mode == 11)
    // {
    // }
}

void CustomController::computePlanner()
{
}

void CustomController::copyRobotData(RobotData &rd_l)
{
    std::memcpy(&rd_cc_, &rd_l, sizeof(RobotData));
}

void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[0], -0.5, 1.0);
    target_vel_y_ = DyrosMath::minmax_cut(joy->axes[1], -0.3, 0.3);
}

void CustomController::quatToTanNorm(const Eigen::Quaterniond& quaternion, Eigen::Vector3d& tangent, Eigen::Vector3d& normal) {
    // Reference direction and normal vectors
    Eigen::Vector3d refDirection(1, 0, 0); // Tangent vector reference
    Eigen::Vector3d refNormal(0, 1, 0);    // Normal vector reference

    // Rotate the reference vectors
    tangent = quaternion * refDirection;
    normal = quaternion * refNormal;

    // Normalize the vectors
    tangent.normalize();
    normal.normalize();
}

Eigen::Vector3d CustomController::quatRotateInverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) {

    Eigen::Vector3d q_vec = q.vec();
    double q_w = q.w();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = 2.0 * q_w * q_vec.cross(v);
    Eigen::Vector3d c = 2.0 * q_vec * q_vec.dot(v);

    return a - b + c;
}