#include "cc.h"

using namespace TOCABI;

CustomController::CustomController(RobotData &rd) : rd_(rd) //, wbc_(dc.wbc_)
{
    ControlVal_.setZero();

    nh_.getParam("/tocabi_cc/weight_dir", weight_dir_);

    if (is_write_file_)
    {
        if (is_on_robot_)
        {
            writeFile.open("/home/dyros/catkin_ws/src/tocabi_cc/result/robot_data.txt", std::ofstream::out | std::ofstream::app);
        }
        else
        {
            writeFile.open("/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/result/data.txt", std::ofstream::out | std::ofstream::trunc);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    initVariable();
    loadNetwork();

    // joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy_gui", 10, &CustomController::joyCallback, this);
    xbox_joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy", 10, &CustomController::xBoxJoyCallback, this);
}

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}

void CustomController::loadNetwork()
{
    state_.setZero();
    rl_action_.setZero();


    string cur_path = "/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/" + weight_dir_;

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/weight/";
    }
    std::ifstream file[22];

    file[0].open(cur_path+"a2c_network_actor_mlp_0_weight.txt", std::ios::in);
    file[1].open(cur_path+"a2c_network_actor_mlp_0_bias.txt", std::ios::in);
    file[2].open(cur_path+"a2c_network_actor_mlp_2_weight.txt", std::ios::in);
    file[3].open(cur_path+"a2c_network_actor_mlp_2_bias.txt", std::ios::in);
    file[4].open(cur_path+"a2c_network_mu_weight.txt", std::ios::in);
    file[5].open(cur_path+"a2c_network_mu_bias.txt", std::ios::in);
    file[6].open(cur_path+"running_mean_std_running_mean.txt", std::ios::in);
    file[7].open(cur_path+"running_mean_std_running_var.txt", std::ios::in);
    file[8].open(cur_path+"a2c_network_critic_mlp_0_weight.txt", std::ios::in);
    file[9].open(cur_path+"a2c_network_critic_mlp_0_bias.txt", std::ios::in);
    file[10].open(cur_path+"a2c_network_critic_mlp_2_weight.txt", std::ios::in);
    file[11].open(cur_path+"a2c_network_critic_mlp_2_bias.txt", std::ios::in);
    file[12].open(cur_path+"a2c_network_value_weight.txt", std::ios::in);
    file[13].open(cur_path+"a2c_network_value_bias.txt", std::ios::in);

    file[14].open(cur_path+"a2c_network__disc_mlp_0_weight.txt", std::ios::in);
    file[15].open(cur_path+"a2c_network__disc_mlp_0_bias.txt", std::ios::in);
    file[16].open(cur_path+"a2c_network__disc_mlp_2_weight.txt", std::ios::in);
    file[17].open(cur_path+"a2c_network__disc_mlp_2_bias.txt", std::ios::in);
    file[18].open(cur_path+"a2c_network__disc_logits_weight.txt", std::ios::in);
    file[19].open(cur_path+"a2c_network__disc_logits_bias.txt", std::ios::in);
    file[20].open(cur_path+"amp_running_mean_std_running_mean.txt", std::ios::in);
    file[21].open(cur_path+"amp_running_mean_std_running_var.txt", std::ios::in);


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

    loadMatrix(file[14], disc_net_w0_);
    loadMatrix(file[15], disc_net_b0_);
    loadMatrix(file[16], disc_net_w2_);
    loadMatrix(file[17], disc_net_b2_);
    loadMatrix(file[18], disc_net_w_);
    loadMatrix(file[19], disc_net_b_);
    loadMatrix(file[20], disc_state_mean_);
    loadMatrix(file[21], disc_state_var_);    

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

    // Discriminator Network
    disc_net_w0_.resize(num_disc_hidden1, num_disc_state);
    disc_net_b0_.resize(num_disc_hidden1, 1);
    disc_net_w2_.resize(num_disc_hidden2, num_disc_hidden1);
    disc_net_b2_.resize(num_disc_hidden2, 1);    
    disc_net_w_.resize(disc_output, num_disc_hidden2);
    disc_net_b_.resize(disc_output, 1);

    disc_hidden_layer1_.resize(num_disc_hidden1, 1);
    disc_hidden_layer2_.resize(num_disc_hidden2, 1);
    disc_value_.resize(disc_output, 1);
    
    disc_state_buffer_.resize(num_disc_cur_state*num_disc_hist, 1);
    disc_state_cur_.resize(num_disc_cur_state, 1);
    disc_state_.resize(num_disc_state, 1);
    disc_state_mean_.resize(num_disc_state, 1);
    disc_state_var_.resize(num_disc_state, 1);


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
        2) root_rot: root rotation                  (3)     2:8
        3) root_vel: local linear velocity          (3)     8:11
        4) root_ang_vel: root angular velocity      (3)     11:14
        5) commands: x, y, yaw                      (3)     14:17
        6) dof_pos: dof position                    (12)    17:29
        7) dof_vel: dof velocity                    (12)    29:41
        8) action: action                           (12)    47:59
    */

    int data_idx = 0;
    
    // 1) root_h: root height (z)                  (1)     0 
    state_cur_(data_idx) = rd_cc_.q_virtual_(2);
    data_idx++;


    // 2) root_rot: root rotation                  (3)     2:8
    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);    

    euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());
    state_cur_(data_idx) = euler_angle_(0);
    data_idx++;
    state_cur_(data_idx) = euler_angle_(1);
    data_idx++;
    state_cur_(data_idx) = euler_angle_(2);
    data_idx++;


    // 4) root_vel: root linear velocity           (3)     8:11
    // 5) root_ang_vel: root angular velocity      (3)     11:14    
    for (int i=0; i<6; i++)
    {
        state_cur_(data_idx) = rd_cc_.q_dot_virtual_(i);
        data_idx++;
    }
    // Vector3d local_lin_vel_ = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(0,3));
    // for (int i=0; i<3; i++)
    // {
    //     disc_state_cur_(disc_data_idx) = local_lin_vel_(i);
    //     disc_data_idx++;
    // }


    // 6) commands: x, y, yaw                      (3)     14:17
    state_cur_(data_idx) = target_vel_x_;
    data_idx++;
    state_cur_(data_idx) = 0.0;
    data_idx++;
    state_cur_(data_idx) = target_vel_yaw_;
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


    // 10) action: action                           (12)    47:59
    for (int i = 0; i <num_actuator_action; i++) 
    {
        state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
        data_idx++;
    }
    

    state_buffer_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist-1),1) = state_buffer_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist-1),1);
    state_buffer_.block(num_cur_state*(num_state_skip*num_state_hist-1), 0, num_cur_state,1) = state_cur_;

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

    // Normalization of State
    // state_ = (state_ - state_mean_).array() / state_var_.cwiseSqrt().array();
    state_ = (state_ - state_mean_).array() / (state_var_.array() + 1e-05).sqrt();

}

void CustomController::processDiscriminator()
{
    /*
    1) root_h
    2) base euler
    3) q pos
    4) q vel
    5) local key pos
    */
    int disc_data_idx = 0;


    // 1) root_h
    disc_state_cur_(disc_data_idx) = rd_cc_.q_virtual_(2);
    disc_data_idx++;


    // 2) base euler
    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);

    Vector3d tan_vec, nor_vec;
    quatToTanNorm(q, tan_vec, nor_vec);
    for (int i = 0; i < 3; i++)
    {
        disc_state_cur_(disc_data_idx) = tan_vec(i);
        disc_data_idx++;
    }
    for (int i = 0; i < 3; i++)
    {
        disc_state_cur_(disc_data_idx) = nor_vec(i);
        disc_data_idx++;
    }

    // euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());
    // for (int i=0; i<3; i++)
    // {
    //     disc_state_cur_(disc_data_idx) = euler_angle_(i);
    //     disc_data_idx++;
    // }


    // local base vel and  local ang vel
    // Vector3d local_lin_vel_ = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(0,3));
    // for (int i=0; i<3; i++)
    // {
    //     disc_state_cur_(disc_data_idx) = local_lin_vel_(i);
    //     disc_data_idx++;
    // }
    // Vector3d local_ang_vel_ = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(3,3));
    // for (int i=0; i<3; i++)
    // {
    //     disc_state_cur_(disc_data_idx) = local_ang_vel_(i);
    //     disc_data_idx++;
    // }


    // 3) q pos
    for (int i = 0; i < 12; i++)
    {
        disc_state_cur_(disc_data_idx) = q_noise_(i);
        disc_data_idx++;
    }    


    // 4) q vel
    for (int i = 0; i < 12; i++)
    {
        disc_state_cur_(disc_data_idx) = q_vel_noise_(i);
        disc_data_idx++;
    }


    // 5) local key pos
    Vector3d global_lfoot_pos = rd_cc_.link_[Left_Foot].xpos;
    Vector3d global_rfoot_pos = rd_cc_.link_[Right_Foot].xpos;
    Vector3d local_lfoot_pos = quatRotateInverse(q, global_lfoot_pos - rd_cc_.q_virtual_.head(3));
    Vector3d local_rfoot_pos = quatRotateInverse(q, global_rfoot_pos - rd_cc_.q_virtual_.head(3));
    for (int i = 0; i < 3; i++)
    {
        disc_state_cur_(disc_data_idx) = local_lfoot_pos(i);
        disc_data_idx++;
    }
    for (int i = 0; i < 3; i++)
    {
        disc_state_cur_(disc_data_idx) = local_rfoot_pos(i);
        disc_data_idx++;
    }


    disc_state_buffer_.block(num_disc_cur_state, 0, num_disc_cur_state*(num_disc_hist-1),1) = disc_state_buffer_.block(0, 0, num_disc_cur_state*(num_disc_hist-1),1); 
    disc_state_buffer_.block(0, 0, num_disc_cur_state,1) = disc_state_cur_;

    disc_state_ = (disc_state_buffer_ - disc_state_mean_).array() / (disc_state_var_.array() + 1e-05).sqrt();   
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

    // Discriminator
    disc_hidden_layer1_ = disc_net_w0_ * disc_state_ + disc_net_b0_;
    for (int i = 0; i < num_disc_hidden1; i++) 
    {
        if (disc_hidden_layer1_(i) < 0)
            disc_hidden_layer1_(i) = 0.0;
    }

    disc_hidden_layer2_ = disc_net_w2_ * disc_hidden_layer1_ + disc_net_b2_;
    for (int i = 0; i < num_disc_hidden2; i++) 
    {
        if (disc_hidden_layer2_(i) < 0)
            disc_hidden_layer2_(i) = 0.0;
    }

    disc_value_ = (disc_net_w_ * disc_hidden_layer2_ + disc_net_b_);
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
            processDiscriminator();
            feedforwardPolicy();
            for (int i = 0; i < num_state_skip*num_state_hist; i++) 
            {
                // state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();
                state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1).setZero();
            }
            disc_state_buffer_.block(num_disc_cur_state, 0, num_disc_cur_state*(num_disc_hist-1),1)= disc_state_buffer_.block(0, 0, num_disc_cur_state*(num_disc_hist-1),1);
        }

        processNoise();

        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/250.0 - 1/10000.0)
        {
            processObservation();
            processDiscriminator();
            feedforwardPolicy();
            
            // action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*1/250.0, 0.0, 1/250.0);

            cout << "Value: " << value_ << endl;
            if (value_ < -10.0)
            {
                cout << "Value: " << value_ << endl;
                if (stop_by_value_thres_ == false)
                {
                    stop_by_value_thres_ = true;
                    stop_start_time_ = rd_cc_.control_time_us_;
                    q_stop_ = q_noise_;
                    std::cout << "Stop by Value Function" << std::endl;
                }
            }
            // soft max disc_value
            // double sum = exp(disc_value_(1)) + exp(disc_value_(2)) + exp(disc_value_(3));
            // Vector3d disc_value_softmax;
            // disc_value_softmax << exp(disc_value_(1))/sum, exp(disc_value_(2))/sum, exp(disc_value_(3))/sum;
            // cout << "Disc Value: " << disc_value_(0) << " " << disc_value_softmax.transpose() << endl;
            if (disc_value_(0) < -0.5)
            {
                cout << "Disc: " << disc_value_(0) << endl;
                if (stop_by_value_thres_ == false)
                {
                    stop_by_value_thres_ = true;
                    stop_start_time_ = rd_cc_.control_time_us_;
                    q_stop_ = q_noise_;
                    std::cout << "Stop by Disc Function" << std::endl;
                }
            }
            checkTouchDown();

            if (is_write_file_)
            {
                // for (int i = 0; i < 3; i++) {
                    // writeFile << rd_cc_.q_virtual_(i) << "\t";
                // }
                // for (int i = 0; i < 6; i++) {
                    // writeFile << rd_cc_.q_dot_virtual_(i) << "\t";
                // }

                // Eigen::Quaterniond q;
                // q.x() = rd_cc_.q_virtual_(3);
                // q.y() = rd_cc_.q_virtual_(4);
                // q.z() = rd_cc_.q_virtual_(5);
                // q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);
                // Eigen::Vector3d local_lin_vel = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(0,3));
                // writeFile << rd_cc_.q_dot_virtual_(0) << "\t";

                writeFile << target_vel_x_ << "\t";
                writeFile << desired_vel_x << "\t";       
                writeFile << local_lin_vel_(0) << "\t";
                writeFile << disc_value_(0) << "\t";

                writeFile << target_vel_yaw_ << "\t";
                writeFile << desired_vel_yaw << "\t";
                writeFile << rd_cc_.q_dot_virtual_(5) << "\t";

                // for (int i = 0; i < 12; i++) {
                //     writeFile << q_noise_(i) << "\t";
                // }
                // for (int i = 0; i < 12; i++) {
                //     writeFile << q_vel_noise_(i) << "\t";
                // }
                // print contact force
                // writeFile << -rd_cc_.LF_FT(2) << "\t" << -rd_cc_.RF_FT(2) << "\t";
                writeFile << -rd_cc_.LF_CF_FT(2) << "\t" << -rd_cc_.RF_CF_FT(2);
                for (int i = 0; i < num_actuator_action; i++) {
                    writeFile << "\t" << torque_rl_(i);
                }                
                writeFile << std::endl;
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
    LF_CF_FT_pre = rd_cc_.LF_CF_FT;
    RF_CF_FT_pre = rd_cc_.RF_CF_FT;
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
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[0], -0.5, 0.5);
    target_vel_y_ = 0.0; // DyrosMath::minmax_cut(joy->axes[1], -0.0, 0.0);
    target_vel_yaw_ = -DyrosMath::minmax_cut(joy->axes[2], -0.5, 0.5);
}

void CustomController::xBoxJoyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[1], -0.5, 0.5);
    target_vel_y_ = DyrosMath::minmax_cut(joy->axes[0], -0.0, 0.0);
    target_vel_yaw_ = DyrosMath::minmax_cut(joy->axes[3], -0.4, 0.4);
}

void CustomController::quatToTanNorm(const Eigen::Quaterniond& quaternion, Eigen::Vector3d& tangent, Eigen::Vector3d& normal) {
    // Reference direction and normal vectors
    Eigen::Vector3d refDirection(1, 0, 0); // Tangent vector reference
    Eigen::Vector3d refNormal(0, 0, 1);    // Normal vector reference

    // Rotate the reference vectors
    tangent = quaternion * refDirection;
    normal = quaternion * refNormal;

    // Normalize the vectors
    tangent.normalize();
    normal.normalize();
}

void CustomController::checkTouchDown() {
    // Check if the foot is in contact with the ground
    if (LF_CF_FT_pre(2) < 10.0 && rd_cc_.LF_CF_FT(2) > 10.0) {
        std::cout << "Left Foot Touch Down" << std::endl;
    }
    if (RF_CF_FT_pre(2) < 10.0 && rd_cc_.RF_CF_FT(2) > 10.0) {
        std::cout << "Right Foot Touch Down" << std::endl;
    }
}

Eigen::Vector3d CustomController::quatRotateInverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) {

    Eigen::Vector3d q_vec = q.vec();
    double q_w = q.w();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = 2.0 * q_w * q_vec.cross(v);
    Eigen::Vector3d c = 2.0 * q_vec * q_vec.dot(v);

    return a - b + c;
}