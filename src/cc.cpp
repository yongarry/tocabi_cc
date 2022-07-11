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
            writeFile.open("/home/kim/tocabi_ws/src/tocabi_cc/result/data.csv", std::ofstream::out | std::ofstream::app);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    initVariable();
    loadNetwork();
}

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}

void CustomController::loadNetwork()
{
    state_.setZero();
    rl_action_.setZero();

    string cur_path = "/home/kim/tocabi_ws/src/tocabi_cc/weight/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/weight/";
    }
    std::ifstream file[8];
    file[0].open(cur_path+"mlp_extractor_policy_net_0_weight.txt", std::ios::in);
    file[1].open(cur_path+"mlp_extractor_policy_net_0_bias.txt", std::ios::in);
    file[2].open(cur_path+"mlp_extractor_policy_net_2_weight.txt", std::ios::in);
    file[3].open(cur_path+"mlp_extractor_policy_net_2_bias.txt", std::ios::in);
    file[4].open(cur_path+"action_net_weight.txt", std::ios::in);
    file[5].open(cur_path+"action_net_bias.txt", std::ios::in);
    file[6].open(cur_path+"obs_mean.txt", std::ios::in);
    file[7].open(cur_path+"obs_variance.txt", std::ios::in);

    if(!file[0].is_open())
    {
        std::cout<<"Can not find the weight file"<<std::endl;
    }

    float temp;
    int row = 0;
    int col = 0;

    while(!file[0].eof() && row != policy_net_w0_.rows())
    {
        file[0] >> temp;
        if(temp != '\n')
        {
            policy_net_w0_(row, col) = temp;
            col ++;
            if (col == policy_net_w0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[1].eof() && row != policy_net_b0_.rows())
    {
        file[1] >> temp;
        if(temp != '\n')
        {
            policy_net_b0_(row, col) = temp;
            col ++;
            if (col == policy_net_b0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[2].eof() && row != policy_net_w2_.rows())
    {
        file[2] >> temp;
        if(temp != '\n')
        {
            policy_net_w2_(row, col) = temp;
            col ++;
            if (col == policy_net_w2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[3].eof() && row != policy_net_b2_.rows())
    {
        file[3] >> temp;
        if(temp != '\n')
        {
            policy_net_b2_(row, col) = temp;
            col ++;
            if (col == policy_net_b2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[4].eof() && row != action_net_w_.rows())
    {
        file[4] >> temp;
        if(temp != '\n')
        {
            action_net_w_(row, col) = temp;
            col ++;
            if (col == action_net_w_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[5].eof() && row != action_net_b_.rows())
    {
        file[5] >> temp;
        if(temp != '\n')
        {
            action_net_b_(row, col) = temp;
            col ++;
            if (col == action_net_b_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[6].eof() && row != state_mean_.rows())
    {
        file[6] >> temp;
        if(temp != '\n')
        {
            state_mean_(row, col) = temp;
            col ++;
            if (col == state_mean_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[7].eof() && row != state_var_.rows())
    {
        file[7] >> temp;
        if(temp != '\n')
        {
            state_var_(row, col) = temp;
            col ++;
            if (col == state_var_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
}

void CustomController::initVariable()
{    
    policy_net_w0_.resize(num_hidden, num_state);
    policy_net_b0_.resize(num_hidden, 1);
    policy_net_w2_.resize(num_hidden, num_hidden);
    policy_net_b2_.resize(num_hidden, 1);
    action_net_w_.resize(num_action, num_hidden);
    action_net_b_.resize(num_action, 1);
    hidden_layer1_.resize(num_hidden, 1);
    hidden_layer2_.resize(num_hidden, 1);
    rl_action_.resize(num_action, 1);
    
    state_.resize(num_state, 1);
    state_mean_.resize(num_state, 1);
    state_var_.resize(num_state, 1);

    q_dot_lpf_.setZero();
    euler_angle_lpf_.setZero();
    q_lpf_ = rd_.q_virtual_.segment(6,MODEL_DOF);
    // q_noise_ = q_noise_pre_ = rd_.q_virtual_.segment(6,MODEL_DOF);
    q_init_ << 0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, 0.0,
                0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                0.0, 0.0,
                -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0;
    q_noise_pre_ = q_noise_ = q_init_;

    torque_bound_ << 333, 232, 263, 289, 222, 166,
                    333, 232, 263, 289, 222, 166,
                    303, 303, 303, 
                    64, 64, 64, 64, 23, 23, 10, 10,
                    10, 10,
                    64, 64, 64, 64, 23, 23, 10, 10;  

    Kp_.setZero();
    Kv_.setZero();
    double kp_scale = 2.0;
    Kp_.diagonal() << 2000.0/ kp_scale, 5000.0/ kp_scale, 4000.0/ kp_scale, 3700.0/ kp_scale, 3200.0/ kp_scale, 3200.0/ kp_scale,
                        2000.0/ kp_scale, 5000.0/ kp_scale, 4000.0/ kp_scale, 3700.0/ kp_scale, 3200.0/ kp_scale, 3200.0/ kp_scale,
                        6000.0/ kp_scale, 10000.0/ kp_scale, 10000.0/ kp_scale,
                        400.0/ kp_scale, 1000.0/ kp_scale, 400.0/ kp_scale, 400.0/ kp_scale, 400.0/ kp_scale, 400.0/ kp_scale, 100.0/ kp_scale, 100.0/ kp_scale,
                        100.0/ kp_scale, 100.0/ kp_scale,
                        400.0/ kp_scale, 1000.0/ kp_scale, 400.0/ kp_scale, 400.0/ kp_scale, 400.0/ kp_scale, 400.0/ kp_scale, 100.0/ kp_scale, 100.0/ kp_scale;
    double kv_scale = 1.0;
    Kv_.diagonal() << 15.0/ kv_scale, 50.0/ kv_scale, 20.0/ kv_scale, 25.0/ kv_scale, 24.0/ kv_scale, 24.0/ kv_scale,
                        15.0/ kv_scale, 50.0/ kv_scale, 20.0/ kv_scale, 25.0/ kv_scale, 24.0/ kv_scale, 24.0/ kv_scale,
                        200.0/ kv_scale, 100.0/ kv_scale, 100.0/ kv_scale,
                        10.0/ kv_scale, 28.0/ kv_scale, 10.0/ kv_scale, 10.0/ kv_scale, 10.0/ kv_scale, 10.0/ kv_scale, 3.0/ kv_scale, 3.0/ kv_scale,
                        2.0/ kv_scale, 2.0/ kv_scale,
                        10.0/ kv_scale, 28.0/ kv_scale, 10.0/ kv_scale, 10.0/ kv_scale, 10.0/ kv_scale, 10.0/ kv_scale, 3.0/ kv_scale, 3.0/ kv_scale;
}

void CustomController::processNoise()
{
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.00001, 0.00001);
    for (int i = 0; i < MODEL_DOF; i++) {
        q_noise_(i) = rd_.q_virtual_(6+i) + dis(gen);
    }
    q_vel_noise_ = (q_noise_ - q_noise_pre_) * 2000.0;
    q_noise_pre_ = q_noise_;
    q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 2000.0, 1.0);
}

void CustomController::processObservation()
{
    int data_idx = 0;

    Eigen::Quaterniond q;
    q.x() = rd_.q_virtual_(3);
    q.y() = rd_.q_virtual_(4);
    q.z() = rd_.q_virtual_(5);
    q.w() = rd_.q_virtual_(MODEL_DOF_QVIRTUAL-1);    

    euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());
    euler_angle_lpf_ =euler_angle_; // DyrosMath::lpf<3>(euler_angle_, euler_angle_lpf_, 2000, 10.0);

    state_(data_idx) = euler_angle_lpf_(0);
    data_idx++;

    state_(data_idx) = euler_angle_lpf_(1);
    data_idx++;

    q_lpf_ = rd_.q_virtual_.segment(6,MODEL_DOF); //DyrosMath::lpf<MODEL_DOF>(rd_.q_virtual_.segment(6,MODEL_DOF), q_lpf_, 2000, 10.0);

    // q_lpf_(23) = 0.0;
    // q_lpf_(24) = 0.0;
    // q_lpf_(22) = 0.0;
    // q_lpf_(32) = 0.0;

    for (int i = 0; i < MODEL_DOF; i++)
    {
        state_(data_idx) = q_lpf_(i);
        data_idx++;
    }

    // q_dot_lpf_(23) = 0.0;
    // q_dot_lpf_(24) = 0.0;
    // q_dot_lpf_(22) = 0.0;
    // q_dot_lpf_(32) = 0.0;
    for (int i = 0; i < MODEL_DOF; i++)
    {
        if (is_on_robot_)
        {
            state_(data_idx) = q_dot_lpf_(i);
        }
        else
        {
            state_(data_idx) =  q_dot_lpf_(i); //rd_.q_dot_virtual_(i+6); q_vel_noise_(i);
        }
        data_idx++;
    }

    float squat_duration = 8.0;
    float phase = std::fmod((rd_.control_time_us_-start_time_)/1e6, squat_duration) / squat_duration;
    state_(data_idx) = sin(2*M_PI*phase);
    data_idx++;
    state_(data_idx) = cos(2*M_PI*phase);
    data_idx++;
}

void CustomController::feedforwardPolicy()
{
    for (int i = 0; i <num_state; i++)
    {
        state_(i) = (state_(i) - state_mean_(i)) / sqrt(state_var_(i) + 1.0e-08);
        state_(i) = DyrosMath::minmax_cut(state_(i), -3.0, 3.0);
    }
    
    hidden_layer1_ = policy_net_w0_ * state_ + policy_net_b0_;
    for (int i = 0; i < num_hidden; i++) 
    {
        if (hidden_layer1_(i) < 0)
            hidden_layer1_(i) = 0.0;
    }

    hidden_layer2_ = policy_net_w2_ * hidden_layer1_ + policy_net_b2_;
    for (int i = 0; i < num_hidden; i++) 
    {
        if (hidden_layer2_(i) < 0)
            hidden_layer2_(i) = 0.0;
    }

    rl_action_ = action_net_w_ * hidden_layer2_ + action_net_b_;
    
}

void CustomController::computeSlow()
{
    if (rd_.tc_.mode == 11)
    {

        if (rd_.tc_init)
        {
            //Initialize settings for Task Control! 
            start_time_ = rd_.control_time_us_;

            rd_.tc_init = false;
            std::cout<<"cc mode 11"<<std::endl;
            torque_init_ = rd_.torque_desired;
        } 

        processNoise();
        
        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        if ((rd_.control_time_us_ - time_inference_pre_)/1e6 > 1/50.0)
        {
            processObservation();
            feedforwardPolicy();
            time_inference_pre_ = rd_.control_time_us_;
        }


        for (int i = 0; i < MODEL_DOF; i++)
        {
            torque_rl_(i) = Kp_(i,i) * (rl_action_(i)*3.14/180.0 + q_init_(i) - q_noise_(i)) - Kv_(i,i) * q_vel_noise_(i);
            torque_rl_(i) = DyrosMath::minmax_cut(torque_rl_(i), -torque_bound_(i), torque_bound_(i));
        }
        
        if (rd_.control_time_us_ < start_time_ + 1e6)
        {
            for (int i = 0; i <MODEL_DOF; i++)
            {
                torque_spline_(i) = DyrosMath::cubic(rd_.control_time_us_, start_time_, start_time_ + 1e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);
            }
            rd_.torque_desired = torque_spline_;
        }
        else
        {
            rd_.torque_desired = torque_rl_;
        }
        
        if (is_write_file_)
        {
            if ((rd_.control_time_us_ - time_inference_pre_)/1e6 > 1/250)
            {
                writeFile << (rd_.control_time_us_ - start_time_)/1e6 << "\t";
                writeFile << euler_angle_.transpose() << "\t";
                writeFile << euler_angle_lpf_.transpose() << "\t";
                writeFile << q_lpf_.transpose() << "\t";
                writeFile << rd_.q_dot_virtual_.segment(6,MODEL_DOF).transpose() << "\t";
                writeFile << q_dot_lpf_.transpose() << "\t";
                writeFile << q_vel_noise_.transpose() << "\t";
                writeFile << rd_.torque_desired.transpose() << std::endl;

                time_inference_pre_ = rd_.control_time_us_;
            }
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