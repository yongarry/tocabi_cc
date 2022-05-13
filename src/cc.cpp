#include "cc.h"

using namespace TOCABI;

CustomController::CustomController(RobotData &rd) : rd_(rd) //, wbc_(dc.wbc_)
{
    ControlVal_.setZero();

    // writeFile.open("/home/kim/tocabi_ws/src/tocabi_cc/result/data.csv", std::ofstream::out | std::ofstream::app);
    // writeFile << std::fixed << std::setprecision(8);

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

void CustomController::processObservation()
{
    int data_idx = 0;

    Eigen::Quaterniond q;
    q.x() = rd_.q_virtual_(3);
    q.y() = rd_.q_virtual_(4);
    q.z() = rd_.q_virtual_(5);
    q.w() = rd_.q_virtual_(MODEL_DOF_QVIRTUAL-1);    

    Eigen::Vector3d euler_angle;
    euler_angle = DyrosMath::rot2Euler_tf(q.toRotationMatrix());

    state_(data_idx) = euler_angle(0);
    data_idx++;

    state_(data_idx) = euler_angle(1);
    data_idx++;

    q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_dot_lpf_, rd_.q_dot_virtual_.segment(6,MODEL_DOF), 2000, 3.0);

    for (int i = 0; i < MODEL_DOF; i++)
    {
        // state_(data_idx) = q_dot_lpf_(i);
        state_(data_idx) = rd_.q_dot_virtual_(i+6);
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

    for (int i = 0; i < MODEL_DOF; i++)
    {
        rl_action_(i) = DyrosMath::minmax_cut(rl_action_(i), -300., 300.);
    }
    
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

        } 

        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        if ((rd_.control_time_us_ - time_inference_pre_)/1e6 > 1/250)
        {
            processObservation();
            feedforwardPolicy();
            time_inference_pre_ = rd_.control_time_us_;
        }

        rd_.torque_desired = rl_action_.cast <double> ();

        // writeFile << rd_.torque_desired.transpose() << std::endl;
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