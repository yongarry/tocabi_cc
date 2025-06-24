#include "cc.h"

using namespace TOCABI;

CustomController::CustomController(RobotData &rd) 
    :   rd_(rd), //, wbc_(dc.wbc_)
        env(ORT_LOGGING_LEVEL_WARNING, "tocabi"),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        session(nullptr)
{    
    ControlVal_.setZero();

    nh_.getParam("/tocabi_cc/weight_dir", weight_dir_);
    
    if (is_write_file_)
    {
        if (is_on_robot_)
        {
            writeFile.open("/home/dyros/catkin_ws/src/tocabi_cc/result/"+weight_dir_+".csv", std::ofstream::out | std::ofstream::app);
        }
        else
        {
            writeFile.open("/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/result/"+weight_dir_+"data.csv", std::ofstream::out | std::ofstream::trunc);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    initVariable();
    loadOnnX();

    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy_gui", 10, &CustomController::joyCallback, this);
    // xbox_joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy", 10, &CustomController::xBoxJoyCallback, this);
}

void CustomController::initVariable()
{    
    rl_action_.resize(num_action, 1);
    rl_action_pre_.resize(num_action, 1);
    torq_diff_.resize(num_action, 1);
    energy.resize(num_action, 1);

    state_cur_.resize(num_cur_state, 1);
    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 1);

    if (is_hist_encoder_) { 
        state_long_hist_.resize(num_hist_state * num_cur_state, 1); 
        state_long_hist_buffer_.resize(num_long_hist_len * num_cur_state, 1);
    }

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

    action_offset_.diagonal() << 0.0,  0.0, -0.25,  0.45, -0.15,  0.0,  
                                 0.0,  0.0, -0.25,  0.45, -0.15,  0.0;
    action_scale_.diagonal()  << 0.42, 0.7, 1.05, 1.05, 0.91, 0.84, 
                                 0.42, 0.7, 1.05, 1.05, 0.91, 0.84;
}


void CustomController::loadOnnX()
{
    string cur_path = "/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/" + weight_dir_;
    // string cur_path = "/home/yong/ros1_ws/tocabi_ws/src/tocabi_cc/policy/" + weight_dir_;
    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/" + weight_dir_;
    }

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.use_deterministic_compute", "1");

    session = Ort::Session(env, cur_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    input_number = session.GetInputCount();
    output_number = session.GetOutputCount();

    input_names.resize(input_number);
    output_names.resize(output_number);

    input_names_char.resize(input_names.size());
    output_names_char.resize(output_names.size());

    for (size_t i = 0; i < input_number; i++) {
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        input_names[i] = input_name.get();
    }
    for (size_t i = 0; i < output_number; i++) {
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        output_names[i] = output_name.get();
    }

    // Print input/output names
    std::cout << "Input names: "; 
    std::copy(input_names.begin(), input_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Output names: ";
    std::copy(output_names.begin(), output_names.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    for (size_t i = 0; i < input_names.size(); ++i) { 
        input_names_char[i] = input_names[i].c_str();
        if (input_names_char[i] == "obs") {input_obs_idx_ = i;}
    }
    for (size_t i = 0; i < output_names.size(); ++i) { output_names_char[i] = output_names[i].c_str();}

    // Initialize input tensors
    for (size_t i = 0; i < input_number; ++i) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();
        cout << "Input " << i << " shape: " << input_shape.size() << endl;
        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
        input_states_buffer.push_back(std::move(input_tensor_values));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer.back().data(),
            input_states_buffer.back().size(),
            input_shape.data(),
            input_shape.size()));
    }
        
}

void CustomController::processNoise()
{
    time_cur_ = rd_cc_.control_time_us_ / 1e6;
    q_vel_noise_pre_ = q_vel_noise_;
    rl_action_pre_ = rl_action_;
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
    int data_idx = 0;
    
    // state_cur_[data_idx++] = rd_cc_.q_virtual_(2);

    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);    

    euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());
    state_cur_[data_idx++] = euler_angle_(0);
    state_cur_[data_idx++] = euler_angle_(1);
    state_cur_[data_idx++] = euler_angle_(2);

    for(int i = 0; i < 6; i++)
    {
        state_cur_[data_idx++] = rd_cc_.q_dot_virtual_(i);
    }
    // Vector3d local_lin_vel_ = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(0,3));
    // for (int i=0; i<3; i++)
    // {
    //     state_cur_[data_idx++] = local_lin_vel_(i);
    // }
    // Vector3d local_ang_vel_ = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(3,3));
    // for (int i=0; i<3; i++)
    // {
    //     state_cur_[data_idx++] = rd_cc_.q_dot_virtual_(i+3);
    // }
    // for (int i = 3; i < 6; i++)
    // {
    //     state_cur_[data_idx++] = rd_cc_.q_dot_virtual_(i);
    // }

    // if (rd_cc_.control_time_us_ < start_time_ + 5.0e6)
    // {
    //     desired_vel_x = 0.4;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 10.0e6)
    // {
    //     desired_vel_x = -0.3;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 15.0e6)
    // {
    //     desired_vel_x = 0.4;
    // }
    // else
    // {
    //     desired_vel_x = 0.0;
    // // }
    // desired_vel_x = target_vel_x_;
    // state_cur_[data_idx++] = desired_vel_x;
    // state_cur_[data_idx++] = 0.0;
    // state_cur_[data_idx++] = 0.0;

    // desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 5.0e6, 0.0, 0.3, 0.0, 0.0);
    // state_cur_[data_idx++] = desired_vel_x;

    // desired_vel_x = 0.3;
    // desired_vel_yaw = 0.0;

    // state_cur_[data_idx++] = desired_vel_x;
    // state_cur_[data_idx++] = 0.0;
    // state_cur_[data_idx++] = desired_vel_yaw;

    state_cur_[data_idx++] = target_vel_x_;
    state_cur_[data_idx++] = target_vel_y_;
    state_cur_[data_idx++] = target_vel_yaw_;

    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_[data_idx++] = q_noise_(i);
    }

    for (int i = 0; i < num_actuator_action; i++)
    {
        if (is_on_robot_)
        {
            state_cur_[data_idx++] = q_vel_noise_(i);
        }
        else
        {
            state_cur_[data_idx++] = q_vel_noise_(i); //rd_cc_.q_dot_virtual_(i+6); //q_vel_noise_(i);
        }
    }

    for (int i = 0; i <num_actuator_action; i++) 
    {
        state_cur_[data_idx++] = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
    }
    
    size_t buffer_size = num_cur_state*num_state_skip*num_state_hist;
    std::copy(state_buffer_.begin() + num_cur_state, state_buffer_.end(), state_buffer_.begin());
    std::copy(state_cur_.begin(), state_cur_.end(), state_buffer_.begin() + buffer_size - num_cur_state);

    // Internal State First
    for (size_t i = 0; i < num_state_hist; ++i) {
        std::copy(state_buffer_.begin() + num_cur_state * (num_state_skip * (i + 1) - 1),
                  state_buffer_.begin() + num_cur_state * (num_state_skip * (i + 1) - 1) + num_cur_internal_state,
                  input_states_buffer[input_obs_idx_].begin() + num_cur_internal_state * i);
    }

    // Action History Second
    for (size_t i = 0; i < num_state_hist - 1; ++i) {
        std::copy(state_buffer_.begin() + num_cur_state * (num_state_skip * (i + 1)) + num_cur_internal_state,
                  state_buffer_.begin() + num_cur_state * (num_state_skip * (i + 1)) + num_cur_internal_state + num_action,
                  input_states_buffer[input_obs_idx_].begin() + num_state_hist * num_cur_internal_state + num_action * i);
    }

    if (is_hist_encoder_){
        std::copy(state_long_hist_.begin() + num_cur_state, state_long_hist_.end(), state_long_hist_.begin());
        std::copy(state_cur_.begin(), state_cur_.end(), state_long_hist_.begin() + num_hist_state * num_cur_state - num_cur_state);

        for (size_t i = 0; i < num_long_hist_len; ++i) {
            std::copy(state_long_hist_.begin() + num_cur_state * (num_long_hist_skip * (i + 1) - 1),
                      state_long_hist_.begin() + num_cur_state * (num_long_hist_skip * (i + 1)),
                      state_long_hist_buffer_.begin() + num_cur_state * i);
        }
        // transpose state_long_hist_buffer_(50,49) to input_states_buffer_(49,50)
        for (size_t i = 0; i < num_long_hist_len; ++i) {
            for (size_t j = 0; j < num_cur_state; ++j) {
                input_states_buffer[0][j * num_long_hist_len + i] = state_long_hist_buffer_[i * num_cur_state + j];
            }
        }

    }

}

void CustomController::feedforwardPolicy()
{
    // std::fill(input_states_buffer[0].begin(), input_states_buffer[0].end(), 0.0);
    // std::fill(input_states_buffer[1].begin(), input_states_buffer[1].end(), 0.0);
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_number, output_names_char.data(), output_number);

    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (!output_tensors[i].IsTensor()) {
            std::cerr << "Output " << i << " is not a valid tensor." << std::endl;
            continue;
        }
    }

    // output tensor to rl_action_
    for (size_t i = 0; i < num_action; i++) {
        rl_action_(i) = output_tensors[0].GetTensorMutableData<float>()[i];
    }
    // cout << "RL Action: " << rl_action_.transpose() << endl;
    // output tensor to value_
    value_ = output_tensors[2].GetTensorMutableData<float>()[0];

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
            feedforwardPolicy();
            for (int i = 0; i < num_state_skip*num_state_hist; i++) 
            {
                std::fill(state_buffer_.begin() + num_cur_state * i, state_buffer_.begin() + num_cur_state * (i + 1), 0.0);
                // std::copy(state_cur_.begin(), state_cur_.begin(), state_buffer_.begin() + num_cur_state * i);
            }
            if (is_hist_encoder_)
            {
                for (size_t i = 0; i < num_hist_state; ++i) {
                    std::fill(state_long_hist_.begin() + num_cur_state * i, state_long_hist_.begin() + num_cur_state * (i + 1), 0.0);
                    // std::copy(state_cur_.begin(), state_cur_.end(), state_long_hist_.begin() + i * num_cur_state);
                }
            }   
        }

        processNoise();

        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        // if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/250.0 - 1/10000.0)
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/100.0)
        {
            processObservation();
            feedforwardPolicy();
            
            if (value_ < 100.0)
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
            if (is_write_file_)
            {
                // writeFile << rd_cc_.q_virtual_(2) << "\t";
                // writeFile << rd_cc_.q_dot_virtual_(2) << "\t";
                // writeFile << desired_vel_x << "\t";
                // writeFile << -rd_cc_.LF_CF_FT(2) << "\t" << -rd_cc_.RF_CF_FT(2);
                
                for (int i = 0; i < num_actuator_action; i++) {
                    torq_diff_(i) = (rl_action_(i) - rl_action_pre_(i))*torque_bound_(i);
                    energy(i) = rl_action_(i) * torque_bound_(i) * q_vel_noise_(i);
                }                
                writeFile << rd_cc_.control_time_ << "\t";
                writeFile << torq_diff_.norm() << "\t";
                writeFile << (q_vel_noise_ - q_vel_noise_pre_).norm() << "\t";
                writeFile << q_vel_noise_.norm() << "\t";
                writeFile << energy.sum() << "\t";
                writeFile << std::pow((desired_vel_x - rd_cc_.q_dot_virtual_(0)),2) + std::pow((desired_vel_yaw - rd_cc_.q_dot_virtual_(5)),2);
                
                writeFile << std::endl;
            }


            
            time_inference_pre_ = rd_cc_.control_time_us_;
        }
        Vector12d target_pos;
        for (int i = 0; i < num_actuator_action; i++)
        {
            // torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i)*torque_bound_(i), -torque_bound_(i), torque_bound_(i));
            target_pos(i) = action_offset_(i,i) + rl_action_(i) * action_scale_(i,i);
        }
        for (int i = 0; i < num_actuator_action; i++)
        {
            torque_rl_(i) = kp_(i,i) / 9.0 * (target_pos(i) - q_noise_(i)) - kv_(i,i) / 3.0 * q_vel_noise_(i);
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
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[0]*0.5, -0.5, 0.5);
    target_vel_y_ = 0.0; // DyrosMath::minmax_cut(joy->axes[1], -0.0, 0.0);
    target_vel_yaw_ = -DyrosMath::minmax_cut(joy->axes[2]*0.3, -0.3, 0.3);
}

void CustomController::xBoxJoyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[1]*0.5, -0.5, 0.5);
    target_vel_y_ = DyrosMath::minmax_cut(joy->axes[0], -0.0, 0.0);
    target_vel_yaw_ = DyrosMath::minmax_cut(joy->axes[3]*0.5, -0.4, 0.4);
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

Eigen::Vector3d CustomController::quatRotateInverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) {

    Eigen::Vector3d q_vec = q.vec();
    double q_w = q.w();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = 2.0 * q_w * q_vec.cross(v);
    Eigen::Vector3d c = 2.0 * q_vec * q_vec.dot(v);

    return a - b + c;
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

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}