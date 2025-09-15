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
    nh_.getParam("/tocabi_cc/is_pd_control", pd_control_);
    
    if (is_write_file_)
    {
        if (is_on_robot_)
        {
            writeFile.open("/home/dyros/catkin_ws/src/tocabi_cc/result/"+weight_dir_+".csv", std::ofstream::out | std::ofstream::app);
        }
        else
        {
            writeFile.open("/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/result/data.csv", std::ofstream::out | std::ofstream::trunc);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    initVariable();
    loadOnnX();

    // joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy_gui", 10, &CustomController::joyCallback, this);
    xbox_joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy", 10, &CustomController::xBoxJoyCallback, this);
}

void CustomController::initVariable()
{    
    rl_action_.resize(num_action, 1);
    rl_action_pre_.resize(num_action, 1);
    torq_diff_.resize(num_action, 1);
    energy.resize(num_action, 1);

    state_cur_.resize(num_cur_state, 1);
    state_buffer_.resize(num_state, 1);

    q_dot_lpf_.setZero();

    torque_bound_ << 333, 232, 263, 289, 222, 166,
                    333, 232, 263, 289, 222, 166,
                    303, 303, 303, 
                    64, 64, 64, 64, 23, 23, 10, 10,
                    10, 10,
                    64, 64, 64, 64, 23, 23, 10, 10;  
                    
    q_init_ << 0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
    // q_init_ <<  0.0, 0.0, -0.5, 1.0, -0.5, 0.0,
                // 0.0, 0.0, -0.5, 1.0, -0.5, 0.0,
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
    kv_.diagonal() <<   15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;

    action_offset <<    0.0,  0.0, -0.25,  0.45, -0.15,  0.0,  
                        0.0,  0.0, -0.25,  0.45, -0.15,  0.0;
    action_scale  <<    0.3, 0.5, 0.75, 0.75, 0.65, 0.6, 
                        0.3, 0.5, 0.75, 0.75, 0.65, 0.6;
    // action_offset.setZero();
    // action_scale << M_PI/2, M_PI/4, M_PI/2, M_PI/2, M_PI/4, M_PI/4,
    //                 M_PI/2, M_PI/4, M_PI/2, M_PI/2, M_PI/4, M_PI/4;
}


void CustomController::loadOnnX()
{
    string cur_path = "/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/" + weight_dir_;
    // string cur_path = "/home/yong/ros1_ws/tocabi_ws/src/tocabi_cc/policy/" + weight_dir_;
    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/policy/" + weight_dir_;
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

    // test policy
    state_buffer_ ={1.9608e-01, -3.2585e-02,  6.0898e-02,  1.9302e-01, -3.0542e-02,
         4.3752e-02,  1.9118e-01, -2.8251e-02,  2.7609e-02,  1.9025e-01,
        -2.5714e-02,  1.2715e-02,  1.8983e-01, -2.2995e-02, -1.2361e-03,
        -1.6920e-01, -2.7193e-01,  4.9914e-01, -1.6177e-01, -2.8240e-01,
         5.0141e-01, -1.5521e-01, -2.8582e-01,  5.0137e-01, -1.4695e-01,
        -2.8365e-01,  4.9998e-01, -1.3631e-01, -2.7864e-01,  4.9839e-01,
         4.7279e-02,  7.9989e-02, -9.9567e-01,  4.4886e-02,  8.1414e-02,
        -9.9567e-01,  4.2451e-02,  8.2782e-02, -9.9566e-01,  4.0033e-02,
         8.4087e-02, -9.9565e-01,  3.7667e-02,  8.5304e-02, -9.9564e-01,
         8.2708e-01,  8.4433e-01,  8.6074e-01,  8.7631e-01,  8.9101e-01,
         2.2840e-01,  0.0000e+00,  4.8161e-01,  2.2840e-01,  0.0000e+00,
         4.7897e-01,  2.2840e-01,  0.0000e+00,  4.7633e-01,  2.2840e-01,
         0.0000e+00,  4.7370e-01,  2.2840e-01,  0.0000e+00,  4.7107e-01,
        -2.4610e-01,  4.5455e-03,  2.4576e-01, -1.5237e-01, -1.1984e-01,
         8.4797e-02,  6.8922e-02,  2.3734e-02, -1.5042e-01,  1.9421e-01,
        -1.9378e-01,  1.1410e-02, -2.5117e-01,  5.1932e-03,  2.5436e-01,
        -1.5783e-01, -1.1983e-01,  8.5068e-02,  8.0420e-02,  1.6936e-02,
        -1.6662e-01,  1.9164e-01, -1.8567e-01,  1.5656e-02, -2.5625e-01,
         5.6905e-03,  2.6249e-01, -1.6230e-01, -1.2032e-01,  8.5396e-02,
         9.0806e-02,  1.0014e-02, -1.8239e-01,  1.8849e-01, -1.7716e-01,
         2.0192e-02, -2.6133e-01,  6.0388e-03,  2.7012e-01, -1.6582e-01,
        -1.2128e-01,  8.5782e-02,  1.0013e-01,  3.0940e-03, -1.9771e-01,
         1.8476e-01, -1.6834e-01,  2.5074e-02, -2.6638e-01,  6.2143e-03,
         2.7724e-01, -1.6846e-01, -1.2269e-01,  8.6239e-02,  1.0847e-01,
        -3.7359e-03, -2.1250e-01,  1.8044e-01, -1.5932e-01,  3.0278e-02,
        -5.1723e-01,  9.1155e-02,  8.9680e-01, -6.3272e-01,  3.1432e-02,
        -5.2664e-02,  1.2350e+00, -6.6426e-01, -1.6495e+00, -2.1604e-01,
         7.7561e-01,  4.0128e-01, -5.1889e-01,  7.3807e-02,  8.5478e-01,
        -5.3224e-01, -1.9438e-02, -4.6605e-02,  1.1211e+00, -6.8913e-01,
        -1.6113e+00, -2.7051e-01,  8.2109e-01,  4.2715e-01, -5.1919e-01,
         5.8603e-02,  8.0754e-01, -4.3359e-01, -6.9138e-02, -3.8301e-02,
         1.0110e+00, -6.9727e-01, -1.5686e+00, -3.2818e-01,  8.5868e-01,
         4.5695e-01, -5.1811e-01,  4.2626e-02,  7.5603e-01, -3.3826e-01,
        -1.1535e-01, -2.9779e-02,  9.0647e-01, -6.9435e-01, -1.5211e+00,
        -3.8702e-01,  8.8693e-01,  4.9264e-01, -5.1559e-01,  2.3728e-02,
         7.0496e-01, -2.5094e-01, -1.5831e-01, -1.7401e-02,  8.0942e-01,
        -6.8320e-01, -1.4671e+00, -4.4516e-01,  9.0589e-01,  5.2360e-01,
        -2.2087e-01,  9.9615e-02, -1.8590e-01,  8.2063e-01, -2.1423e-01,
         1.6019e-01,  4.6371e-02,  2.1793e-02, -7.0565e-01,  1.2000e+00,
        -6.5580e-01,  2.6194e-02, -2.2537e-01,  9.1871e-02, -1.9521e-01,
         8.0479e-01, -2.2496e-01,  1.5557e-01,  5.9959e-02,  2.4091e-02,
        -7.1215e-01,  1.2000e+00, -6.4890e-01,  3.2230e-02, -2.3014e-01,
         8.5779e-02, -2.0575e-01,  7.8718e-01, -2.3647e-01,  1.5198e-01,
         7.3705e-02,  2.4522e-02, -7.1830e-01,  1.2000e+00, -6.4237e-01,
         3.8725e-02, -2.3498e-01,  8.0840e-02, -2.1570e-01,  7.6975e-01,
        -2.4725e-01,  1.4806e-01,  8.7344e-02,  2.3209e-02, -7.2331e-01,
         1.2000e+00, -6.3573e-01,  4.4474e-02, -2.3995e-01,  7.7108e-02,
        -2.2448e-01,  7.5312e-01, -2.5688e-01,  1.4345e-01,  1.0021e-01,
        -2.2448e-01,  7.5312e-01, -2.5688e-01,  1.4345e-01,  1.0021e-01,
         2.0559e-02, -7.2748e-01,  1.2000e+00, -6.2875e-01,  4.9381e-02};
    
    std::copy(state_buffer_.begin(), state_buffer_.end(), input_states_buffer[input_obs_idx_].begin());

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
    std::cout << "RL Action: " << rl_action_.transpose() << std::endl;
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
        // std::random_device rd;  
        // std::mt19937 gen(rd());
        // std::uniform_real_distribution<> dis(-0.00001, 0.00001);
        // for (int i = 0; i < MODEL_DOF; i++) {
        //     q_noise_(i) = rd_cc_.q_virtual_(6+i) + dis(gen);
        // }
        // if (time_cur_ - time_pre_ > 0.0)
        // {
        //     q_vel_noise_ = (q_noise_ - q_noise_pre_) / (time_cur_ - time_pre_);
        //     q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        // }
        // else
        // {
        //     q_vel_noise_ = q_vel_noise_;
        //     q_dot_lpf_ = q_dot_lpf_;
        // }
        // q_noise_pre_ = q_noise_;
        q_noise_ = rd_cc_.q_virtual_.segment(6,MODEL_DOF);
        q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
    }
    time_pre_ = time_cur_;
}

void CustomController::processObservation()
{
    int data_idx = 0;
    
    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);  

    // linear and angular velocity in base frame
    Vector3d local_lin_vel_ = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(0,3));
    Vector3d local_ang_vel_ = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(3,3));
    for (int i = 0; i < 3; i++)
    {state_cur_[data_idx++] = local_lin_vel_(i);}
    for (int i = 0; i < 3; i++)
    {state_cur_[data_idx++] = local_ang_vel_(i);}

    // projected gravity vector
    Eigen::Vector3d projected_gravity = quatRotateInverse(q, Eigen::Vector3d(0, 0, -1.0));
    for (int i = 0; i < 3; i++)
    {state_cur_[data_idx++] = projected_gravity(i);}

    // clock input
    state_cur_[data_idx++] = sin(2 * M_PI * time_cur_ / step_time_); // sin wave with period of 10 seconds
    // state_cur_[data_idx++] = cos(2 * M_PI * time_cur_ / step_time_); // cos wave with period of 10 seconds
    state_cur_[data_idx++] = -sin(2 * M_PI * time_cur_ / step_time_); // cos wave with period of 10 seconds

    // Velocity Commands
    state_cur_[data_idx++] = 0.3;
    state_cur_[data_idx++] = 0.0;
    state_cur_[data_idx++] = 0.0;
    // state_cur_[data_idx++] = target_vel_x_;
    // state_cur_[data_idx++] = target_vel_y_;
    // state_cur_[data_idx++] = target_vel_yaw_;

    for (int i = 0; i < 12; i++)
    {
        state_cur_[data_idx++] = q_noise_(i) - q_init_(i);;
    }

    for (int i = 0; i < 12; i++)
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

    for (int i = 0; i < 12; i++) 
    {
        if (pd_control_)
            state_cur_[data_idx++] = target_pos(i);
        else
            state_cur_[data_idx++] = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
    }

    // Shift the buffer to the left and add the new state at the end
    std::copy(state_buffer_.begin()+num_cur_state, state_buffer_.end(), state_buffer_.begin());
    std::copy(state_cur_.begin(), state_cur_.end(), state_buffer_.end()-num_cur_state);

    // update the input tensor for ONNX feedforward
    std::copy(state_buffer_.begin(), state_buffer_.end(), input_states_buffer[input_obs_idx_].begin());
}

void CustomController::feedforwardPolicy()
{
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
    // output tensor to value_
    value_ = output_tensors[1].GetTensorMutableData<float>()[0];

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
            time_pre_ = time_cur_ - 0.01;
            // time_inference_pre_ = rd_cc_.control_time_us_ - (1/249.9)*1e6;
            time_inference_pre_ = rd_cc_.control_time_us_ - (1/99.9)*1e6;
            // ft_left_init_ = abs(rd_cc_.LF_FT(2));
            // ft_right_init_ = abs(rd_cc_.RF_FT(2));

            rd_.tc_init = false;
            std::cout<<"cc mode 7"<<std::endl;
            torque_init_ = rd_cc_.torque_desired;

            processNoise();
            processObservation();
            feedforwardPolicy();
        }

        processNoise();

        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        // if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/250.0 - 1/10000.0)
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/100.0)
        {
            processObservation();
            feedforwardPolicy();

            if (value_ < 1.0 && use_value_stop == true)
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
                writeFile << time_cur_ << "\t";
                writeFile << rd_cc_.q_virtual_.segment(3,3).transpose() << "\t";
                writeFile << rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1) << "\t";
                writeFile << rd_cc_.q_dot_virtual_.segment(0,3).transpose() << "\t";
                writeFile << rd_cc_.q_dot_virtual_.segment(3,3).transpose() << "\t" << endl;
            }
            time_inference_pre_ = rd_cc_.control_time_us_;
        }
        // compute lower body torque from policy output
        for (int i = 0; i < num_actuator_action; i++)
        {
            if (pd_control_) {
                float action_value = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
                target_pos(i) = action_offset(i) + action_value * action_scale(i);
                torque_rl_(i) = kp_(i,i) / 9.0 * (target_pos(i) - q_noise_(i)) - kv_(i,i) / 3.0 * q_vel_noise_(i);
            }
            else {
                torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i)*torque_bound_(i), -torque_bound_(i), torque_bound_(i));
            }
        }
        // compute upper body torque - maintain the initial pose
        for (int i = num_actuator_action; i < MODEL_DOF; i++)
        {
            torque_rl_(i) = kp_(i,i) / 9.0 * (q_init_(i) - q_noise_(i)) - kv_(i,i) / 3.0 * q_vel_noise_(i);
        }

        // send torque command
        if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
        {
            for (int i = 0; i <MODEL_DOF; i++)
            {
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.1e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);
            }
            rd_.torque_desired = torque_spline_;
        }
        else
            rd_.torque_desired = torque_rl_;
            // cout << "Torque Desired: " << rd_.torque_desired.transpose() << endl;

        // stop by value function -> maintain the stop position
        if (stop_by_value_thres_)
            rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_*q_vel_noise_;
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
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[0]*0.8, -0.8, 0.8);
    // target_vel_y_ = 0.0;
    target_vel_y_ = DyrosMath::minmax_cut(joy->axes[1]*0.4, -0.4, 0.4);
    target_vel_yaw_ = -DyrosMath::minmax_cut(joy->axes[2]*0.3, -0.3, 0.3);
}

void CustomController::xBoxJoyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[1]*0.5, -0.5, 0.5);
    target_vel_y_ = DyrosMath::minmax_cut(joy->axes[0], -0.3, 0.3);
    target_vel_yaw_ = DyrosMath::minmax_cut(joy->axes[3]*0.4, -0.4, 0.4);
    if (joy->buttons[5] == 1){ // A button
        step_time_ += 0.01;
        step_time_ = DyrosMath::minmax_cut(step_time_, 1.0, 2.0);
        cout << "Step time: " << step_time_ << endl;
    }
    if (joy->buttons[4] == 1){ // B button
        step_time_ -= 0.01;
        step_time_ = DyrosMath::minmax_cut(step_time_, 1.0, 2.0);
        cout << "Step time: " << step_time_ << endl;
    }
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