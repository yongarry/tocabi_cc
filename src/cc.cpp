#include "cc.h"
#include <utility>

using namespace TOCABI;

void loadConfig(const std::string& path, bool& write_file, std::string& weight, std::string& cmd, int& cmd_mode, int& policy_mode, double& hz, double& preview_height)
{
    try {
        YAML::Node cfg = YAML::LoadFile(path);
        if (cfg["write_file"]) write_file = cfg["write_file"].as<bool>();
        if (cfg["weight"]) weight = cfg["weight"].as<std::string>();
        if (cfg["cmd"]) cmd = cfg["cmd"].as<std::string>();
        if (cfg["cmd_mode"]) cmd_mode = cfg["cmd_mode"].as<int>();
        if (cfg["policy_mode"]) policy_mode = cfg["policy_mode"].as<int>();
        if (cfg["hz"]) hz = cfg["hz"].as<double>();
        if (cfg["preview_height"]) preview_height = cfg["preview_height"].as<double>();
    }
    catch (const YAML::Exception& e) {
        ROS_WARN_STREAM("tocabi_cc: YAML parse error in " << path << ": " << e.what());}
    catch (const std::exception& e) {
        ROS_WARN_STREAM("tocabi_cc: Cannot open config  " << path << ": " << e.what());}
}

CustomController::CustomController(RobotData &rd)
    :   rd_(rd), //, wbc_(dc.wbc_)
        env(ORT_LOGGING_LEVEL_WARNING, "tocabi"),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        session(nullptr)
{
    std::string cfg_path = workspace_dir_ + "config/tocabi_cc.yaml";
    if (is_on_robot_)
        cfg_path = "/home/dyros/catkin_ws/src/tocabi_cc/config/tocabi_cc.yaml";
    loadConfig(cfg_path, write_file_, weight_file_, cmd_file_, cmd_mode_, policy_mode, hz_, vrp_height_);
    writeFile.open("/home/yong20/ros_ws/ros1/footsteptocabi_ws/src/tocabi_cc/result/log.txt", ofstream::out);
    if (is_on_robot_)
        writeFile.open("/home/dyros/catkin_ws/src/tocabi_cc/result/log.txt", ofstream::out);
    initVariable();
    std::cout << "Load network start\n" << std::endl;
    loadNetwork();
    std::cout << "Load network end\n" << std::endl;

    del_t = 1.0 / hz_;
    preview_horizon_ = 2.0 * hz_;
    preview_ctrl_ = std::make_unique<PreviewController>(del_t, preview_horizon_);
    preview_ctrl_->init(vrp_height_);

    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &CustomController::joyCallback, this);
}

void CustomController::loadNetwork()
{
    state_.resize(num_state, 0);
    rl_action_.resize(num_actuator_action, 1);

    string cur_path = workspace_dir_ + "policy/" + weight_file_;
    if (is_on_robot_)
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/policy/" + weight_file_;

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
    state_.resize(num_state, 0.0);
    std::copy(state_.begin(), state_.end(), input_states_buffer[input_obs_idx_].begin());
    // forward policy
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_number, output_names_char.data(), output_number);
    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (!output_tensors[i].IsTensor()) {
            std::cerr << "Output " << i << " is not a valid tensor." << std::endl;
            continue;
        }
    }

    // output tensor to rl_action_
    for (size_t i = 0; i < num_actuator_action; i++) {
        rl_action_(i) = output_tensors[0].GetTensorMutableData<float>()[i];
    }
    // value_ = output_tensors[1].GetTensorMutableData<float>()[0];
    std::cout << "RL Action: " << rl_action_.transpose() << std::endl;
    std::cout << "Value: " << value_ << std::endl;
}

void CustomController::initVariable()
{    
    rl_action_.resize(num_actuator_action, 1);

    state_.resize(num_state, 0);
    state_cur_.resize(num_cur_state, 0);
    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 0);

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

    kp_.setZero(); kv_.setZero();
    kp_.diagonal() <<   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        6000.0, 10000.0, 10000.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
                        100.0, 100.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0;
    kp_.diagonal() /= 9.0;  
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;
    kv_.diagonal() /= 3.0;

    q_lower_limit_ << -0.3, -0.5, -1.0, -0.3, -0.8, -0.6, -0.3, -0.5, -1.0, -0.3, -0.8, -0.6;
    q_upper_limit_ <<  0.3,  0.5,  0.5,  1.2,  0.5,  0.6,  0.3,  0.5,  0.5,  1.2,  0.5,  0.6;

    loadCommands();
    q_leg_desired_ = q_init_.segment(0, num_actuator_action);
    foot_commands_.setZero(number_of_foot_step, 9);
    phase_indicator_.setZero(number_of_foot_step);
    t_total_.setZero(number_of_foot_step);
    com_z_command_.setZero(number_of_foot_step + 1);

    target_com_state_stance_.setZero(9);
    target_com_state_global_.setZero(9);
    target_com_state_global_.segment(0, 3) = rd_cc_.link_[COM_id].xpos;

    initBias();
}

void CustomController::loadCommands()
{
    string cur_path = workspace_dir_ + "cmd/" + cmd_file_;
    if (is_on_robot_)
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/cmd/" + cmd_file_;
    // Read CSV file and fill foot_commands_planner_ (step x 9) + optional comz row
    ifstream file(cur_path);
    if (!file.is_open()) {
        cerr << "Failed to open command file: " << cur_path << std::endl;
        return;
    }

    auto split = [](const string& s) {
        vector<string> out;
        stringstream ss(s);
        string cell;
        while (getline(ss, cell, ','))
            out.push_back(cell);
        return out;
    };

    auto all_empty = [](const vector<string>& cells){
        return all_of(cells.begin(), cells.end(), [](const string& c){ return c.empty(); });};

    number_of_planner_step = 0;
    vector<pair<string, vector<double>>> command_rows;
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;
        auto cells = split(line);
        if (cells.empty() || all_empty(cells)) continue;
        if (cells[0].empty()) continue;
        if (number_of_planner_step <= 0)
            number_of_planner_step = static_cast<int>(cells.size()) - 1;
        vector<double> values(number_of_planner_step, 0.0);
        for (int i = 0; i < number_of_planner_step && i + 1 < static_cast<int>(cells.size()); ++i)
            values[i] = cells[i + 1].empty() ? 0.0 : stod(cells[i + 1]);
        command_rows.emplace_back(cells[0], move(values));
    }

    foot_commands_planner_.setZero(number_of_planner_step, 9);
    com_z_planner_.setZero(number_of_planner_step);

    static const char* kFootKeys[9] = {
        "posx", "posy", "posz", "rotr", "rotp", "roty", "tssp", "tdsp", "foot"
    };
    for (size_t r = 0; r < command_rows.size(); ++r) {
        const string& key = command_rows[r].first;
        const vector<double>& values = command_rows[r].second;
        bool matched = false;
        for (int cmd = 0; cmd < 9; ++cmd) {
            if (key == kFootKeys[cmd]) {
                for (int step = 0; step < number_of_planner_step; ++step)
                    foot_commands_planner_(step, cmd) = values[step];
                matched = true;
                break;
            }
        }
        if (matched) continue;
        if (key == "comz" || key == "com_z") {
            for (int step = 0; step < number_of_planner_step; ++step)
                com_z_planner_(step) = values[step];
            continue;
        }
        // Backward compatible: first 9 unnamed/positional rows -> foot columns
        if (r < 9) {
            for (int step = 0; step < number_of_planner_step; ++step)
                foot_commands_planner_(step, static_cast<int>(r)) = values[step];
        } else if (r == 9) {
            for (int step = 0; step < number_of_planner_step; ++step)
                com_z_planner_(step) = values[step];
        }
    }

    cout << "Number of Planner Step: " << number_of_planner_step << endl;
    cout << "Foot Commands Planner: \n" << foot_commands_planner_ << endl;
    cout << "CoM Z Planner: " << com_z_planner_.transpose() << endl;
}

void CustomController::fillComZFromPlanner(int start_idx)
{
    const int LA = number_of_foot_step;
    const int N = number_of_planner_step;
    auto at = [&](int idx) -> double {
        if (N <= 0) return 0.0;
        if (idx < 0) return com_z_planner_(0);
        if (idx < N) return com_z_planner_(idx);
        return com_z_planner_(N - 1);
    };
    for (int s = 0; s < LA; ++s)
        com_z_command_(s) = at(start_idx + s);
    // Extra lookahead slot (LA+1), same convention as train / g1 fill_global_buffer_
    com_z_command_(LA) = at(start_idx + LA - 1);
}

void CustomController::processNoise()
{
    time_cur_ = rd_cc_.control_time_us_ / 1e6;
    if (is_on_robot_)
    {
        q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);
        q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
        if (time_cur_ - time_pre_ > 0.0)
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        else
            q_dot_lpf_ = q_dot_lpf_;
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
        // q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);
        // q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
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
    
    // 1. base lin vel, ang vel (LPF updated in processNoise)
    Vector3d base_lin_vel = q.conjugate()*(rd_cc_.q_dot_virtual_.segment(0,3));
    Vector3d base_ang_vel = (rd_cc_.q_dot_virtual_.segment(3,3));

    for (int i = 0; i < 3; i++)
        state_cur_[data_idx++] = base_lin_vel(i);
    for (int i = 0; i < 3; i++)
        state_cur_[data_idx++] = base_ang_vel(i);
    
    // 2. projected gravity
    Vector3d grav, projected_grav;
    grav << 0, 0, -1.;
    projected_grav = q.conjugate()*grav;
    for (int i = 0; i < 3; i++)
        state_cur_[data_idx++] = projected_grav(i);

    // 3. joint positions relative to initial position
    for (int i = 0; i < num_actuator_action; i++)
        state_cur_[data_idx++] = q_noise_(i) - q_init_(i);

    // 4. joint velocities
    for (int i = 0; i < num_actuator_action; i++)
    {
        if (is_on_robot_)
            state_cur_[data_idx++] = q_vel_noise_(i);
        else
            state_cur_[data_idx++] = q_vel_noise_(i); //rd_cc_.q_dot_virtual_(i+6);
    }

    // // 5. target joint positions
    for (int i = 0; i < num_actuator_action; i++)
        state_cur_[data_idx++] = q_leg_desired_(i);

    // 6. phase input
    if (planner_index_ > number_of_planner_step + 1)
        walking_tick = 0;
    state_cur_[data_idx++] = cos(float(walking_tick) / float(t_total_(0)) * 2 * M_PI);
    state_cur_[data_idx++] = sin(float(walking_tick) / float(t_total_(0)) * 2 * M_PI);

    // 7. LIPM foot commands
    for (int i = 0; i < 9; i++){
        state_cur_[data_idx++] = foot_commands_(0, i);
        // if (i == 1) cout << "foot_commands_(0, 1): " << foot_commands_(0, i) << endl;
    }
    // 8. com_z_command (matches train command[..., 23] / foot_commands_w_comz)
    // state_cur_[data_idx++] = com_z_command_(0);
    // 9. previous action
    for (int i = 0; i <num_actuator_action; i++) 
        state_cur_[data_idx++] = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);

    for (int i = 0; i < 3; i++)
        writeFile << base_lin_vel(i) << "\t";
    for (int i = 0; i < 3; i++)
        writeFile << base_ang_vel(i) << "\t";
    for (int i = 0; i < 3; i++)
        writeFile << projected_grav(i) << "\t";
    for (int i = 0; i < num_actuator_action; i++)
        writeFile << q_noise_(i) << "\t";
    for (int i = 0; i < num_actuator_action; i++)
        writeFile << q_vel_noise_(i) << "\t";
    for (int i = 0; i < num_actuator_action; i++)
        writeFile << q_leg_desired_(i) << "\t";
    for (int i = 0; i < 9; i++)
        writeFile << foot_commands_(0, i) << "\t";
    writeFile << com_z_command_(0) << "\t";
    
    std::copy(state_buffer_.begin() + num_cur_state, state_buffer_.end(), state_buffer_.begin());
    std::copy(state_cur_.begin(), state_cur_.end(), state_buffer_.begin() + num_cur_state*(num_state_skip*num_state_hist-1));

    for (int i = 0; i < num_state_hist; i++)
        std::copy(state_buffer_.begin() + num_cur_state*(num_state_skip*(i+1)-1), state_buffer_.begin() + num_cur_state*(num_state_skip*(i+1)-1) + num_cur_state, state_.begin() + num_cur_state*i);
}

void CustomController::feedforwardPolicy()
{
    // update the input tensor for ONNX feedforward
    std::copy(state_.begin(), state_.end(), input_states_buffer[input_obs_idx_].begin());
    // forward policy
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_number, output_names_char.data(), output_number);
    // check output tensor
    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (!output_tensors[i].IsTensor()) {
            std::cerr << "Output " << i << " is not a valid tensor." << std::endl;
            continue;
        }
    }
    // output tensor to rl_action_
    for (size_t i = 0; i < num_actuator_action; i++) {
        rl_action_(i) = output_tensors[0].GetTensorMutableData<float>()[i];
        rl_action_(i) = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
    }

    // output tensor to value_
    // value_ = output_tensors[1].GetTensorMutableData<float>()[0];
}

void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{   
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
            q_leg_desired_ = rd_cc_.q_.segment(0,12);
            time_cur_ = start_time_ / 1e6;
            time_pre_ = time_cur_ - 0.005;
            time_inference_pre_ = rd_cc_.control_time_us_ - (1/(hz_))*1e6;

            rd_.tc_init = false;
            std::cout<<"cc mode 7"<<std::endl;
            torque_init_ = rd_cc_.torque_desired;

            updateCommand();
            updateRobotStates();
            generateVRP();
            Matrix3d init_preview_state = Matrix3d::Zero();
            init_preview_state.row(0) = com_pos_state_stance_;
            preview_ctrl_->update_state(init_preview_state);
            generateCoM();
            generateFeet();
            getTargetJointPos();
            processNoise();
            processBias();
            processObservation();
            for (int i = 0; i < num_state_skip*num_state_hist; i++) 
                std::copy(state_cur_.begin(), state_cur_.end(), state_buffer_.begin() + num_cur_state*i);
            feedforwardPolicy();
        }
        processNoise();
        processBias();
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/hz_) // 125 is the control frequency
        {
            if (walking_tick > t_total_(0))
            {
                updateCommand();
                updateRobotStates();
                generateVRP();
            }
            updateRobotStates();
            generateCoM();
            generateFeet();
            getTargetJointPos();
            if (write_file_)
            {
                for (int i = 0; i < 3; i++)
                    writeFile << vrp_ref_(walking_tick, i) << "\t";
                for (int i = 0; i < 3; i++)
                    writeFile << target_com_state_stance_(i) << "\t";
                for (int i = 0; i < 3; i++)
                    writeFile << target_pelvis_stance_.translation()(i) << "\t";
                for (int i = 0; i < 3; i++)
                    writeFile << target_lfoot_stance_.translation()(i) << "\t";
                for (int i = 0; i < 3; i++)
                    writeFile << target_rfoot_stance_.translation()(i) << "\t";
                for (int i = 0; i < 6; i++)
                    writeFile << q_leg_desired_(i) << "\t";
                for (int i = 0; i < 6; i++)
                    writeFile << q_target_(i) << "\t";
                for (int i = 0; i < 6; i++)
                    writeFile << q_noise_(i) << "\t";
                for (int i = 0; i < 6; i++)
                    writeFile << rd_.torque_desired(i) << "\t";
                for (int i = 0; i < 2; i++)
                    writeFile << rd_cc_.link_[Pelvis].xpos(i) << "\t";
                writeFile << DyrosMath::rot2Euler(rd_cc_.link_[Pelvis].rotm)(2) << "\t";
                for (int i = 0; i < 2; i++)
                    writeFile << rd_cc_.link_[COM_id].xpos(i) << "\t";
                writeFile << DyrosMath::rot2Euler(rd_cc_.link_[COM_id].rotm)(2) << "\t";
                for (int i = 0; i < 2; i++)
                    writeFile << rd_cc_.link_[Left_Foot].xpos(i) << "\t";
                writeFile << DyrosMath::rot2Euler(rd_cc_.link_[Left_Foot].rotm)(2) << "\t";
                for (int i = 0; i < 2; i++)
                    writeFile << rd_cc_.link_[Right_Foot].xpos(i) << "\t";
                writeFile << DyrosMath::rot2Euler(rd_cc_.link_[Right_Foot].rotm)(2) << "\t";
                writeFile << endl;
            }

            processNoise();
            processObservation();
            feedforwardPolicy();
            action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_actuator_action-1)*5/hz_, 0.0, 5/hz_);
            if (value_ < 0.0)
            {
                if (stop_by_value_thres_ == false)
                {
                    stop_by_value_thres_ = true;
                    stop_start_time_ = rd_cc_.control_time_us_;
                    q_stop_ = q_noise_;
                    std::cout << "Stop by Value Function : " << walking_tick << ", Value : " << value_ << std::endl;
                }
            }
            time_inference_pre_ = rd_cc_.control_time_us_;
            walking_tick++;
        }

        for (int i = 0; i < num_actuator_action; i++){
            // q_target_(i) = 0.5 * (q_upper_limit_(i) + q_lower_limit_(i)) + 0.5 * (q_upper_limit_(i) - q_lower_limit_(i)) * rl_action_(i);
            // torque_rl_(i) = kp_(i,i) * (q_target_(i) - q_noise_(i)) - kv_(i,i) * q_vel_noise_(i);
            // torque_rl_(i) = kp_(i,i) * (q_leg_desired_(i) - q_noise_(i)) - kv_(i,i) * q_vel_noise_(i);
            if (policy_mode == 0)
                torque_rl_(i) = rl_action_(i)*torque_bound_(i);
            else{
                // torque_rl_(i) = kp_(i,i) * (q_leg_desired_(i) - q_noise_(i)) - kv_(i,i) * q_vel_noise_(i) + rl_action_(i)*torque_bound_(i);
                torque_rl_(i) = kp_(i,i) * (rl_action_(i) + q_init_(i) - q_noise_(i)) - kv_(i,i) * q_vel_noise_(i);
            }
        }

        for (int i = num_actuator_action; i < MODEL_DOF; i++)
            torque_rl_(i) = kp_(i,i) * (q_init_(i) - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i);

        if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
        {
            for (int i = 0; i <MODEL_DOF; i++)
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.1e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);
            rd_.torque_desired = torque_spline_;    
        }
        else
            rd_.torque_desired = torque_rl_;

        if (stop_by_value_thres_)
            rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_*q_vel_noise_;
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

void CustomController::updateCommand()
{
    if (cmd_mode_ == 0) {
        if (walking_tick == 0){
            foot_commands_ = foot_commands_planner_.block(0, 0, number_of_foot_step, 9);
            planner_index_ = number_of_foot_step;
            if (is_right_stance_first) phase_indicator_(0) = 1;
            else phase_indicator_(0) = 0;
            t_total_(0) = floor((foot_commands_(0, 6) + foot_commands_(0, 7)*2) * hz_); // dsp + ssp + dsp 
            if (phase_indicator_(0) == 0) foot_commands_(0, 1) *= -1; // if left foot stance, y cmd should be negative
            // if (phase_indicator_(0) == 0) foot_commands_(0, 5) *= -1; // if left foot stance, yaw cmd should be negative

            for (int i = 1; i < number_of_foot_step; i++){
                phase_indicator_(i) = 1 - phase_indicator_(i-1);
                t_total_(i) = floor((foot_commands_(i, 6) + foot_commands_(i, 7)*2) * hz_); // dsp + ssp + dsp 
                if (phase_indicator_(i) == 0) foot_commands_(i, 1) *= -1; // if left foot stance, y cmd should be negative
                // if (phase_indicator_(i) == 0) foot_commands_(i, 5) *= -1; // if left foot stance, yaw cmd should be negative
            }
            fillComZFromPlanner(0);
        }
        else if (walking_tick > t_total_(0)){
            cout << "================================================" << endl;
            cout << "Foot Position Error  : " << swing_state_stance_.translation().transpose() - foot_commands_.row(0).segment(0, 3) << endl;
            cout << "Next Foot Commands   : " << foot_commands_.row(0).segment(0, 3) << "\t" << foot_commands_.row(0)(5) << endl;

            foot_commands_.block(0, 0, number_of_foot_step - 1, 9) = foot_commands_.block(1, 0, number_of_foot_step - 1, 9);
            if (planner_index_ < number_of_planner_step) 
                foot_commands_.block(number_of_foot_step - 1, 0, 1, 9) = foot_commands_planner_.block(planner_index_, 0, 1, 9);
            else
                foot_commands_.block(number_of_foot_step - 1, 0, 1, 9) << 0.0, 0.205, 0, 0, 0, 0, 0.9, 0.15, 0.08;
            phase_indicator_.segment(0, number_of_foot_step - 1) = phase_indicator_.segment(1, number_of_foot_step - 1);
            phase_indicator_(number_of_foot_step - 1) = 1 - phase_indicator_(number_of_foot_step - 2);
            t_total_.segment(0, number_of_foot_step - 1) = t_total_.segment(1, number_of_foot_step - 1);
            t_total_(number_of_foot_step - 1) = floor((foot_commands_(number_of_foot_step - 1, 6) + foot_commands_(number_of_foot_step - 1, 7)*2) * hz_); // dsp + ssp + dsp 
            if (phase_indicator_(number_of_foot_step - 1) == 0) foot_commands_(number_of_foot_step - 1, 1) *= -1; // if left foot stance, y cmd should be negative
            // if (phase_indicator_(number_of_foot_step - 1) == 0) foot_commands_(number_of_foot_step - 1, 5) *= -1; // if left foot stance, yaw cmd should be negative

            planner_index_++;
            fillComZFromPlanner(planner_index_ - number_of_foot_step);
            walking_tick = 0;
        }
    }
    else if (cmd_mode_ == 1) {
        if (walking_tick == 0){
            lfoot_global_state_.segment(0,2) = rd_cc_.link_[Left_Foot].xpos.segment(0,2);
            rfoot_global_state_.segment(0,2) = rd_cc_.link_[Right_Foot].xpos.segment(0,2);
            lfoot_global_state_(2) = DyrosMath::rot2Euler(rd_cc_.link_[Left_Foot].rotm)(2);
            rfoot_global_state_(2) = DyrosMath::rot2Euler(rd_cc_.link_[Right_Foot].rotm)(2);

            current_step_number_ = 0;
            planner_index_ = number_of_foot_step;

            for (int i = 0; i < number_of_foot_step; i++){
                if (i == 0) phase_indicator_(i) = is_right_stance_first ? 1 : 0;
                else phase_indicator_(i) = 1 - phase_indicator_(i-1);

                foot_commands_(i, 3) = foot_commands_planner_(i, 3);
                foot_commands_(i, 4) = foot_commands_planner_(i, 4);
                foot_commands_(i, 6) = foot_commands_planner_(i, 6);
                foot_commands_(i, 7) = foot_commands_planner_(i, 7);
                foot_commands_(i, 8) = foot_commands_planner_(i, 8);
                t_total_(i) = floor((foot_commands_(i, 6) + foot_commands_(i, 7)*2) * hz_);
            }

            // global foothold → local stance frame
            Eigen::Vector3d &stance = is_right_stance_first ? rfoot_global_state_ : lfoot_global_state_;
            double x_len = foot_commands_planner_(0, 0) - stance(0);
            double y_len = foot_commands_planner_(0, 1) - stance(1);
            foot_commands_(0, 0) = cos(-stance(2))*x_len - sin(-stance(2))*y_len;
            foot_commands_(0, 1) = sin(-stance(2))*x_len + cos(-stance(2))*y_len;
            foot_commands_(0, 2) = foot_commands_planner_(0, 2);
            foot_commands_(0, 5) = wrap_to_pi(foot_commands_planner_(0, 5) - stance(2));

            for (int i = 1; i < number_of_foot_step; i++){
                x_len = foot_commands_planner_(i, 0) - foot_commands_planner_(i-1, 0);
                y_len = foot_commands_planner_(i, 1) - foot_commands_planner_(i-1, 1);
                foot_commands_(i, 0) = cos(-foot_commands_planner_(i-1, 5))*x_len - sin(-foot_commands_planner_(i-1, 5))*y_len;
                foot_commands_(i, 1) = sin(-foot_commands_planner_(i-1, 5))*x_len + cos(-foot_commands_planner_(i-1, 5))*y_len;
                foot_commands_(i, 2) = foot_commands_planner_(i, 2) - foot_commands_planner_(i-1, 2);
                foot_commands_(i, 5) = wrap_to_pi(foot_commands_planner_(i, 5) - foot_commands_planner_(i-1, 5));
            }
            fillComZFromPlanner(0);
        }
        else if (walking_tick > t_total_(0)){
            cout << "================================================" << endl;
            cout << "Foot Position Error  : " << swing_state_stance_.translation().transpose() - foot_commands_.row(0).segment(0, 3) << endl;
            // Vector3d swing_pos = swing_state_stance_.translation();
            // double swing_yaw = DyrosMath::rot2Euler(swing_state_stance_.linear())(2);
            // cout << "Foot Position Error  : " << sqrt(pow(swing_pos(0) - foot_commands_(0, 0), 2) + pow(swing_pos(1) - foot_commands_(0, 1), 2)) << " [m]" << endl;
            // cout << ">> X error : " << (swing_pos(0) - foot_commands_(0, 0)) << " [m]" << endl;
            // cout << ">> Y error : " << (swing_pos(1) - foot_commands_(0, 1)) << " [m]" << endl;
            // cout << "Foot Yaw error : " << (wrap_to_pi(swing_yaw - foot_commands_(0, 5))) << " [rad]" << endl;

            foot_commands_.block(0, 0, number_of_foot_step - 1, 9) = foot_commands_.block(1, 0, number_of_foot_step - 1, 9);
            phase_indicator_.segment(0, number_of_foot_step - 1) = phase_indicator_.segment(1, number_of_foot_step - 1);
            phase_indicator_(number_of_foot_step - 1) = 1 - phase_indicator_(number_of_foot_step - 2);
            t_total_.segment(0, number_of_foot_step - 1) = t_total_.segment(1, number_of_foot_step - 1);

            planner_index_++;
            current_step_number_++;
            walking_tick = 0;

            if (current_step_number_ < number_of_planner_step) {
                lfoot_global_state_.segment(0,2) = rd_cc_.link_[Left_Foot].xpos.segment(0,2);
                rfoot_global_state_.segment(0,2) = rd_cc_.link_[Right_Foot].xpos.segment(0,2);
                lfoot_global_state_(2) = DyrosMath::rot2Euler(rd_cc_.link_[Left_Foot].rotm)(2);
                rfoot_global_state_(2) = DyrosMath::rot2Euler(rd_cc_.link_[Right_Foot].rotm)(2);
                Eigen::Vector3d &stance = (phase_indicator_(0) == 1) ? rfoot_global_state_ : lfoot_global_state_;

                double x_len = foot_commands_planner_(current_step_number_, 0) - stance(0);
                double y_len = foot_commands_planner_(current_step_number_, 1) - stance(1);

                foot_commands_(0, 0) = cos(-stance(2))*x_len - sin(-stance(2))*y_len;
                foot_commands_(0, 1) = sin(-stance(2))*x_len + cos(-stance(2))*y_len;
                foot_commands_(0, 2) = foot_commands_planner_(current_step_number_, 2) - foot_commands_planner_(current_step_number_-1, 2);
                foot_commands_(0, 3) = foot_commands_planner_(current_step_number_, 3);
                foot_commands_(0, 4) = foot_commands_planner_(current_step_number_, 4);
                foot_commands_(0, 5) = wrap_to_pi(foot_commands_planner_(current_step_number_, 5) - stance(2));
                foot_commands_(0, 6) = foot_commands_planner_(current_step_number_, 6);
                foot_commands_(0, 7) = foot_commands_planner_(current_step_number_, 7);
                foot_commands_(0, 8) = foot_commands_planner_(current_step_number_, 8);
                t_total_(0) = floor((foot_commands_(0, 6) + foot_commands_(0, 7)*2) * hz_);

                for (int step = 1; step < number_of_foot_step; step++){
                    int planned_idx = step + current_step_number_;
                    if (planned_idx < number_of_planner_step){
                        x_len = foot_commands_planner_(planned_idx, 0) - foot_commands_planner_(planned_idx-1, 0);
                        y_len = foot_commands_planner_(planned_idx, 1) - foot_commands_planner_(planned_idx-1, 1);
                        foot_commands_(step, 0) = cos(-foot_commands_planner_(planned_idx-1, 5))*x_len - sin(-foot_commands_planner_(planned_idx-1, 5))*y_len;
                        foot_commands_(step, 1) = sin(-foot_commands_planner_(planned_idx-1, 5))*x_len + cos(-foot_commands_planner_(planned_idx-1, 5))*y_len;
                        foot_commands_(step, 2) = foot_commands_planner_(planned_idx, 2) - foot_commands_planner_(planned_idx-1, 2);
                        foot_commands_(step, 3) = foot_commands_planner_(planned_idx, 3);
                        foot_commands_(step, 4) = foot_commands_planner_(planned_idx, 4);
                        foot_commands_(step, 5) = wrap_to_pi(foot_commands_planner_(planned_idx, 5) - foot_commands_planner_(planned_idx-1, 5));
                        foot_commands_(step, 6) = foot_commands_planner_(planned_idx, 6);
                        foot_commands_(step, 7) = foot_commands_planner_(planned_idx, 7);
                        foot_commands_(step, 8) = foot_commands_planner_(planned_idx, 8);
                        t_total_(step) = floor((foot_commands_(step, 6) + foot_commands_(step, 7)*2) * hz_);
                    }
                    else {
                        foot_commands_.row(step) << 0.0, 0.205, 0, 0, 0, 0, 0.9, 0.15, 0.08;
                        if (phase_indicator_(step) == 0) foot_commands_(step, 1) *= -1;
                        t_total_(step) = floor((foot_commands_(step, 6) + foot_commands_(step, 7)*2) * hz_);
                    }
                }
                cout << "Next Foot Commands   : " << foot_commands_.row(0).segment(0, 3) << " " << foot_commands_.row(0)(5) << endl;
            }
            else {
                int step = number_of_foot_step - 1;
                foot_commands_.row(step) << 0.0, 0.205, 0, 0, 0, 0, 0.9, 0.15, 0.08;
                if (phase_indicator_(step) == 0) foot_commands_(step, 1) *= -1;
                t_total_(step) = floor((foot_commands_(step, 6) + foot_commands_(step, 7)*2) * hz_);
            }
            fillComZFromPlanner(current_step_number_);
        }
    }
}

void CustomController::updateRobotStates()
{
    WBC::SetContact(rd_, 1 - phase_indicator_(0), phase_indicator_(0));

    pelvis_state_global_.translation() = rd_cc_.link_[Pelvis].xpos;
    pelvis_state_global_.linear() = rd_cc_.link_[Pelvis].rotm;
    com_pos_state_global_ = rd_cc_.link_[COM_id].xpos;
    com_vel_state_global_ = rd_cc_.link_[COM_id].v;
    lfoot_global_current_.translation() = rd_cc_.link_[Left_Foot].xpos;
    lfoot_global_current_.linear() = rd_cc_.link_[Left_Foot].rotm;
    rfoot_global_current_.translation() = rd_cc_.link_[Right_Foot].xpos;
    rfoot_global_current_.linear() = rd_cc_.link_[Right_Foot].rotm;

    if (phase_indicator_(0) == 0) {
        stance_foot_state_global_.translation() = rd_cc_.link_[Left_Foot].xpos;
        stance_foot_state_global_.linear() = DyrosMath::rotateWithZ(DyrosMath::rot2Euler(rd_cc_.link_[Left_Foot].rotm)(2));
        swing_foot_state_global_.translation() = rd_cc_.link_[Right_Foot].xpos;
        swing_foot_state_global_.linear() = DyrosMath::rotateWithZ(DyrosMath::rot2Euler(rd_cc_.link_[Right_Foot].rotm)(2));
    }
    else {
        stance_foot_state_global_.translation() = rd_cc_.link_[Right_Foot].xpos;
        stance_foot_state_global_.linear() = DyrosMath::rotateWithZ(DyrosMath::rot2Euler(rd_cc_.link_[Right_Foot].rotm)(2));
        swing_foot_state_global_.translation() = rd_cc_.link_[Left_Foot].xpos;
        swing_foot_state_global_.linear() = DyrosMath::rotateWithZ(DyrosMath::rot2Euler(rd_cc_.link_[Left_Foot].rotm)(2));
    }

    pelvis_state_stance_.translation() = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(stance_foot_state_global_), pelvis_state_global_.translation());
    pelvis_state_stance_.linear() = DyrosMath::inverseIsometry3d(stance_foot_state_global_).linear() * pelvis_state_global_.linear();
    com_pos_state_stance_ = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(stance_foot_state_global_), com_pos_state_global_);
    com_vel_state_stance_ = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(stance_foot_state_global_), com_vel_state_global_);
    swing_state_stance_ = DyrosMath::inverseIsometry3d(stance_foot_state_global_) * swing_foot_state_global_;

    // // Compute Global Foot States, estimates
    // lfoot_support_current_ = DyrosMath::inverseIsometry3d(stance_foot_state_global_) * lfoot_global_current_;
    // rfoot_support_current_ = DyrosMath::inverseIsometry3d(stance_foot_state_global_) * rfoot_global_current_;
    // Eigen::Vector3d &stance = (phase_indicator_(0)) ? rfoot_global_state_ : lfoot_global_state_;
    // Eigen::Vector3d &swing = (phase_indicator_(0)) ? lfoot_global_state_ : rfoot_global_state_;
    // Eigen::Isometry3d &swing_stance = (phase_indicator_(0)) ? lfoot_support_current_ : rfoot_support_current_;
    
    // double swing_yaw_stance = DyrosMath::rot2Euler(swing_stance.linear())(2);
    // swing(0) = stance(0) + cos(stance(2))*swing_stance.translation()(0) - sin(stance(2))*swing_stance.translation()(1);
    // swing(1) = stance(1) + sin(stance(2))*swing_stance.translation()(0) + cos(stance(2))*swing_stance.translation()(1);
    // // stance(2) = DyrosMath::rot2Euler(stance_foot_state_global_.linear())(2);
    // swing(2) = stance(2) + swing_yaw_stance;
}

void CustomController::generateVRP()
{
    vrp_state_.segment(0, 2) = swing_state_stance_.translation().segment(0, 2) / 2; // x, y: middle point of two feet in stance foot frame
    vrp_state_(2) = com_pos_state_stance_(2); // z: com z position in stance foot frame
    vrp_state_(3) = DyrosMath::rot2Euler(swing_state_stance_.linear())[2] / 2;

    unsigned int vrp_horizon_ = vrp_horizon_s_ * hz_;

    vrp_ref_.setZero(vrp_horizon_, 3);
    com_yaw_ref_.setZero(vrp_horizon_);
    com_yaw_vel_ref_.setZero(vrp_horizon_);

    Eigen::MatrixXd vrp_temp_;
    Eigen::VectorXd com_yaw_temp_;
    Eigen::VectorXd com_yaw_vel_temp_;
    unsigned int index = 0;

    // calculate vrp points based on foot commands
    // z uses vrp_height + com_z deltas (train vrp_generator.generate_vrp_online)
    target_stance_foot_state_first_stance_.setZero(number_of_foot_step, 4); // x, y, z, yaw
    target_stance_foot_state_first_stance_(0, 2) = vrp_height_ + com_z_command_(0);

    target_swing_foot_state_first_stance_.setZero(number_of_foot_step, 4); // x, y, z, yaw
    target_swing_foot_state_first_stance_(0, 0) = target_stance_foot_state_first_stance_(0, 0) + foot_commands_(0, 0);
    target_swing_foot_state_first_stance_(0, 1) = target_stance_foot_state_first_stance_(0, 1) + foot_commands_(0, 1);
    // h_0 + step_z + (com_z[1] - com_z[0])
    target_swing_foot_state_first_stance_(0, 2) = target_stance_foot_state_first_stance_(0, 2) + foot_commands_(0, 2)
                                                 + (com_z_command_(1) - com_z_command_(0));
    target_swing_foot_state_first_stance_(0, 3) = target_stance_foot_state_first_stance_(0, 3) + foot_commands_(0, 5); // yaw

    const double vrpx_offset = 0.03;
    const double vrpy_offset = 0.02;

    for (unsigned int step = 1; step < number_of_foot_step; step++) {
        target_stance_foot_state_first_stance_.row(step) = target_swing_foot_state_first_stance_.row(step-1);
        target_swing_foot_state_first_stance_(step, 0) =  target_stance_foot_state_first_stance_(step, 0)
                                                        + foot_commands_(step, 0) * cos(target_stance_foot_state_first_stance_(step, 3))
                                                        - foot_commands_(step, 1) * sin(target_stance_foot_state_first_stance_(step, 3));
        target_swing_foot_state_first_stance_(step, 1) =  target_stance_foot_state_first_stance_(step, 1)
                                                        + foot_commands_(step, 0) * sin(target_stance_foot_state_first_stance_(step, 3))
                                                        + foot_commands_(step, 1) * cos(target_stance_foot_state_first_stance_(step, 3));
        target_swing_foot_state_first_stance_(step, 2) = target_stance_foot_state_first_stance_(step, 2) + foot_commands_(step, 2)
                                                        + (com_z_command_(step + 1) - com_z_command_(step));
        target_swing_foot_state_first_stance_(step, 3) = target_stance_foot_state_first_stance_(step, 3) + foot_commands_(step, 5);
    }
    for (unsigned int step = 0; step < number_of_foot_step; step++) {
        const double stance_yaw = target_stance_foot_state_first_stance_(step, 3);
        const double phase_indicator = copysign(1.0, foot_commands_(step, 1));
        target_stance_foot_state_first_stance_(step, 0) -= phase_indicator * vrpy_offset * sin(stance_yaw);
        target_stance_foot_state_first_stance_(step, 1) += phase_indicator * vrpy_offset * cos(stance_yaw);
        target_stance_foot_state_first_stance_(step, 0) += vrpx_offset * cos(stance_yaw);
        target_stance_foot_state_first_stance_(step, 1) += vrpx_offset * sin(stance_yaw);
        target_swing_foot_state_first_stance_(step, 0) -= -phase_indicator * vrpy_offset * sin(stance_yaw);
        target_swing_foot_state_first_stance_(step, 1) += -phase_indicator * vrpy_offset * cos(stance_yaw);
        target_swing_foot_state_first_stance_(step, 0) += vrpx_offset * cos(stance_yaw);
        target_swing_foot_state_first_stance_(step, 1) += vrpx_offset * sin(stance_yaw);
    }


    for (unsigned int i = 0; i < number_of_foot_step; i++) {
        oneStepVRP(i, vrp_temp_, com_yaw_temp_, com_yaw_vel_temp_);
        vrp_ref_.block(index, 0, t_total_(i), 3) = vrp_temp_;
        com_yaw_ref_.segment(index, t_total_(i)) = com_yaw_temp_;
        com_yaw_vel_ref_.segment(index, t_total_(i)) = com_yaw_vel_temp_;
        index += t_total_(i);
    }
    if (t_total_.sum() < vrp_horizon_){
        for (int i = t_total_.sum(); i < vrp_horizon_; i++){
            vrp_ref_.row(i) = vrp_ref_.row(i-1);
            com_yaw_ref_(i) = com_yaw_ref_(i-1);
            com_yaw_vel_ref_(i) = com_yaw_vel_ref_(i-1);
        }
    }

    // update pelvis state for preview control
    Matrix3d update_preview_state = Matrix3d::Zero();
    update_preview_state.row(0) = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(stance_foot_state_global_), target_com_state_global_.segment(0, 3));
    update_preview_state.row(1) = DyrosMath::inverseIsometry3d(stance_foot_state_global_).linear() * target_com_state_global_.segment(3, 3);
    update_preview_state.row(2) = DyrosMath::inverseIsometry3d(stance_foot_state_global_).linear() * target_com_state_global_.segment(6, 3);
    preview_ctrl_->update_state(update_preview_state);

    // update for foot trajectory
    swing_foot_start_pos_stance_ = swing_state_stance_.translation();
    swing_foot_start_rot_stance_ = DyrosMath::rot2Euler(swing_state_stance_.linear());
    swing_foot_end_pos_stance_ = foot_commands_.block(0, 0, 1, 3).transpose();
    swing_foot_end_rot_stance_ = foot_commands_.block(0, 3, 1, 3).transpose();
}

void CustomController::oneStepVRP(int step, Eigen::MatrixXd &vrp_temp_, Eigen::VectorXd &com_yaw_temp_, Eigen::VectorXd &com_yaw_vel_temp_)
{
    vrp_temp_.setZero(t_total_(step), 3);
    com_yaw_temp_.setZero(t_total_(step));
    com_yaw_vel_temp_.setZero(t_total_(step));

    double dsp = floor(foot_commands_(step, 7) * hz_);
    double ssp = floor(foot_commands_(step, 6) * hz_);
    double t_total = t_total_(step);

    for (int i = 0; i < t_total; i++) {
        if (i < dsp) { // first dsp time
            if (step == 0) {
                vrp_temp_(i, 0) = DyrosMath::cubic(i, 0.0, dsp, vrp_state_(0), target_stance_foot_state_first_stance_(step, 0), 0.0, 0.0);
                vrp_temp_(i, 1) = DyrosMath::cubic(i, 0.0, dsp, vrp_state_(1), target_stance_foot_state_first_stance_(step, 1), 0.0, 0.0);
                vrp_temp_(i, 2) = DyrosMath::cubic(i, 0.0, dsp, vrp_state_(2), target_stance_foot_state_first_stance_(step, 2), 0.0, 0.0);
            }
            else {
                vrp_temp_(i, 0) = DyrosMath::cubic(i, 0.0, dsp, (target_stance_foot_state_first_stance_(step-1, 0) + target_swing_foot_state_first_stance_(step-1, 0)) / 2, target_stance_foot_state_first_stance_(step, 0), 0.0, 0.0);
                vrp_temp_(i, 1) = DyrosMath::cubic(i, 0.0, dsp, (target_stance_foot_state_first_stance_(step-1, 1) + target_swing_foot_state_first_stance_(step-1, 1)) / 2, target_stance_foot_state_first_stance_(step, 1), 0.0, 0.0);
                vrp_temp_(i, 2) = DyrosMath::cubic(i, 0.0, dsp, (target_stance_foot_state_first_stance_(step-1, 2) + target_swing_foot_state_first_stance_(step-1, 2)) / 2, target_stance_foot_state_first_stance_(step, 2), 0.0, 0.0);
            }
            com_yaw_temp_(i) = vrp_state_(3);
            com_yaw_vel_temp_(i) = 0;
        }
        else if (i >= dsp && i < dsp + ssp) { // ssp time
            vrp_temp_.row(i) = target_stance_foot_state_first_stance_.row(step).segment(0, 3);
            if (step == 0) {
                com_yaw_temp_(i) = DyrosMath::cubic(i, dsp, dsp + ssp, vrp_state_(3), (target_stance_foot_state_first_stance_(step, 3) + target_swing_foot_state_first_stance_(step, 3)) / 2, 0.0, 0.0);
                com_yaw_vel_temp_(i) = DyrosMath::cubicDot(i, dsp, dsp + ssp, vrp_state_(3), (target_stance_foot_state_first_stance_(step, 3) + target_swing_foot_state_first_stance_(step, 3)) / 2, 0.0, 0.0);
            }
            else {
                com_yaw_temp_(i) = DyrosMath::cubic(i, dsp, dsp + ssp, (target_stance_foot_state_first_stance_(step-1, 3) + target_swing_foot_state_first_stance_(step-1, 3)) / 2, (target_stance_foot_state_first_stance_(step, 3) + target_swing_foot_state_first_stance_(step, 3)) / 2, 0.0, 0.0);
                com_yaw_vel_temp_(i) = DyrosMath::cubicDot(i, dsp, dsp + ssp, (target_stance_foot_state_first_stance_(step-1, 3) + target_swing_foot_state_first_stance_(step-1, 3)) / 2, (target_stance_foot_state_first_stance_(step, 3) + target_swing_foot_state_first_stance_(step, 3)) / 2, 0.0, 0.0);
            }
        }
        else {
            vrp_temp_(i, 0) = DyrosMath::cubic(i, dsp + ssp, t_total, target_stance_foot_state_first_stance_(step, 0), (target_stance_foot_state_first_stance_(step, 0) + target_swing_foot_state_first_stance_(step, 0)) / 2, 0.0, 0.0);
            vrp_temp_(i, 1) = DyrosMath::cubic(i, dsp + ssp, t_total, target_stance_foot_state_first_stance_(step, 1), (target_stance_foot_state_first_stance_(step, 1) + target_swing_foot_state_first_stance_(step, 1)) / 2, 0.0, 0.0);
            vrp_temp_(i, 2) = DyrosMath::cubic(i, dsp + ssp, t_total, target_stance_foot_state_first_stance_(step, 2), (target_stance_foot_state_first_stance_(step, 2) + target_swing_foot_state_first_stance_(step, 2)) / 2, 0.0, 0.0);
            com_yaw_temp_(i) = (target_stance_foot_state_first_stance_(step, 3) + target_swing_foot_state_first_stance_(step, 3)) / 2;
            com_yaw_vel_temp_(i) = 0;
        }
    }
}

void CustomController::generateCoM()
{
    MatrixXd preview_output = preview_ctrl_->compute_target_state(vrp_ref_.block(walking_tick, 0, preview_horizon_, 3));
    target_com_state_stance_.segment(0, 3) = preview_output.row(0);
    target_com_state_stance_.segment(3, 3) = preview_output.row(1);
    target_com_state_stance_.segment(6, 3) = preview_output.row(2);

    target_com_state_global_.segment(0, 3) = DyrosMath::multiplyIsometry3dVector3d(stance_foot_state_global_, target_com_state_stance_.segment(0, 3));
    target_com_state_global_.segment(3, 3) = stance_foot_state_global_.linear() * target_com_state_stance_.segment(3, 3);
    target_com_state_global_.segment(6, 3) = stance_foot_state_global_.linear() * target_com_state_stance_.segment(6, 3);

    // target_pelvis_stance_.translation() = pelvis_state_stance_.translation() + 0.7 * (target_com_state_stance_.segment(0, 3) - com_pos_state_stance_);
    target_pelvis_stance_.translation() = target_com_state_stance_.segment(0, 3);
    target_pelvis_stance_.translation()(2) += 0.04;
    target_pelvis_stance_.linear() = DyrosMath::rotateWithZ(com_yaw_ref_(walking_tick));
}

void CustomController::generateFeet()
{
    double t_dsp = floor(foot_commands_(0, 7) * hz_);
    double t_ssp = floor(foot_commands_(0, 6) * hz_);
    if (walking_tick < t_dsp) {
        // first dsp time: swing foot stays in the same position
        target_swing_foot_stance_.translation() = swing_foot_start_pos_stance_;
        target_swing_foot_stance_.linear() = DyrosMath::Euler2rot(swing_foot_start_rot_stance_(0), swing_foot_start_rot_stance_(1), swing_foot_start_rot_stance_(2));
    }
    else if (walking_tick >= t_dsp && walking_tick < t_dsp + t_ssp) {
        target_swing_foot_stance_.translation()(0) = DyrosMath::cubic(walking_tick, t_dsp, t_dsp + t_ssp, swing_foot_start_pos_stance_(0), swing_foot_end_pos_stance_(0), 0.0, 0.0);
        target_swing_foot_stance_.translation()(1) = DyrosMath::cubic(walking_tick, t_dsp, t_dsp + t_ssp, swing_foot_start_pos_stance_(1), swing_foot_end_pos_stance_(1), 0.0, 0.0);
        
        Vector3d target_euler;
        target_euler(0) = DyrosMath::cubic(walking_tick, t_dsp, t_dsp + t_ssp, swing_foot_start_rot_stance_(0), swing_foot_end_rot_stance_(0), 0.0, 0.0);
        target_euler(1) = DyrosMath::cubic(walking_tick, t_dsp, t_dsp + t_ssp, swing_foot_start_rot_stance_(1), swing_foot_end_rot_stance_(1), 0.0, 0.0);
        target_euler(2) = DyrosMath::cubic(walking_tick, t_dsp, t_dsp + t_ssp, swing_foot_start_rot_stance_(2), swing_foot_end_rot_stance_(2), 0.0, 0.0);
        target_swing_foot_stance_.linear() = DyrosMath::Euler2rot(target_euler(0), target_euler(1), target_euler(2));

        // for z, divide it to 3 phases(lift up, maintain, lift down)
        double lift_up_time = t_ssp * 0.1;
        double maintain_time = t_ssp * 0.9;

        double lift_up_height = std::max({
            swing_foot_start_pos_stance_(2),
            swing_foot_end_pos_stance_(2),
            target_stance_foot_stance_.translation()(2)
        }) + foot_commands_(0, 8);

        if (walking_tick < t_dsp + lift_up_time)
            target_swing_foot_stance_.translation()(2) = DyrosMath::cubic(walking_tick, t_dsp, t_dsp + lift_up_time, swing_foot_start_pos_stance_(2), lift_up_height, 0.0, 0.0);
        else if (walking_tick >= t_dsp + lift_up_time && walking_tick < t_dsp + maintain_time)
            target_swing_foot_stance_.translation()(2) = lift_up_height;
        else // if (walking_tick >= t_dsp + maintain_time && walking_tick < t_dsp + t_ssp) 
            target_swing_foot_stance_.translation()(2) = DyrosMath::cubic(walking_tick, t_dsp + maintain_time, t_dsp + t_ssp, lift_up_height, swing_foot_end_pos_stance_(2), 0.0, 0.0);
    }
    else if (walking_tick >= t_dsp + t_ssp) {
        target_swing_foot_stance_.translation() = swing_foot_end_pos_stance_;
        target_swing_foot_stance_.linear() = DyrosMath::Euler2rot(swing_foot_end_rot_stance_(0), swing_foot_end_rot_stance_(1), swing_foot_end_rot_stance_(2));
    }
    target_stance_foot_stance_.translation().setZero();
    target_stance_foot_stance_.linear().setIdentity();
}

void CustomController::getTargetJointPos()
{
    target_lfoot_stance_ = (phase_indicator_(0) == 0) ? target_stance_foot_stance_ : target_swing_foot_stance_;
    target_rfoot_stance_ = (phase_indicator_(0) == 0) ? target_swing_foot_stance_ : target_stance_foot_stance_;

    target_pelvis_float_.translation().setZero();
    target_pelvis_float_.linear().setIdentity();

    target_lfoot_float_ = DyrosMath::inverseIsometry3d(target_pelvis_stance_) * target_lfoot_stance_;
    target_rfoot_float_ = DyrosMath::inverseIsometry3d(target_pelvis_stance_) * target_rfoot_stance_;
    computeIkControl(target_pelvis_float_, target_lfoot_float_, target_rfoot_float_, q_leg_desired_);
}

void CustomController::computeIkControl(const Eigen::Isometry3d &float_trunk_transform, const Eigen::Isometry3d &float_lleg_transform, const Eigen::Isometry3d &float_rleg_transform, Eigen::Vector12d &q_des)
{
    Eigen::Vector3d R_r, R_D, L_r, L_D;

    L_D << 0.11, +0.1025, -0.1025;
    R_D << 0.11, -0.1025, -0.1025;

    L_r = float_lleg_transform.rotation().transpose() * (float_trunk_transform.translation() + float_trunk_transform.rotation() * L_D - float_lleg_transform.translation());
    R_r = float_rleg_transform.rotation().transpose() * (float_trunk_transform.translation() + float_trunk_transform.rotation() * R_D - float_rleg_transform.translation());

    double R_C = 0, L_C = 0, L_upper = 0.351, L_lower = 0.351, R_alpha = 0, L_alpha = 0;

    L_C = sqrt(pow(L_r(0), 2) + pow(L_r(1), 2) + pow(L_r(2), 2));
    R_C = sqrt(pow(R_r(0), 2) + pow(R_r(1), 2) + pow(R_r(2), 2));
     
    double knee_acos_var_L = 0;
    double knee_acos_var_R = 0;

    knee_acos_var_L = (pow(L_upper, 2) + pow(L_lower, 2) - pow(L_C, 2))/ (2 * L_upper * L_lower);
    knee_acos_var_R = (pow(L_upper, 2) + pow(L_lower, 2) - pow(R_C, 2))/ (2 * L_upper * L_lower);

    knee_acos_var_L = DyrosMath::minmax_cut(knee_acos_var_L, -0.99, + 0.99);
    knee_acos_var_R = DyrosMath::minmax_cut(knee_acos_var_R, -0.99, + 0.99);

    q_des(3) = (-acos(knee_acos_var_L) + M_PI);  
    q_des(9) = (-acos(knee_acos_var_R) + M_PI);
    q_des(5) = atan2(L_r(1), L_r(2));                                                                                  // Ankle roll
    q_des(11) = atan2(R_r(1), R_r(2));

    L_alpha = asin(DyrosMath::minmax_cut(L_upper / L_C * sin(M_PI - q_des(3)), -0.99, 0.99));
    R_alpha = asin(DyrosMath::minmax_cut(L_upper / R_C * sin(M_PI - q_des(9)), -0.99, 0.99));
    // L_alpha =asin( L_upper / L_C * sin(M_PI - q_des(3)));
    // R_alpha = asin(L_upper / R_C * sin(M_PI - q_des(9)));
    
    q_des(4) = -atan2(L_r(0), sqrt(pow(L_r(1), 2) + pow(L_r(2), 2))) - L_alpha;
    q_des(10) = -atan2(R_r(0), sqrt(pow(R_r(1), 2) + pow(R_r(2), 2))) - R_alpha;

    Eigen::Matrix3d R_Knee_Ankle_Y_rot_mat, L_Knee_Ankle_Y_rot_mat;
    Eigen::Matrix3d R_Ankle_X_rot_mat, L_Ankle_X_rot_mat;
    Eigen::Matrix3d R_Hip_rot_mat, L_Hip_rot_mat;

    L_Knee_Ankle_Y_rot_mat = DyrosMath::rotateWithY(-q_des(3) - q_des(4));
    L_Ankle_X_rot_mat = DyrosMath::rotateWithX(-q_des(5));
    R_Knee_Ankle_Y_rot_mat = DyrosMath::rotateWithY(-q_des(9) - q_des(10));
    R_Ankle_X_rot_mat = DyrosMath::rotateWithX(-q_des(11));

    L_Hip_rot_mat.setZero();
    R_Hip_rot_mat.setZero();

    L_Hip_rot_mat = float_trunk_transform.rotation().transpose() * float_lleg_transform.rotation() * L_Ankle_X_rot_mat * L_Knee_Ankle_Y_rot_mat;
    R_Hip_rot_mat = float_trunk_transform.rotation().transpose() * float_rleg_transform.rotation() * R_Ankle_X_rot_mat * R_Knee_Ankle_Y_rot_mat;

    q_des(0) = atan2(-L_Hip_rot_mat(0, 1), L_Hip_rot_mat(1, 1));                                                       // Hip yaw
    q_des(1) = atan2(L_Hip_rot_mat(2, 1), -L_Hip_rot_mat(0, 1) * sin(q_des(0)) + L_Hip_rot_mat(1, 1) * cos(q_des(0))); // Hip roll
    q_des(2) = atan2(-L_Hip_rot_mat(2, 0), L_Hip_rot_mat(2, 2));                                                       // Hip pitch
    q_des(3) = q_des(3);                                                                                               // Knee pitch
    q_des(4) = q_des(4);                                                                                               // Ankle pitch

    q_des(6) = atan2(-R_Hip_rot_mat(0, 1), R_Hip_rot_mat(1, 1));
    q_des(7) = atan2(R_Hip_rot_mat(2, 1), -R_Hip_rot_mat(0, 1) * sin(q_des(6)) + R_Hip_rot_mat(1, 1) * cos(q_des(6)));
    q_des(8) = atan2(-R_Hip_rot_mat(2, 0), R_Hip_rot_mat(2, 2));
    q_des(9) = q_des(9);
    q_des(10) = q_des(10);
}

void CustomController::copyRobotData(RobotData &rd_l)
{
    std::memcpy(&rd_cc_, &rd_l, sizeof(RobotData));
}

void CustomController::initBias()
{
    q_bias_.setZero();
    if (~is_on_robot_){
        std::random_device rd;  
        std::mt19937 gen(rd());
        float bias_std = 0.;
        std::uniform_real_distribution<> dis(-bias_std, bias_std);
        q_bias_(2) = dis(gen);
        q_bias_(3) = dis(gen);
        q_bias_(4) = dis(gen);
        q_bias_(8) = dis(gen);
        q_bias_(9) = dis(gen);
        q_bias_(10) = dis(gen);
    }
}

void CustomController::processBias()
{
    for (int i = 0; i < MODEL_DOF; i++){
        q_noise_(i) += q_bias_(i);
    }
}
