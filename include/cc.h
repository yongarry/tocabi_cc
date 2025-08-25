#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>

#include "onnxruntime_cxx_api.h"

class CustomController
{
public:
    CustomController(RobotData &rd);
    Eigen::VectorQd getControl();

    //void taskCommandToCC(TaskCommand tc_);
    
    void computeSlow();
    void computeFast();
    void computePlanner();
    void copyRobotData(RobotData &rd_l);

    RobotData &rd_;
    RobotData rd_cc_;

    void loadOnnX();
    void processNoise();
    void processObservation();
    void processDiscriminator();
    void feedforwardPolicy();
    void initVariable();
    
    void quatToTanNorm(const Eigen::Quaterniond& quaternion, Eigen::Vector3d& tangent, Eigen::Vector3d& normal);
    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);
    Eigen::Vector3d quatRotateInverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v);


    /////////////////////////////////// ONNX Runtime by Yongarry ///////////////////////////////////////
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;

    std::vector<std::vector<float>> input_states_buffer;
    std::vector<float> state_cur_, state_buffer_;

    // for long history observation
    std::vector<float> state_long_hist_, state_long_hist_buffer_;

    int input_obs_idx_ = 0;
    int debug = 0;

    ///////////////////////////////////// Actor-Critic Network ///////////////////////////////////////
    static const int num_action = 12;
    static const int num_actuator_action = 12;

    static const int num_cur_state = 50; 
    static const int num_hist_step = 5;
    static const int num_state = num_cur_state * num_hist_step;
    Eigen::MatrixXd rl_action_, rl_action_pre_, torq_diff_, energy;
    double value_;

    Vector12d action_offset, action_scale;
    Vector12d target_pos;
    bool pd_control_ = true; // use PD control or not
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;

    std::ofstream writeFile;

    bool is_on_robot_ = false;
    bool is_write_file_ = true;
    bool is_hist_encoder_ = false;


    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_, q_vel_noise_pre_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    Eigen::VectorQd Gravity_MJ_;

    Eigen::Vector6d LF_CF_FT_pre, RF_CF_FT_pre = Eigen::Vector6d::Zero();

    float start_time_;
    float time_inference_pre_ = 0.0;
    float time_write_pre_ = 0.0;

    double time_cur_;
    double time_pre_;

    Eigen::Vector3d euler_angle_;
    Eigen::Vector3d tan_vec, nor_vec;

    // float ft_left_init_ = 500.0;
    // float ft_right_init_ = 500.0;

    string weight_dir_ = "";
    // Joystick
    ros::NodeHandle nh_;

    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    void xBoxJoyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;
    ros::Subscriber xbox_joy_sub_;

    Eigen::Vector3d local_lin_vel_;

    double target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;
    double target_vel_yaw_ = 0.0;
    double step_time_ = 1.2;

private:
    Eigen::VectorQd ControlVal_;

    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    const std::string reset = "\033[0m";     // Reset color
    const std::string red = "\033[31m";     // Red
    const std::string green = "\033[32m";   // Green
    const std::string yellow = "\033[33m";  // Yellow
    const std::string blue = "\033[34m"; 
};