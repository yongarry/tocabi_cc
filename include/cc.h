#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <tocabi_msgs/WalkingCommand.h>

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

    //////////////////////////////////////////// Donghyeon RL /////////////////////////////////////////
    void loadNetwork();
    void processNoise();
    void processObservation();
    void processDiscriminator();
    void feedforwardPolicy();
    void initVariable();
    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);
    void quatToTanNorm(const Eigen::Quaterniond& quaternion, Eigen::Vector3d& tangent, Eigen::Vector3d& normal);
    void checkTouchDown();
    Eigen::Vector3d quatRotateInverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v);

    ///////////////////////////////////// Actor-Critic Network ///////////////////////////////////////
    static const int num_action = 12;
    static const int num_actuator_action = 12;
    // static const int num_cur_state = 49; // 37 + 12
    static const int num_cur_state = 48; // 36 + 12
    // static const int num_cur_internal_state = 37;
    static const int num_cur_internal_state = 36;
    static const int num_state_skip = 2;
    static const int num_state_hist = 10;
    static const int num_state = num_cur_internal_state*num_state_hist+num_action*(num_state_hist-1);
    // static const int num_state = 59;
    static const int num_hidden1 = 512;
    static const int num_hidden2 = 512;

    Eigen::MatrixXd policy_net_w0_;
    Eigen::MatrixXd policy_net_b0_;
    Eigen::MatrixXd policy_net_w2_;
    Eigen::MatrixXd policy_net_b2_;
    Eigen::MatrixXd action_net_w_;
    Eigen::MatrixXd action_net_b_;

    Eigen::MatrixXd hidden_layer1_;
    Eigen::MatrixXd hidden_layer2_;
    Eigen::MatrixXd rl_action_;

    Eigen::MatrixXd value_net_w0_;
    Eigen::MatrixXd value_net_b0_;
    Eigen::MatrixXd value_net_w2_;
    Eigen::MatrixXd value_net_b2_;
    Eigen::MatrixXd value_net_w_;
    Eigen::MatrixXd value_net_b_;

    Eigen::MatrixXd value_hidden_layer1_;
    Eigen::MatrixXd value_hidden_layer2_;
    double value_;

    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;
    
    Eigen::MatrixXd state_;
    Eigen::MatrixXd state_cur_;
    Eigen::MatrixXd state_buffer_;
    Eigen::MatrixXd state_mean_;
    Eigen::MatrixXd state_var_;
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////// Discriminator Network ///////////////////////////////////////
    static const int num_disc_state = 40 * 2; //34 * 2;
    static const int num_disc_cur_state = 40; //34;
    static const int disc_output = 1;
    static const int num_disc_hidden1 = 256;
    static const int num_disc_hidden2 = 256;

    Eigen::MatrixXd disc_net_w0_;
    Eigen::MatrixXd disc_net_b0_;
    Eigen::MatrixXd disc_net_w2_;
    Eigen::MatrixXd disc_net_b2_;
    Eigen::MatrixXd disc_net_w_;
    Eigen::MatrixXd disc_net_b_;

    Eigen::MatrixXd disc_hidden_layer1_;
    Eigen::MatrixXd disc_hidden_layer2_;
    Eigen::MatrixXd disc_value_;

    Eigen::MatrixXd disc_state_;
    Eigen::MatrixXd disc_state_buffer_;
    Eigen::MatrixXd disc_state_cur_;
    Eigen::MatrixXd disc_state_mean_;
    Eigen::MatrixXd disc_state_var_;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    std::ofstream writeFile;

    bool is_on_robot_ = false;
    bool is_write_file_ = true;

    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;

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
    double action_dt_accumulate_ = 0.0;

    Eigen::Vector3d euler_angle_;
    Eigen::Vector3d tan_vec, nor_vec;

    // float ft_left_init_ = 500.0;
    // float ft_right_init_ = 500.0;

    string weight_dir_ = "";
    // Joystick
    ros::NodeHandle nh_;

    void joyCallback(const tocabi_msgs::WalkingCommand::ConstPtr& joy);
    void xBoxJoyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    // void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;
    ros::Subscriber xbox_joy_sub_;

    Eigen::Vector3d local_lin_vel_;

    double target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;
    double target_vel_yaw_ = 0.0;

    float desired_vel_x = 0.0;
    float desired_vel_yaw = 0.0;

private:
    Eigen::VectorQd ControlVal_;
};