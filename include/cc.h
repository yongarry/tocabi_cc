#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

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
    
    void initVariable();
    void processNoise();
    void writeDesiredRPY();
    
    RobotData &rd_;
    RobotData rd_cc_;

    bool is_on_robot_ = false;
    bool is_write_file_ = true;
    std::ofstream writeFile;

    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_desired_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    float start_time_;
    float time_inference_pre_ = 0.0;

    double time_cur_;
    double time_pre_;

    Eigen::Vector3d ankle_desired_rpy_;
    double arm_ankle_roll_ = 0.0;
    double arm_ankle_pitch_ = -1.0;
    double arm_ankle_yaw_ = 0.0;
    double q_left_arm_desired_ = -1.0;

    ros::NodeHandle nh_;
    ros::Subscriber arm_ankle_sub_;
    void AnkleCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);

private:
    Eigen::VectorQd ControlVal_;
};