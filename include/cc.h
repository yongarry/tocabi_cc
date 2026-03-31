#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/PoseArray.h>

#include "onnxruntime_cxx_api.h"
#include "preview_controller.h"

using namespace Eigen;

class CustomController
{
public:
    CustomController(RobotData &rd);
    Eigen::VectorQd getControl();

    //void taskCommandToCC(TaskCommand tc_);
    string workspace_dir_ = "/home/yong/ubuntu-20-04/catkin_ws/src/tocabi_cc/";
    string weight_file_ = "";
    string cmd_file_ = "";

    const double hz_ = 125.;
    const double pd_hz_ = 2000;
    double del_t = 1 / hz_;
    double preview_horizon_ = 2.0 * hz_;

    void computeSlow();
    void computeFast();
    void copyRobotData(RobotData &rd_l);

    RobotData &rd_;
    RobotData rd_cc_;

    bool is_on_robot_ = false;
    /////////////////////////////////// ONNX Runtime by Yongarry ///////////////////////////////////////
    void loadNetwork();
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;

    std::vector<std::vector<float>> input_states_buffer;
    std::vector<float> state_, state_cur_, state_buffer_;

    int input_obs_idx_ = 0;
    void initVariable();
    void loadCommands();

    void processNoise();
    void processObservation();
    void feedforwardPolicy();

    static const int num_actuator_action = 12;
    int num_cur_state = 68;
    static const int num_state_skip = 2;
    static const int num_state_hist = 10;
    int num_state = num_cur_state * num_state_hist;

    MatrixXd rl_action_;
    Vector12d q_lower_limit_, q_upper_limit_, q_target_;

    double value_;
    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;
    
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;
    Vector12d q_leg_desired_;

    // void processBias();
    // void initBias();
    // Eigen::Matrix<double, MODEL_DOF, 1> q_bias_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    float start_time_;
    float time_inference_pre_ = 0.0;
    float time_write_pre_ = 0.0;

    double time_cur_;
    double time_pre_;
    double action_dt_accumulate_ = 0.0;

    // Joystick
    ros::NodeHandle nh_;
    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;
    
    // BIPED WALKING PARAMETER
    const int number_of_foot_step = 2;
    MatrixXd foot_commands_;
    VectorXd phase_indicator_; // 0 means left foot stance(right swing), 1 means right foot stance(left swing)
    VectorXd t_total_;
    bool is_right_stance_first = false; 
    const double vrp_height_ = 0.728;
    // const double vrp_height_ = 0.68;

    int number_of_planner_step = 0;
    int planner_index_ = 0;
    MatrixXd foot_commands_planner_;


    // VRP + Preview Control
    PreviewController preview_ctrl_{del_t, preview_horizon_};

    void updateCommand();
    void updateRobotStates();

    void generateVRP();
    void oneStepVRP(int step, Eigen::MatrixXd &vrp_temp_, Eigen::VectorXd &com_yaw_temp_, Eigen::VectorXd &com_yaw_vel_temp_);
    void generateCoM();
    void generateFeet();
    
    void getTargetJointPos();
    void computeIkControl(const Eigen::Isometry3d &float_trunk_transform, const Eigen::Isometry3d &float_lleg_transform, const Eigen::Isometry3d &float_rleg_transform, Eigen::Vector12d &q_des);

    // Robot States
    Vector3d com_pos_state_global_;
    Vector3d com_vel_state_global_;
    Isometry3d pelvis_state_global_;
    Isometry3d stance_foot_state_global_;
    Isometry3d swing_foot_state_global_;

    Isometry3d pelvis_state_stance_;
    Vector3d com_pos_state_stance_;
    Vector3d com_vel_state_stance_;
    Isometry3d swing_state_stance_;

    Vector4d vrp_state_;
    double vrp_horizon_s_ = 4.0;
    MatrixXd vrp_ref_;
    VectorXd com_yaw_ref_, com_yaw_vel_ref_;

    MatrixXd target_stance_foot_state_first_stance_;
    MatrixXd target_swing_foot_state_first_stance_;

    Vector3d swing_foot_start_pos_stance_, swing_foot_start_rot_stance_;
    Vector3d swing_foot_end_pos_stance_, swing_foot_end_rot_stance_;

    Isometry3d target_swing_foot_stance_, target_stance_foot_stance_;
    Isometry3d target_lfoot_stance_, target_rfoot_stance_;
    Isometry3d target_pelvis_stance_, target_pelvis_global_;

    Isometry3d target_lfoot_float_, target_rfoot_float_;
    Isometry3d target_pelvis_float_;

    VectorXd target_com_state_stance_, target_com_state_global_;

    // Utility functions
    static double wrap_to_pi(double angles){
        angles = fmod(angles, 2*M_PI);
        if (angles > M_PI){
          angles -= 2*M_PI;    
        }
        return angles;
    }


private:
    VectorQd ControlVal_;

    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    unsigned int walking_tick = 0;
};