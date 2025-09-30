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

class CustomController
{
public:
    CustomController(RobotData &rd);
    Eigen::VectorQd getControl();

    //void taskCommandToCC(TaskCommand tc_);
    string workspace_dir_ = "/home/yong20/ros_ws/ros1/tocabi_ws/src/tocabi_cc/";
    string weight_dir_ = "";

    const double hz_ =125.;
    const double pd_hz_ = 2000;
    double del_t = 1 / hz_;

    void computeSlow();
    void computeFast();
    void copyRobotData(RobotData &rd_l);

    RobotData &rd_;
    RobotData rd_cc_;


    /////////////////////////////////// ONNX Runtime by Yongarry ///////////////////////////////////////
    void loadNetwork();
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;

    std::vector<std::vector<float>> input_states_buffer;
    std::vector<float> state_, state_cur_, state_buffer_;

    int input_obs_idx_ = 0;

    void processNoise();
    void processBias();
    void initBias();
    Eigen::Matrix<double, MODEL_DOF, 1> q_bias_;
    void processObservation();
    void feedforwardPolicy();

    void initVariable();

    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);

    static const int num_action = 12;
    static const int num_actuator_action = 12;
    int num_cur_state = 65;
    static const int num_cur_state_intern = 51;
    float swing_ratio = 0.428;
    static const int num_state_skip = 2;
    static const int num_state_hist = 10;
    int num_state = num_cur_state * num_state_hist;

    Eigen::MatrixXd rl_action_;

    double value_;

    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;
    
    // Eigen::MatrixXd state_;
    // Eigen::MatrixXd state_cur_;
    // Eigen::MatrixXd state_buffer_;

    std::ofstream writeFile;
    std::ofstream evalFile;

    float phase_ = 0.0;

    bool is_on_robot_ = true;
    bool is_write_file_ = true;
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;
    Eigen::Vector12d q_leg_desired_;

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

    Vector3_t base_lin_vel, base_ang_vel;

    // Joystick
    ros::NodeHandle nh_;
    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;
    void loadCommand(const std::string &command_file);
    void loadCommand_QR(const std::string &command_file);
    ros::Subscriber aruco_sub_, marker_ids_sub_;
    void ArUcoPoseCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);
    // void ArucoIDCallback(const std_msgs::Int32MultiArray::ConstPtr& msg);
    std::vector<int> marker_ids_;
    bool load_qr_only_first = true;
    std::vector<Eigen::Vector3d> aruco_pos_;
    // std::vector<Eigen::Quaterniond> aruco_quat_;
    std::vector<Eigen::Isometry3d> stepcmd_pelv_, stepcmd_global_;



    // BIPED WALKING PARAMETER
    void walkingParameterSetting();
    const int number_of_foot_step = 2;
    Eigen::Vector3d phase_indicator_;
    Eigen::Vector3d step_length_x_;
    Eigen::Vector3d step_length_y_;
    Eigen::Vector3d step_yaw_;
    Eigen::Vector3d t_dsp_;
    Eigen::Vector3d t_ssp_;
    Eigen::Vector3d t_dsp_seconds;
    Eigen::Vector3d t_ssp_seconds;
    Eigen::Vector3d foot_height_;
    Eigen::Vector3d t_total_;
    int first_stance_foot_ = 1; // 1 means right foot stance, 0 means left foot stance
    const double com_height_ = 0.68;

    double t_last_;
    double t_start_;
    double t_temp_;  
    double t_double1_;
    double t_double2_;
    double zmp_offset = 0.02;
    double zmp_offset_x = 0.03;



    // Policy input
    Eigen::VectorXd target_com_state_stance_frame_;
    Eigen::VectorXd target_swing_state_stance_frame_;


    Eigen::MatrixXd foot_step_;
    Eigen::MatrixXd foot_step_offset_;
    Eigen::MatrixXd foot_step_support_frame_;
    Eigen::MatrixXd foot_step_support_frame_offset_;

    Eigen::Isometry3d pelv_support_start_;
    // Eigen::Isometry3d pelv_support_init_;
    Eigen::Isometry3d pelv_support_init_yaw_;

    Eigen::Vector3d com_desired_;
    Eigen::Vector3d com_desired_dot_;

    Eigen::Vector3d com_support_init_yaw_;
    Eigen::Vector3d com_support_init_dot_yaw_;

    Eigen::Vector3d com_support_current_;
    Eigen::Vector3d com_support_current_dot_;
    Eigen::Vector3d com_support_current_dot_prev_;
    Eigen::Vector3d com_global_current_;
    Eigen::Vector3d com_global_current_dot_;
    Eigen::Vector3d com_global_current_dot_prev_;

    Eigen::Vector3d pelv_rpy_current_;
    Eigen::Vector3d rfoot_rpy_current_;
    Eigen::Vector3d lfoot_rpy_current_;
    Eigen::Isometry3d pelv_yaw_rot_current_from_global_;
    Eigen::Isometry3d rfoot_roll_rot_;
    Eigen::Isometry3d lfoot_roll_rot_;
    Eigen::Isometry3d rfoot_pitch_rot_;
    Eigen::Isometry3d lfoot_pitch_rot_;
    Eigen::Isometry3d rfoot_yaw_rot_;
    Eigen::Isometry3d lfoot_yaw_rot_;

    Eigen::Isometry3d pelv_global_current_;
    Eigen::Isometry3d lfoot_global_current_;
    Eigen::Isometry3d rfoot_global_current_;
    Eigen::Isometry3d pelv_global_init_;
    Eigen::Isometry3d lfoot_global_init_;
    Eigen::Isometry3d rfoot_global_init_;

    Eigen::Isometry3d pelv_trajectory_support_; //local frame
    Eigen::Isometry3d pelv_trajectory_support_fast_; //local frame
    Eigen::Isometry3d pelv_trajectory_support_slow_; //local frame
    
    Eigen::Isometry3d rfoot_trajectory_support_;  //local frame
    Eigen::Isometry3d lfoot_trajectory_support_;
    Eigen::Isometry3d lfoot_trajectory_support_fast_;
    Eigen::Isometry3d lfoot_trajectory_support_slow_;

    Eigen::Vector3d rfoot_trajectory_euler_support_;
    Eigen::Vector3d lfoot_trajectory_euler_support_;

    Eigen::Isometry3d pelv_trajectory_float_; //pelvis frame

    Eigen::Vector3d com_trajectory_float_;

    Eigen::Isometry3d lfoot_trajectory_float_;
    Eigen::Isometry3d lfoot_trajectory_float_fast_;
    Eigen::Isometry3d lfoot_trajectory_float_slow_;

    Eigen::Isometry3d rfoot_trajectory_float_;
    Eigen::Isometry3d rfoot_trajectory_float_fast_;
    Eigen::Isometry3d rfoot_trajectory_float_slow_;

    Eigen::Vector3d pelv_support_euler_init_yaw_;
    Eigen::Vector3d lfoot_support_euler_init_yaw_;
    Eigen::Vector3d rfoot_support_euler_init_yaw_;
    double wn = sqrt(GRAVITY / com_height_);

    double walking_end_flag = 0;
    
    Eigen::Isometry3d swingfoot_global_current_; 
    Eigen::Isometry3d supportfoot_global_current_; // Only yaw is considered

    Eigen::Isometry3d pelv_support_current_;
    Eigen::Isometry3d lfoot_support_current_;
    Eigen::Isometry3d rfoot_support_current_;

    Eigen::Isometry3d lfoot_support_init_yaw_;
    Eigen::Isometry3d rfoot_support_init_yaw_;
    
    Eigen::Isometry3d target_com_state_float_frame_, target_lfoot_state_float_frame_, target_rfoot_state_float_frame_;

    
    Eigen::MatrixXd ref_zmp_;
    Eigen::MatrixXd ref_com_xy_vel_; // for heuristic com planner
    Eigen::MatrixXd ref_zmp_container;
    Eigen::MatrixXd ref_zmp_thread3;
    Eigen::VectorXd ref_com_yaw_;
    Eigen::VectorXd ref_com_yawvel_;
    // PREVIEW CONTROL
    Eigen::Vector3d x_preview_;
    Eigen::Vector3d y_preview_;

    Eigen::Vector3d xs_preview_;
    Eigen::Vector3d ys_preview_;
    Eigen::Vector3d xd_preview_;
    Eigen::Vector3d yd_preview_; 

    Eigen::MatrixXd Gi_preview_;
    Eigen::MatrixXd Gx_preview_;
    Eigen::VectorXd Gd_preview_;
    Eigen::MatrixXd A_preview_;
    Eigen::VectorXd B_preview_;
    Eigen::MatrixXd C_preview_;
    double UX_preview_, UY_preview_, EX_preview_, EY_preview_;


    Eigen::Vector6d l_ft_;
    Eigen::Vector6d r_ft_;
    Eigen::Vector6d l_ft_LPF;
    Eigen::Vector6d r_ft_LPF;
    Eigen::Vector2d zmp_measured_mj_;
    Eigen::Vector2d zmp_measured_LPF_;

    double P_angle_i = 0;
    double P_angle = 0;
    double P_angle_input_dot = 0;
    double P_angle_input = 0;
    double R_angle = 0;
    double R_angle_input_dot = 0;
    double R_angle_input = 0;
    double aa = 0; 
    double Y_angle_input = 0;

    // BOOLEAN OPERATOR
    bool walking_enable_;
    bool is_lfoot_support = false;
    bool is_rfoot_support = false;
    bool is_dsp1 = false;
    bool is_ssp  = false;
    bool is_dsp2 = false;
    bool is_preview_ctrl_init = true;

    // Logging
    Eigen::Isometry3d supportfoot_global_init_; // Used just to compute init_yaw_
    Eigen::Isometry3d supportfoot_global_init_yaw_;
    Eigen::VectorXd swing_state_stance_frame_;
    Eigen::VectorXd com_state_stance_frame_;

    // USER COMMAND
    double Lcommand_step_length_x_ = 0.;
    double Lcommand_step_length_y_ = 0.21;
    double Lcommand_step_yaw_ = 0.;
    double Lcommand_t_dsp_ = 0.1;
    double Lcommand_t_ssp_ = .7;
    double Lcommand_foot_height_ = 0.08;

    double Rcommand_step_length_x_ = 0.;
    double Rcommand_step_length_y_ = 0.21;
    double Rcommand_step_yaw_ = 0.;
    double Rcommand_t_dsp_ = 0.1;
    double Rcommand_t_ssp_ = .7;
    double Rcommand_foot_height_ = 0.08;

    bool ideal_preview = false;

    int ctrl_mode = 2; // 0 for Joystick Mode 1 for Stepping Stone, 2 for Random Command, 3 for Data Collection
    int policy_mode = 0; // 0 for ral, 1 for heuri, 2 for intern
    int current_step_number = 0;
    int planned_step_number = 30; // 30
    Eigen::VectorXd foothold_x_planned; // These are the locations and desired yaw angles of the stepping stones, in global frame coordinates.
    Eigen::VectorXd foothold_y_planned;
    Eigen::VectorXd foothold_yaw_planned;
    Eigen::VectorXd t_dsp_planned;
    Eigen::VectorXd t_ssp_planned;
    Eigen::VectorXd foot_height_planned;
    Eigen::Vector3d lfoot_global_state; // These are (X,Y,yaw) info in global frame coordinates. Used for generating next foothold commands when stepping stone coordinates are given in global frame coordinates.
    Eigen::Vector3d rfoot_global_state;


    int iter_x_l=0;
    int iter_x_r=0;
    int iter_y_l_ls =0;
    int iter_y_l_rs =0;
    int iter_y_r_ls =0;
    int iter_y_r_rs =0;
    int iter_y_r =0;
    int iter_yaw_l_ls=0;
    int iter_yaw_l_rs=0;
    int iter_yaw_r_ls=0;
    int iter_yaw_r_rs=0;
    int iter_end_x=0;
    int iter_end_y_l_ls =0;
    int iter_end_y_l_rs =0;
    int iter_end_y_r_ls =0;
    int iter_end_y_r_rs =0;
    int iter_end_y_r =0;
    int iter_end_yaw_l=0;
    int iter_end_yaw_r=0;
    double joy_x;
    double joy_y;
    double joy_height = 0.12;
    double joy_length =0.0;
    double joy_length_y =0.21;
    double joy_length_temp =0.2;
    double joy_length_y_temp =0.35;
    double joy_length_command =0.;
    double joy_length_y_r_command =0.;
    double joy_length_y_l_command =0.;
    double joy_length_previous=0.0;
    double joy_length_r =0.;
    double joy_length_l =0.;
    double joy_length_y_l =0.21;
    double joy_length_y_r =0.21;
    double joy_length_y_r_temp =0.21;
    double joy_length_y_l_temp =0.21;
    double joy_length_y_l_previous =0.21;
    double joy_length_y_r_previous =0.21;
    double joy_yaw_r =0.;
    double joy_yaw_l =0.;
    double joy_yaw_l_previous =0.;
    double joy_yaw_r_previous =0.;
    double joy_yaw_r_command =0.;
    double joy_yaw_l_command =0.;
    double swing_previous_l =0.;
    double swing_previous_r =0.;
    double joy_temp;
    
    
    double joy_left_foot_pos;
    double joy_right_foot_pos;
    double joy_walking_tick;
    double previous_l =0.;
    double previous_r =0.;
    double current_l =0.;
    double current_r =0.;
    double previous_lswing =0.;
    double previous_rswing =0.;
    double current_lswing =0.;
    double current_rswing =0.;


    std::vector<int> last_buttons_0;
    std::vector<int> last_buttons_1;
    std::vector<int> last_buttons_2;
    std::vector<int> last_buttons_3;
    std::vector<double> joy_walking_tick_temp;
    std::vector<double> joy_length_previous_vec= std::vector<double>(10,0.0);;
    std::vector<double> swing_previous_r_vec;
    std::vector<double> swing_previous_l_vec;
    std::vector<double> joy_yaw_l_previous_vec = std::vector<double>(10,0.0);
    std::vector<double> joy_yaw_r_previous_vec= std::vector<double>(10,0.0);
    std::vector<double> joy_length_y_r_previous_vec= std::vector<double>(10,0.21);;
    std::vector<double> joy_length_y_l_previous_vec= std::vector<double>(10,0.21);
    std::vector<double> last_buttons_7;

    double x_error = 0;
    double y_error = 0;
    double yaw_error = 0;

    double norm;

    // PREVIEW CONTROL
    void updateInitialState();
    void updateFootstepCommand();
    void updateFootstepCommand_intern();
    void getRobotState();
    void walkingStateMachine();
    void calculateFootStepTotal();
    void supportToFloatPattern();
    void floatToSupportFootstep();
    void updateNextStepTime();
    void resetPreviewState();
    void windupPreview();
    void computeIkControl(const Eigen::Isometry3d &float_trunk_transform, const Eigen::Isometry3d &float_lleg_transform, const Eigen::Isometry3d &float_rleg_transform, Eigen::Vector12d &q_des);

    void getZmpTrajectory();
    void addZmpOffset();
    void zmpGenerator(const unsigned int norm_size);
    void onestepZmp(unsigned int current_step_number, Eigen::VectorXd &temp_px, Eigen::VectorXd &temp_py, Eigen::VectorXd& temp_yaw, Eigen::VectorXd &temp_yawvel);
    void comHeuristicGenerator(const unsigned int norm_size);
    void onestepCoMHeuri(unsigned int current_step_number, Eigen::VectorXd &temp_px, Eigen::VectorXd &temp_py, Eigen::VectorXd &temp_vx, Eigen::VectorXd &temp_vy, Eigen::VectorXd &temp_yaw, Eigen::VectorXd &temp_yawvel); // CoM Yaw as well.
    void onestepCoMHeuri2(unsigned int current_step_number, Eigen::VectorXd &temp_px, Eigen::VectorXd &temp_py, Eigen::VectorXd &temp_vx, Eigen::VectorXd &temp_vy, Eigen::VectorXd &temp_yaw, Eigen::VectorXd &temp_yawvel); // CoM Yaw as well.
    void onestepCoMHeuri3(unsigned int current_step_number, Eigen::VectorXd &temp_px, Eigen::VectorXd &temp_py, Eigen::VectorXd &temp_vx, Eigen::VectorXd &temp_vy, Eigen::VectorXd &temp_yaw, Eigen::VectorXd &temp_yawvel); // CoM Yaw as well.

    void getComTrajectory(); 
    void previewcontroller(double dt, int NL, int tick, 
                           Eigen::Vector3d &x_k, Eigen::Vector3d &y_k, double &UX, double &UY,
                           const Eigen::MatrixXd &Gi, const Eigen::VectorXd &Gd, const Eigen::MatrixXd &Gx, 
                           const Eigen::MatrixXd &A,  const Eigen::VectorXd &B,  const Eigen::MatrixXd &C);
    void preview_Parameter(double dt, int NL, Eigen::MatrixXd& Gi, Eigen::VectorXd& Gd, Eigen::MatrixXd& Gx, Eigen::MatrixXd& A, Eigen::VectorXd& B, Eigen::MatrixXd& C);
    void getComTrajectory_mpc();
    void getFootTrajectory(); 
    void getTargetState(); 

    // Data Collection
    double step_length_x_fixed = 0.45;
    double step_length_y_fixed = 0.21;
    double target_heading = 0.0;
    double heading_error = 0.0;
    double t_dsp_fixed = 0.05;
    double t_ssp_fixed = 0.5;
    double foot_height_fixed = 0.06;

    static double wrap_to_pi(double angles){
        angles = fmod(angles, 2*M_PI);
        if (angles > M_PI){
          angles -= 2*M_PI;    
        }
        return angles;
    }


private:
    Eigen::VectorQd ControlVal_;

    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    unsigned int walking_tick = 0;
    unsigned int walking_tick_container = 0;

};