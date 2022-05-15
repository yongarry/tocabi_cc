#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"


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
    void processObservation();
    void feedforwardPolicy();
    void initVariable();

    static const int num_state = 70;
    static const int num_hidden = 256;
    static const int num_action = 33;

    Eigen::Matrix<float, num_hidden, num_state> policy_net_w0_;
    Eigen::Matrix<float, num_hidden, 1> policy_net_b0_;
    Eigen::Matrix<float, num_hidden, num_hidden> policy_net_w2_;
    Eigen::Matrix<float, num_hidden, 1> policy_net_b2_;
    Eigen::Matrix<float, num_action, num_hidden> action_net_w_;
    Eigen::Matrix<float, num_action, 1> action_net_b_;
    Eigen::Matrix<float, num_hidden, 1> hidden_layer1_;
    Eigen::Matrix<float, num_hidden, 1> hidden_layer2_;
    Eigen::Matrix<float, num_action, 1> rl_action_;
    
    
    Eigen::Matrix<float, num_state, 1> state_;
    Eigen::Matrix<float, num_state, 1> state_mean_;
    Eigen::Matrix<float, num_state, 1> state_var_;

    std::ofstream writeFile;

    bool is_on_robot_ = true;
    Eigen::Matrix<double, MODEL_DOF, 1> q_lpf_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;
    Eigen::Matrix<double, MODEL_DOF, 1> rl_action_lpf_;
    Eigen::Matrix<double, 3, 1> euler_angle_lpf_;

    float start_time_;
    float time_inference_pre_;

    Eigen::Vector3d euler_angle_;

private:
    Eigen::VectorQd ControlVal_;
};
