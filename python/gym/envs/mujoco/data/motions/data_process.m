clc
clear all

Deg2Rad = pi/180;
Rad2Deg = 180/pi;

fname = 'humanoid3d_walk.txt';
val = jsondecode(fileread(fname));
data = val.Frames;
total_time = 1.3;

% root position and orientation
data_quat2joint(:,1:8) = data(:,1:8);

% right hip rotation
data_quat2joint(:,9:11) = eulerd(quaternion(data(:,17:20)),'YXZ','frame');
data_quat2joint(:,11) = -data_quat2joint(:,11);
% right knee roatation
data_quat2joint(:,12) = -data(:,21)*Rad2Deg;
% right ankle rotation
right_ankle = eulerd(quaternion(data(:,22:25)),'ZXY','frame');
data_quat2joint(:,13) = -right_ankle(:,1);
data_quat2joint(:,14) = right_ankle(:,2);

% left hip rotation
data_quat2joint(:,15:17) = eulerd(quaternion(data(:,31:34)),'YXZ','frame');
data_quat2joint(:,17) = -data_quat2joint(:,17);
% left knee roatation
data_quat2joint(:,18) = -data(:,35)*Rad2Deg;
% left ankle rotation
left_ankle = eulerd(quaternion(data(:,36:39)),'ZXY','frame');
data_quat2joint(:,19) = -left_ankle(:,1);
data_quat2joint(:,20) = left_ankle(:,2);

% chest rotation
data_quat2joint(:,21:23) = eulerd(quaternion(data(:,9:12)),'YZX','frame');
data_quat2joint(:,22) = -data_quat2joint(:,22);

% right shoulder rotation
data_quat2joint(:,24:26) = eulerd(quaternion(data(:,26:29)),'YZX','frame');
data_quat2joint(:,25) = -data_quat2joint(:,25);
% right elbow roatation
data_quat2joint(:,27) = -data(:,30)*Rad2Deg;

% left shoulder rotation
data_quat2joint(:,28:30) = eulerd(quaternion(data(:,40:43)),'YZX','frame');
data_quat2joint(:,29) = -data_quat2joint(:,29);
% left elbow roatation
data_quat2joint(:,31) = -data(:,44)*Rad2Deg;

data_quat2joint(:,9:31) = data_quat2joint(:,9:31)*Deg2Rad;

original_data_dt = 0.033332;
cubic_data_dt = 0.005;

original_data_prev_idx = 39;
original_data_cur_idx = 1;
original_data_next_idx = 2;
original_data_next_next_idx = 3;
cubic_data_idx = 1;
cycle_finished = false;

tmp=zeros(3,1000);
for t=0.0:cubic_data_dt:total_time
    
    if(t > data(original_data_next_idx,1) && cycle_finished==false)
        if(original_data_cur_idx ==37)
            original_data_prev_idx = 37;
            original_data_cur_idx = 38;
            original_data_next_idx = 39;
            original_data_next_next_idx = 1;
        elseif(original_data_cur_idx ==38)
            original_data_prev_idx = 38;
            original_data_cur_idx = 39;
            original_data_next_idx = 1;
            original_data_next_next_idx = 2;
            cycle_finished = true;
        else
            original_data_prev_idx = original_data_cur_idx;
            original_data_cur_idx = original_data_next_idx;
            original_data_next_idx = original_data_next_next_idx;
            original_data_next_next_idx = original_data_next_next_idx+1;
        end
    end
    

    
    time_0 = data(original_data_cur_idx,1);
    time_f = data(original_data_next_idx,1);
    if (cycle_finished)
        time_f = 1.3;
    end
    data_cubic(cubic_data_idx,1) = t;
    
    for data_type_idx = 2:size(data_quat2joint,2)
        x_dot_0 = 0.0;%(data_quat2joint(original_data_next_idx,data_type_idx) - 2*data_quat2joint(original_data_cur_idx,data_type_idx) + data_quat2joint(original_data_prev_idx,data_type_idx)) / (2*original_data_dt);
        x_dot_f = 0.0;%(data_quat2joint(original_data_next_next_idx,data_type_idx) - 2*data_quat2joint(original_data_next_idx,data_type_idx) + data_quat2joint(original_data_cur_idx,data_type_idx)) / (2*original_data_dt);
        data_cubic(cubic_data_idx, data_type_idx) = cubic(t,time_0, time_f, data_quat2joint(original_data_cur_idx,data_type_idx), data_quat2joint(original_data_next_idx,data_type_idx), x_dot_0, x_dot_f);
    end
    
    tmp(1:3,cubic_data_idx) = [original_data_cur_idx;time_f;t];
    cubic_data_idx = cubic_data_idx+1;
end

save('processed_data.txt', 'data_quat2joint', '-ascii', '-double', '-tabs')
