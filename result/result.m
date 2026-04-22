data_ = importdata("log.txt");

obs = data_(:, [1:54]);
data = data_(:, [55:end]);

vrp = data(:, [1,2,3]);
com_des = data(:, [4,5,6]);
pelv_des = data(:, [7,8,9]);
lfoot_des = data(:, [10,11,12]);
rfoot_des = data(:, [13,14,15]);
q_des = data(:, [16:21]);
action_q = data(:, [22:27]);
q_ = data(:, [28:33]);

%% com trajectory plot
fig = figure;
for index = 1:3
    subplot(3,1,index);
    plot(vrp(:, index));
    hold on;
    plot(com_des(:, index));
    plot(pelv_des(:, index));
    hold off;
end

%% joint trajectory plot
fig1 = figure;
for index = 1:6
   subplot(3,2,index);
   plot(q_(1:end, index));
   hold on;
   plot(q_des(1:end, index));
   % plot(action_q(1:end, index));
   hold off;
    title(['Plot for Index ', num2str(index)]);
    xlabel('Time');
    ylabel('Value');
    ylim([-1.5,2.0])
end

%% torque trajectory plot
fig2 = figure;
for index = 1:6
    subplot(3,2,index);
    plot(data(:, [33+index]));
    legend;
end

%% foot trajectory plot 
fig3 = figure;
for index = 1:3
    subplot(1,3,index);
    plot(data(:,[39+index, 42+index]));
    if (index == 3)
        ylim([-0.5,0.5]);
    end
    legend;
end

%% etc
% fig4 = figure;
% for index = 0:1
%     subplot(2,1,index+1);
%     plot(data(:,[40+index*3+2,49+index*3+2]));
%     legend;
%     grid on;
% end

%% obs traj plot
fig5 = figure;
subplot(3,2,1);
plot(obs(:,[1:3]));

subplot(3,2,2);
plot(obs(:,[4:6]));

subplot(3,2,3);
plot(obs(:,[7:9]));

subplot(3,2,4);
plot(obs(:,[10:21]));

subplot(3,2,5);
plot(obs(:,[22:33]));

subplot(3,2,6);
plot(obs(:,[34:45]));
