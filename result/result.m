data = importdata("log.txt");

vrp = data(:, [1,2,3]);
com_des = data(:, [4,5,6]);
pelv_des = data(:, [7,8,9]);
lfoot_des = data(:, [10,11,12]);
rfoot_des = data(:, [13,14,15]);
q_des = data(:, [16:21]);
action_q = data(:, [22:27]);
q_ = data(:, [28:33]);

fig = figure;
for index = 1:3
    subplot(3,1,index);
    plot(vrp(:, index));
    hold on;
    plot(com_des(:, index));
    plot(pelv_des(:, index));
    hold off;
end

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
fig2 = figure;
plot(data(:,[34:39]));
legend;

fig3 = figure;
for index = 1:3
    subplot(1,3,index);
    plot(data(:,[39+index, 42+index]));
    legend;
end