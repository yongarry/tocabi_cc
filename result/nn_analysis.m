%%
clear all;
d = load('data.csv');
var = load('var.csv');
mean = load('mean.csv');

%%
t = d(:,1);
euler_angle = d(:,2:3);
euler_angle_lpf = d(:,5:6);
q = d(:,8:40);
q_lpf = d(:,41:73);
qdot = d(:,74:106);
qdot_lpf = d(:,107:139);
tau = d(:,140:172);

phase = mod(t, 8.0) / 8.0;
sin_phase = sin(2*pi*phase);
cos_phase = cos(2*pi*phase);

%%
obs = [euler_angle, q, qdot, sin_phase, cos_phase];
obs_lpf = [euler_angle, q, qdot_lpf, sin_phase, cos_phase];
normalized_obs = obs;
normalized_obs_lpf = obs_lpf;
for i=1:size(obs,2)
    normalized_obs(:,i) = (normalized_obs(:,i) - mean(i)) / sqrt(var(i));
    normalized_obs_lpf(:,i) = (normalized_obs_lpf(:,i) - mean(i)) / sqrt(var(i));
end

%%
figure()
for i=1:size(obs,2)
    subplot(7,10,i)
    plot(normalized_obs(:,i))
    hold on
    plot(normalized_obs_lpf(:,i))
end
