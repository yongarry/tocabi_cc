%%
clear all;
d = load('data.csv');

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
q(:,24) = 0.0;
q(:,25) = 0.0;
qdot(:,24) = 0.0;
qdot(:,25) = 0.0;
qdot_lpf(:,24) = 0.0;
qdot_lpf(:,25) = 0.0;

%%
obs = [euler_angle, q, qdot, sin_phase, cos_phase];
obs_lpf = [euler_angle, q, qdot_lpf, sin_phase, cos_phase];
normalized_obs = obs;
normalized_obs_lpf = obs_lpf;
for i=1:size(obs,2)
    normalized_obs(:,i) = (normalized_obs(:,i) - mean(i)) / sqrt(var(i)+1e-8);
    normalized_obs_lpf(:,i) = (normalized_obs_lpf(:,i) - mean(i)) / sqrt(var(i)+1e-8);
end

%%
figure()
for i=1:size(obs,2)
    subplot(7,10,i)
    plot(t,normalized_obs(:,i))
    hold on
    plot(t,normalized_obs_lpf(:,i))
end

%%
w0 = load('mlp_extractor_policy_net_0_weight.txt');
b0 = load('mlp_extractor_policy_net_0_bias.txt');
w2 = load('mlp_extractor_policy_net_2_weight.txt');
b2 = load('mlp_extractor_policy_net_2_bias.txt');
w4 = load('action_net_weight.txt');
b4 = load('action_net_bias.txt');
var = load('obs_variance.txt');
mean = load('obs_mean.txt');

normalized_obs(normalized_obs < -3.0) = -3.0;
normalized_obs(normalized_obs > 3.0) = 3.0;

layer0 = normalized_obs*w0' + b0';
layer0(layer0 < 0.0) = 0.0;

layer2 = layer0*w2' +b2';
layer2(layer2 < 0.0) = 0.0;

output = layer2*w4' +b4';


normalized_obs_lpf(normalized_obs_lpf < -3.0) = -3.0;
normalized_obs_lpf(normalized_obs_lpf > 3.0) = 3.0;

layer0_lpf = normalized_obs_lpf*w0' + b0';
layer0_lpf(layer0_lpf < 0.0) = 0.0;

layer2_lpf = layer0_lpf*w2' +b2';
layer2_lpf(layer2_lpf < 0.0) = 0.0;

output_lpf = layer2_lpf*w4' +b4';

figure()
for i=1:33
    subplot(6,6,i)
    plot(t,output(:,i))
    hold on
    plot(t,output_lpf(:,i))
end

%%
input = repmat(obs(5000,:),100,1);
for i=2:size(input,1)
    input(i,38) = input(i-1,38) + 0.01; 
end

normalized_obs = input;
for i=1:size(obs,2)
    normalized_obs(:,i) = (normalized_obs(:,i) - mean(i)) / sqrt(var(i)+1e-8);
end


normalized_obs(normalized_obs < -3.0) = -3.0;
normalized_obs(normalized_obs > 3.0) = 3.0;

layer0 = normalized_obs*w0' + b0';
layer0(layer0 < 0.0) = 0.0;

layer2 = layer0*w2' +b2';
layer2(layer2 < 0.0) = 0.0;

output = layer2*w4' +b4';
figure()
for i=1:33
    subplot(6,6,i)
    plot(output(:,i))
end