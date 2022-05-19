d1 = load('data.csv');
d2 = load('data_low_frictionloss.csv');

figure();
for i=1:12
    subplot(2,6,i);
    plot(d1(:,i));
    hold on
    plot(d2(:,i));
end

figure();
for i=13:33
    subplot(4,6,i-12);
    plot(d1(:,i));
    hold on
    plot(d2(:,i));
end

%%
clear all
d3 = load('data.csv');

figure();
for i=1:33
    subplot(6,6,i);
    plot(d3(:,1),d3(:,7+i))
end


figure();
for i=1:33
    subplot(6,6,i);
    plot(d3(:,1),d3(:,73+i))
    hold on
plot(d3(:,1),d3(:,106+i))
end
