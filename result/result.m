% import csv data to matrix
data = importdata('eval_data.csv');
xy = data(:,1);
x = data(:,2);
y = data(:,3);
yaw = data(:,4);


%means_intern = [mean(xy),mean(x),mean(y),mean(yaw)]
%stds_intern = [std(xy),std(x),std(y),std(yaw)]
means_heuri = [mean(xy),mean(x),mean(y),mean(yaw)]
stds_heuri = [std(xy),std(x),std(y),std(yaw)]
%means_ral = [mean(xy),mean(x),mean(y),mean(yaw)]
%stds_ral = [std(xy),std(x),std(y),std(yaw)]