% import csv data to matrix
data_int = importdata('eval_data_int.csv');
data_heuri = importdata('eval_data_heu.csv');
% data_ral = importdata('eval_data_ral.csv');

xy_int = data_int(:,1);
x_int = data_int(:,2);
y_int = data_int(:,3);

xy_heuri = data_heuri(:,1);
x_heuri = data_heuri(:,2);
y_heuri = data_heuri(:,3);

% xy_ral = data_ral(:,1);             
% x_ral = data_ral(:,2);
% y_ral = data_ral(:,3);

means_intern = [mean(xy_int),mean(x_int),mean(y_int)];
stds_intern = [std(xy_int),std(x_int),std(y_int)];
means_heuri = [mean(xy_heuri),mean(x_heuri),mean(y_heuri)];
stds_heuri = [std(xy_heuri),std(x_heuri),std(y_heuri)];
% means_ral = [mean(xy_ral),mean(x_ral),mean(y_ral)];
% stds_ral = [std(xy_ral),std(x_ral),std(y_ral)];