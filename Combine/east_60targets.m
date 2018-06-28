close all;
figure;
Targets_x = [9.935796782364969,10.619636632452558,10.626033256090645,11.228544575737994,11.250231679163441,11.472210465039678,11.53157749271567,11.738353910604092,11.720271671086385,12.109896935945962,12.091678758488731,];
Targets_y = [
21.4243764060143,21.60751223972419,22.173181914098258,22.2722100450375,22.58439610664356,21.713012518701127,23.027165475680366,21.457577948639788,22.012453159170995,21.437212462409786,22.038809978680266,];
center_x = [9.87568460453193];
center_y = [22.472398504829062];
image_corners_x = [11.985028038995928,7.529205318884105,7.766341170067932,12.222163890179754,11.985028038995928];
image_corners_y = [23.944939079943104,23.526789529022288,20.99985792971502,21.418007480635836,23.944939079943104];
labs1 = 1:length(Targets_x);
plot(Targets_x, Targets_y,'go',image_corners_x,image_corners_y,'g');
hold on;
plot(center_x,center_y,'gx','MarkerSize',12)
labs1 = ['C1'];
labelpoints(center_x,center_y,labs1,'NE',0.3,1)
axis equal;
axis tight;
hold on;

Targets2_x = [11.618359007025802,11.487006381637562,11.528132969271134,11.983231954658788,11.058022478288652,11.86271197233604,12.104522910465858,11.154787119044734,11.52249955810122,11.732403750494624,10.637826734246634,12.10502170222971,11.177972814647854,11.597031763446841,10.612754807412678,10.958961461207972,10.18086430346668,9.443121511532603,]
Targets2_y = [
23.137213501285576,22.732643232136983,22.484791234427565,22.345240384850452,22.308520518131633,22.06216614405581,21.899678994757505,21.85628256166398,21.570005504371206,21.50859857338956,21.610180080154155,21.381719778089664,21.384014906072224,21.247825125521004,21.23454159862926,21.043394843427723,20.950127986728784,20.54733001961832,]
center2_x = [9.87031282368891]
center2_y = [21.897388860143085]
image_corners2_x = [11.69611943437536,7.3778604264858965,8.044506213002458,12.362765220891921,11.69611943437536]
image_corners2_y = [23.709607304549614,22.534088567658745,20.085170415736556,21.260689152627425,23.709607304549614]
plot(Targets2_x, Targets2_y,'bo',image_corners2_x,image_corners2_y,'b');
hold on;
plot(center2_x,center2_y,'bx','MarkerSize',12)
labs1 = ['C2'];
labelpoints(center2_x,center2_y,labs1,'NE',0.3,1)
axis equal;
axis tight;
hold on;

set(gca,'XTickLabel',[0.4 2.5 5 7.5 10 12.5 15] );
set(gca,'YTickLabel',[0.5 1 1.5 2 2.5 3 3.5 4 4.5] );