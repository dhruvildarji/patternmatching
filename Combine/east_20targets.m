close all;
figure;
Targets_x = [10.619636632452558,11.228544575737994,11.472210465039678,11.53157749271567,11.738353910604092,11.720271671086385,12.109896935945962,];
Targets_y = [
21.60751223972419,22.2722100450375,21.713012518701127,23.027165475680366,21.457577948639788,22.012453159170995,21.437212462409786,];
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

Targets2_x = [11.496678322372773,11.265001157383523,11.678911065845343,11.494467190886255,11.699196990122335,10.757751664654538,11.049771428755646,11.581684669485096,10.681119835647127,9.602685268976877,]
Targets2_y = [
22.856306401367835,22.238769527374377,22.002314711308532,21.756328755178238,21.544576662284776,21.706692616101463,21.34592999354511,21.03401866844248,21.231919470731018,21.035493282483408,]
center2_x = [9.5034465019176]
center2_y = [22.02905554641843]
image_corners2_x = [11.682767399769892,7.2117559216916725,7.324125604065308,11.795137082143526,11.682767399769892]
image_corners2_y = [23.395900780546217,23.19775557396071,20.662210312290643,20.86035551887615,23.395900780546217];
plot(Targets2_x, Targets2_y,'bo',image_corners2_x,image_corners2_y,'b');
hold on;
plot(center2_x,center2_y,'bx','MarkerSize',12)
labs1 = ['C2'];
labelpoints(center2_x,center2_y,labs1,'NE',0.3,1)
axis equal;
axis tight;
hold on;

set(gca,'XTickLabel',[0.5 2.5 5 7.5 10 12.5 15] );
set(gca,'YTickLabel',[0.5 1 1.5 2 2.5 3 3.5 4 4.5] );
