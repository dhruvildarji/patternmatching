%close all;
figure;
Targets1_x = [6.147936978521045,6.0801799141285695,6.258484504466308,6.557227498077212,6.803580815945082,7.066934555568254,7.346880108088991,7.441747821032568,7.6295215401933,7.812373691525818,8.192801175246332,8.190884100570226,8.435056362243778,8.511997812726454,8.469046474565815,8.711572563486335,8.973801172988045,9.01704417428634,9.287133451503472,9.636705401421313,9.683857975362738,9.790609063601245,10.03684924653274,10.288034868128186,10.413480292546502,10.552118969824802,];
Targets1_y = [
20.766077612520842,21.96878638855494,21.160223202516615,21.583255699183773,20.711071291268855,21.632874072052314,20.666964031746158,21.05052609398604,21.702208806311763,20.92410327640694,20.918413634859817,21.161330681044653,21.21887130410981,20.79379475702045,21.51341306193993,21.583536631132866,20.687832503715423,21.10730175295345,21.586311043628516,21.196901572854816,20.934305876924533,21.716170132890653,21.138332004173698,20.979218921811597,21.520361028596923,20.94977637048434,];
center_x = [8.354342344890089];
center_y = [21.020603321293827];
image_corners1_x = [10.408517903451672,5.970932235722346,6.300166786328505,10.737752454057832,10.408517903451672]
image_corners1_y = [22.56917306285595,21.988622805287086,19.472033579731704,20.052583837300567,22.56917306285595]
plot(Targets1_x, Targets1_y,'go',image_corners1_x,image_corners1_y,'g');
hold on;
plot(center1_x,center1_y,'gx','MarkerSize',12)
labs1 = ['C1'];
labelpoints(center1_x,center1_y,labs1,'NE',0.3,1)
axis equal;
axis tight;
hold on;
Targets2_x = [9.16720243210134,9.831969521394814,9.563242215222967,10.400408224911194,9.545114124670945,10.264512272852043,10.305365498076428,10.70755615091841,11.121494576422466,11.430880337773008,11.003956163084935,]
Targets2_y = [
21.227379469869227,21.322911436401615,20.78064333606445,21.52631178703101,20.33924864843885,20.970231174906694,20.697971704918217,20.96922264655434,21.415175365410878,21.461599484984045,20.849195275876443,];
center2_x = [11.389667915815668];
center2_y = [20.696171092086644];
image_corners2_x = [13.414282688845923,8.98843105417451,9.365053142785413,13.790904777456825,13.414282688845923]
image_corners2_y = [22.283193615777225,21.619083332859997,19.109148568396062,19.773258851313287,22.283193615777225]
plot(Targets2_x, Targets2_y,'bo',image_corners2_x,image_corners2_y,'b');
hold on;
plot(center2_x,center2_y,'bx','MarkerSize',12)
labs1 = ['C2'];
labelpoints(center2_x,center2_y,labs1,'NE',0.3,1)
axis equal;
axis tight;

set(gca,'XTickLabel',[0.5 1.25 2.5 3.75 5 6.25 7.5 8.75 10 11.25] );
set(gca,'YTickLabel',[0.5 1 1.5 2 2.5 3 3.5 4 4.5] );