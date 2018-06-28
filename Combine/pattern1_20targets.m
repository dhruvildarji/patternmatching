close all;
figure;
Targets_x = [5.988717688103261,6.319206758700634,7.190109898399197,7.48410467058113,8.572485190322425,8.96307024387794,9.966015264236212,10.243151105356233,];
Targets_y = [
21.020478102167797,20.62348559160751,20.778421768415825,21.260788443116237,20.94268085936743,21.02011828015014,21.416227571768175,21.182512788273087,];
center_x = [7.902286360389227];
center_y = [22.09858297840011];
image_corners_x = [9.936954757334561,5.506968446292532,5.867617963443893,10.297604274485922,9.936954757334561];
image_corners_y = [23.6726954217616,23.03675010651803,20.52447053503862,21.160415850282188,23.6726954217616];
plot(Targets_x, Targets_y,'go',image_corners_x,image_corners_y,'g');
hold on;
plot(center_x,center_y,'gx','MarkerSize',12)
labs1 = ['C1'];
labelpoints(center_x,center_y,labs1,'NE',0.3,1)
axis equal;
axis tight;
hold on;

Targets2_x = [9.682699412067567,9.951723918476725,8.75539906839205,8.396515406570014,7.382822458184031,7.825649758881277,7.138187183948914,5.9942791669061934,6.334847608001705,5.472598539871912,]
Targets2_y = [
21.36958969411471,21.13079854460435,21.007080986680418,20.89382476949229,21.199278864997392,20.72597499759697,20.783197807620347,21.002141929257515,20.648311074118556,21.232439135850747,]
center2_x = [7.824649116993028]
center2_y = [20.32146482051199]
image_corners2_x = [9.871265761186546,5.4365572997896,5.77803247279951,10.212740934196455,9.871265761186546]
image_corners2_y = [21.88001086782135,21.277876312747207,18.762918773202628,19.365053328276773,21.88001086782135];
plot(Targets2_x, Targets2_y,'bo',image_corners2_x,image_corners2_y,'b');
hold on;
plot(center2_x,center2_y,'bx','MarkerSize',12)
labs1 = ['C2'];
labelpoints(center2_x,center2_y,labs1,'NE',0.3,1)

axis equal;
axis tight;
hold on;

set(gca,'XTickLabel',[0.5 1.25 2.5 3.75 5 6.25 7.5 8.75 10 11.25] );
set(gca,'YTickLabel',[0.5 1 1.5 2 2.5 3 3.5 4 4.5 5] );