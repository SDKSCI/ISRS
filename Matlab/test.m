img = imread("2018.jpg");

[labels, numlabels]=ISRS(img,100);
save("2018.mat","labels")
Displaysuperpixels(labels,img,"2018_ISRS.jpg");