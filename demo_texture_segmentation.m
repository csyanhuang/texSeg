clear 
close all;
clc

addpath('./tools');
warning off;

% parameter setting
mask_size = 5;  % 做标记的大小
T = 1;
numTrainPerRegion = 20;

beta1 = 2.5;
beta2 = 0.05;
lambda = 0.01;
alpha = 0.002;

% filters
f1 = fspecial('log',[5,5],.8);
f2 = fspecial('log',[7,7],1.2);
f3 = fspecial('log',[9,9],1.8);
f4 = gabor_fn(3.5,pi/2);
f5 = gabor_fn(3.5,0);
f6 = gabor_fn(3.5,pi/4);
f7 = gabor_fn(3.5,-pi/4);
f8 = gabor_fn(2.5,pi/2);
f9 = gabor_fn(2.5,0);
f10 = gabor_fn(2.5,pi/4);
f11 = gabor_fn(2.5,-pi/4);

% datapath
setName = 'Prague';
imgDir = ['./datasets/',setName]; % input image folder
imgName = 'tm1_1_1.png';
img = imread([imgDir,'/',imgName]);
   
% label generation
gt = imread([imgDir,'/gt' imgName(3:end-6) '.png']);
mask = im2mask(gt,mask_size,0);

% feature extraction
cf = makecform('srgb2lab');
Ilab = applycform(img,cf);
Ig1 = subImg(Ilab(:,:,1),f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11);
Ig = cat(3,single(Ilab),Ig1);
ws = 60;
[Y_cols,EdgeMap] = featExtraction(Ig,ws);
Y_cols = double(Y_cols);

 %% performing segmentation
opts.T = T;
opts.numTrainPerRegion = numTrainPerRegion;
opts.EdgeMap = EdgeMap;
opts.mask_size = mask_size;
[Lab1] = WSSCG(Y_cols,mask,beta1,beta2,lambda,alpha,opts);

figure,imagesc(Lab1)



