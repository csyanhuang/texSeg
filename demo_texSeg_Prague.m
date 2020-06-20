clear 
close all;
clc

addpath('./tools');
warning off;

% datapath
imgDir='./datasets/Prague'; % input image folder
% paremeter setting
mask_size=9;  % 做标记的大小 5*5正方形
ws=60;  % 直方图统计窗口大小
T = 7;
numTrainPerRegion = 20;

beta1 = 2.5;
beta2 = 0.05;
lambda = 0.01;
alpha = 0.002;

opts.T = T;
opts.numTrainPerRegion = numTrainPerRegion;
opts.mask_size = mask_size;

nrings = 0; % parameter for generating mask 

% filters
f1=fspecial('log',[5,5],.8);
f2=fspecial('log',[7,7],1.2);
f3=fspecial('log',[9,9],1.8);
f4=gabor_fn(3.5,pi/2);
f5=gabor_fn(3.5,0);
f6=gabor_fn(3.5,pi/4);
f7=gabor_fn(3.5,-pi/4);
f8=gabor_fn(2.5,pi/2);
f9=gabor_fn(2.5,0);
f10=gabor_fn(2.5,pi/4);
f11=gabor_fn(2.5,-pi/4);

% results save path
savepath = ['./ResWsscgPrague_','S_',num2str(mask_size),'_T_',num2str(T),'_nTrain_',num2str(numTrainPerRegion),'_b1_',num2str(beta1),'_b2_',num2str(beta2),'_lam_',num2str(lambda),'_a_',num2str(alpha)];

segDir1=[savepath,'/noprocess']; % output image folder  算法直接出来的结果保存路径

if ~exist(segDir1)
    mkdir(segDir1);
end
segDir2=[savepath,'/process']; % output image folder  结果经过后处理后的保存路径
if ~exist(segDir2)
    mkdir(segDir2);
end
segDir3=[savepath,'/processB']; % output image folder  结果经过后处理后的保存路径
if ~exist(segDir3)
    mkdir(segDir3);
end

metricDir = [savepath,'/metricValue'];
if ~exist(metricDir,'dir')
    mkdir(metricDir);
end

%% 
iids = dir(fullfile(imgDir,'tm*.png'));
for n=1:numel(iids)
    n 
    rng(0,'v5uniform')
    if exist([segDir2 '/seg' iids(n).name(3:end-4) '.png'],'file')
      continue;
    end
    img=imread([imgDir,'/',iids(n).name]);
   
    %% label generation
    gt=imread([imgDir,'/gt' iids(n).name(3:end-6) '.png']);
    mask=im2mask(gt,mask_size,nrings);
   
    %% feature extraction
    cf=makecform('srgb2lab');
    Ilab=applycform(img,cf);
    Ig1=subImg(Ilab(:,:,1),f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11);
%     Igg=log(1+2000*Ig1.^2);
    Ig=cat(3,single(Ilab),Ig1);
    [Y_cols,EdgeMap]=featExtraction(Ig,ws);
    Y_cols=double(Y_cols);
    
     %% performing segmentation
    [Lab1]=WSSCG(Y_cols,mask,beta1,beta2,lambda,alpha,opts);
%     figure
%     imagesc(Lab1)
    imwrite(uint8(Lab1),[segDir1 '/seg' iids(n).name(3:end-4) '.png'])
    opts1 = struct;
    opts1.smThr1 = 400;
    opts1.smThr2 = 1200;
    opts1.ratioThr = 10;
    opts1.edgeThr = 0.2;
    [res_cc]=TxtMerge(Lab1, EdgeMap,opts1);  %  后处理
%     figure
%     imagesc(res_cc)
    imwrite(uint8(res_cc),[segDir2 '/seg' iids(n).name(3:end-4) '.png'])
    
    resBdry = drawBdry(res_cc,img);
%     figure
%     imagesc(res_shw)
    imwrite(resBdry,[segDir3 '/Bdry' iids(n).name(3:end-4) '.png'])
end
gt_path = imgDir;
[criteria,mean_value] = evaluatePrague(segDir2,gt_path);
save([metricDir,'/','wsscgMetric.mat'],'criteria','mean_value');


