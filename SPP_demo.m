clc
clear
close all

img = imread('cameraman.tif');
img = repmat(img,[1,1,3]);

net = alexnet();
fmap = activations(net, img, 'relu5');

pyramid{1}.pool_operator = 'max';
pyramid{1}.pool_size = [4,4];
pyramid{1}.stride = [2,2];

pyramid{2}.pool_operator = 'max';
pyramid{2}.pool_size = [3,3];
pyramid{2}.stride = [1,1];

pyramid{3}.pool_operator = 'max';
pyramid{3}.pool_size = [2,2];
pyramid{3}.stride = [1,1];

pyramid_fvec = SPP(fmap, pyramid);

