close all;
clear all;

run(fullfile(fileparts(mfilename('fullpath')), ...
    'matconvnet', 'matlab', 'vl_setupnn.m')) ;

% load pre-trained model
load('categoryIDX.mat');

% Trained CNN - CHANGE THIS TO EVALUATE DIFFERENT NEURAL NETS
path_model = 'net-epoch-60.mat';
load([path_model]) ;

% WHETHER RUNNING ON VAL OR TEST SET, CHANGE THIS TO CHANGE WHAT RUNNING ON
run_on_val_set = true;

if run_on_val_set
    % Run for all images in the valid set
    files = dir('data/images/val/*.jpg');
    num_files = length(files);

    fileID = fopen('val_results.txt','w');
    formatSpec = '%s %d %d %d %d %d\n';
else
    % Run for all images in the test set
    files = dir('data/images/test/*.jpg');
    fileID = fopen('test_results.txt','w');
    
    fileID = fopen('test_results.txt','w');
    formatSpec = '%s %d %d %d %d %d\n';
end

% change the last layer of CNN from softmaxloss to softmax
net.layers{1,end}.type = 'softmax';
net.layers{1,end}.name = 'prob';

for file = files'
    % load and preprocess an image
    im = imread(fullfile('data/images/val/', file.name)) ;
    im_resize = imresize(im, net.normalization.imageSize(1:2)) ;
    im_ = single(im_resize) ;
    for i=1:3
        im_(:,:,i) = im_(:,:,i)-net.normalization.averageImage(i);
    end
    
    % run the CNN
    res = vl_simplenn(net, im_) ;
    
    scores = squeeze(gather(res(end).x)) ;
    [score_sort, idx_sort] = sort(scores,'descend') ;
    
    % Write the data to the text file
    fprintf(fileID,formatSpec,strcat('val/', file.name), idx_sort(1), ...
        idx_sort(2), idx_sort(3), idx_sort(4), idx_sort(5));
end

fclose(fileID);