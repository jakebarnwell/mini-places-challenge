close all;
clear all;

run(fullfile(fileparts(mfilename('fullpath')), ...
    'matconvnet', 'matlab', 'vl_setupnn.m')) ;

% load pre-trained model
load('categoryIDX.mat');

num_models = 1;

% Trained CNN #1
path_model = 'nets/refaug-big8/net-epoch-60.mat';
load([path_model]) ;
disp(strcat('Loaded Model ', path_model));

% change the last layer of CNN from softmaxloss to softmax
net.layers{1,end}.type = 'softmax';
net.layers{1,end}.name = 'prob';

net_struct(1) = net;
% info_struct(1) = info; don't think this is used

% % Trained CNN #2
% path_model = 'nets/refnet1-again/net-epoch-60.mat';
% load([path_model]) ;
% disp(strcat('Loaded Model ', path_model));
% 
% % change the last layer of CNN from softmaxloss to softmax
% net.layers{1,end}.type = 'softmax';
% net.layers{1,end}.name = 'prob';
% 
% net_struct(2) = net;
% % info_struct(2) = info;

% WHETHER RUNNING ON VAL OR TEST SET, CHANGE THIS TO CHANGE WHAT RUNNING ON
run_on_val_set = true;

if run_on_val_set
    % Run for all images in the valid set
    files = dir('data/images/val/*.jpg');
    num_files = length(files);
    
    fileID = fopen('val_results.txt','w');
    formatSpec = '%s %d %d %d %d %d\n';
    disp('RUNNING ON VAL SET');
else
    % Run for all images in the test set
    files = dir('data/images/test/*.jpg');
    fileID = fopen('test_results.txt','w');
    
    formatSpec = '%s %d %d %d %d %d\n';
    disp('RUNNING ON TEST SET');
end


for file = files'
    for j=1:num_models
        net = net_struct(j);
        % load and preprocess an image
        if run_on_val_set
            im = imread(fullfile('data/images/val/', file.name)) ;
        else
            im = imread(fullfile('data/images/test/', file.name)) ;
        end
        
        im_resize = imresize(im, net.normalization.imageSize(1:2)) ;
        im_ = single(im_resize) ;
        for i=1:3
            im_(:,:,i) = im_(:,:,i)-net.normalization.averageImage(i);
        end
        
        % Averaging Models
        scores = single(zeros(100,1));
        
        % run the CNN
        res = vl_simplenn(net, im_) ;
        temp_scores = squeeze(gather(res(end).x));
        scores = scores + temp_scores;
    end
    
    scores = scores ./ num_models;
    
    [score_sort, idx_sort] = sort(scores,'descend') ;
    
    if run_on_val_set
        % Write the data to the text file
        fprintf(fileID,formatSpec,strcat('val/', file.name), idx_sort(1)-1, ...
            idx_sort(2)-1, idx_sort(3)-1, idx_sort(4)-1, idx_sort(5)-1);
    else
        % Write the data to the text file
        fprintf(fileID,formatSpec,strcat('test/', file.name), idx_sort(1)-1, ...
            idx_sort(2)-1, idx_sort(3)-1, idx_sort(4)-1, idx_sort(5)-1);
    end
end

fclose(fileID);
