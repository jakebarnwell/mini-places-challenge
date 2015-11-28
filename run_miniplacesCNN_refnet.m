run(fullfile(fileparts(mfilename('fullpath')), ...
  'matconvnet', 'matlab', 'vl_setupnn.m')) ;

% load pre-trained model
load('categoryIDX.mat');
path_model = 'net-epoch-60.mat';
load([path_model]) ;

% Run for all images in the valid set
files = dir('data/images/val/*.jpg');
num_files = length(files);

fileID = fopen('val_results.txt','w');
formatSpec = '%s %d %d %d %d %d\n';

% % Run for all images in the test set
% files = dir('data/images/test/*.jpg');
% fileID = fopen('test_results.txt','w');

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
    
%     line_data = cell(1, 6);
%     line_data(1) = java.lang.String(line_data.name);
%     
%     for i=1:5
%     %     write as a line to a text file
%         line_data(i+1) = idx_sort(i);
% %         disp(sprintf('%s (%d), score %.3f', categoryIDX{idx_sort(i),1}, idx_sort(i), score_sort(i)));
%     end
    
%     line_data = {};
    
    % Write the data to the text file
    fprintf(fileID,formatSpec,strcat('val/', file.name), idx_sort(1), ...
        idx_sort(2), idx_sort(3), idx_sort(4), idx_sort(5));
end

fclose(fileID);
