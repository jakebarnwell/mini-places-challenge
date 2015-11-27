function [net, info] = refNet1_train(varargin)
% Training refNet1 CNN

run(fullfile(fileparts(mfilename('fullpath')), ...
  'matconvnet', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data','ILSVRC2012') ;
opts.modelType = 'refNet1';
opts.networkType = 'simplenn';
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
opts.expDir = fullfile('data', sprintf('places', ...
                                       sfx, opts.networkType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [] ; % jake you'll want to change this
opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.cudnn = true ;
opts.train.expDir = opts.expDir ;
if ~opts.batchNormalization
  opts.train.learningRate = logspace(-2, -4, 60) ;
else
  opts.train.learningRate = logspace(-1, -4, 20) ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end





opts.modelType = 'refNet1_train';
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.learningRate = 0.001 ; % not sure what should have here
opts.train.weightDecay = 0.001 ; % not sure what should have here

% Need to update with location of the data
opts.expDir = fullfile('data') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(opts.expDir, 'images', 'train');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train.batchSize = 100 ;
opts.train.continue = true ;

opts.train.gpus = [] ; % probably want to use this, ask for what should put here
% Jake, this is where you put in the GPU stuff

opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

categories = readtable(fullfile(fileparts(mfilename('fullpath')), ...
  'development_kit', 'data', 'categories.txt'), 'Delimiter',' ');

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getChallengeImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net = sample_refNet_initial();

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% Saving the trained CNN 
save sample_train.mat net info

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, im=fliplr(im) ; end
end

% --------------------------------------------------------------------
function imdb = getChallengeImdb(opts)
% --------------------------------------------------------------------
% Prepare the imdb structure, returns image data with mean image subtracted
% Need to update with path

img_paths = getAllFiles(opts.dataDir);
% We can do this same thing with the xml data stuff later
num_imgs = length(img_paths);

data = cell(1, num_imgs);
labels = cell(1, num_imgs);
sets = cell(1, num_imgs);

for i = 1:numel(num_imgs)
  data{i} = imread(char(img_paths(i)));
%   data{i} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
%   labels{fi} = fd.labels' + 1; % Index from 1

  sets{fi} = 1; % for the test set this should be set to 3?
end
% set needs to eventually be (I think) a 1 x 110k matrix
% where the first 100k are 1's (meaning that we have 100k
% training images) and the last 10 are 3's or 2's
% (meaning we have 10k validation/test images)
set = cat(2, sets{:});
% data = single(cat(4, data{:}));

% % remove mean in any case
% dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean);

% Add in these later to improve
% % normalize by image mean and std as suggested in `An Analysis of
% % Single-Layer Networks in Unsupervised Feature Learning` Adam
% % Coates, Honglak Lee, Andrew Y. Ng
% 
% if opts.contrastNormalization
%   z = reshape(data,[],60000) ;
%   z = bsxfun(@minus, z, mean(z,1)) ;
%   n = std(z,0,1) ;
%   z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
%   data = reshape(z, 32, 32, 3, []) ;
% end
% 
% if opts.whitenData
%   z = reshape(data,[],60000) ;
%   W = z(:,set == 1)*z(:,set == 1)'/60000 ;
%   [V,D] = eig(W) ;
%   % the scale is selected to approximately preserve the norm of W
%   d2 = diag(D) ;
%   en = sqrt(mean(d2)) ;
%   z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
%   data = reshape(z, 32, 32, 3, []) ;
% end

% clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
% imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
% imdb.meta.classes = clNames.label_names;

end

% Searches recursively through all subdirectories of a given directory, 
% collecting a list of all file paths it finds
function fileList = getAllFiles(dirName)

  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
  end
end
end