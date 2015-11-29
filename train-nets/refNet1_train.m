function refNet1_train_imagenet(varargin)
% REFNET1_TRAIN_IMAGENET  Copies the style of cnn_imagenet
%   This tries to train the miniplaces competition net

addpath(fullfile('matconvnet','examples'));
addpath(fullfile('matconvnet','matlab'));

run(fullfile(fileparts(mfilename('fullpath')), ...
  'matconvnet', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data') ; %Contains our images/objects etc
opts.modelType = 'refNet1' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(opts.dataDir, 'refnet', ...
    sprintf('refnet-%s-%s', sfx, opts.networkType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [1, 2, 3, 4] ;
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

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  % TODO this function
  imdb = refNet1_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% All imdb.* fields are 1 x n (not n x 1)
% imdb.classes.name = {className1, className2, ...}
% imdb.classes.description = {classDescr1, classDescr2, ...}
% where "className" is like an ID, and "classDescr" is like 'White Tiger'
% imdb.imageDir is the top level directory of the images, e.g. if we had
% data/imagenet12/images/{test,train,val} then imdb.imageDir is
% data/imagenet12/images
% imdb.images.id is an array of unique IDs of images, e.g. [1, 2, ...]
% imdb.images.name is a cell array of full path names (relative to
% imageDir) to each of the images, e.g. {train/classNameA/classAimage1.JPG,
% ...}
% imdb.images.set is an array of the set type of the images, where 1 =
% train, 2 = validate, 3 = test. So it will probably look like this:
%    [1 1 1 .... 2 2 2 .... 3 3 3 ... ]
% imdb.images.label is an array, where each element is the index into the
% classes cell array that tells us to which class the image belongs. e.g.
% [28 193 930 68 ... ]

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = sample_refNet_initial('model', opts.modelType, ...
			    'batchNormalization', opts.batchNormalization, ...
                        'weightInitMethod', opts.weightInitMethod) ;
bopts = net.normalization ;
bopts.numThreads = opts.numFetchThreads ;

% compute image statistics (mean, RGB covariances etc)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% One can use the average RGB value, or use a different average for
% each pixel
%net.normalization.averageImage = averageImage ;
net.normalization.averageImage = rgbMean ;

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

[v,d] = eig(rgbCovariance) ;
bopts.transformation = 'stretch' ;
bopts.averageImage = rgbMean ;
bopts.rgbVariance = 0.1*sqrt(d)*v' ;
useGpu = numel(opts.train.gpus) > 0 ;

%  getBatchSimpleNNWrapper, cnn_train stuff
fn = getBatchSimpleNNWrapper(bopts) ;
[net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true) ;

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...\n', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{t} = mean(temp, 4) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
