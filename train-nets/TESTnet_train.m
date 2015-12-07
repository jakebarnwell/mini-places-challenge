function TESTnet_train(varargin)
% REFNET1_TRAIN_IMAGENET  Copies the style of cnn_imagenet
%   This tries to train the miniplaces competition net

addpath(fullfile('..','matconvnet','examples'));
addpath(fullfile('..','matconvnet','matlab'));
addpath(fullfile('..'));

run(fullfile(fileparts(mfilename('fullpath')), ...
	     '..','matconvnet', 'matlab', 'vl_setupnn.m')) ;

% Contains our images/objects etc
opts.dataDir = fullfile('..','datatest') ;
opts.modelType = 'refNet1' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(opts.dataDir, 'testnet', ...
    sprintf('refnet-%s-%s', sfx, opts.networkType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 32 ;

opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
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
  imdb = TESTsetup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = cnn_init('model', opts.modelType, ...
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
[net,info] = cnn_train_better(net, imdb, fn, opts.train, 'conserveMemory', true) ;

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = get_batch_aug2(images, opts, ...
                            'prefetch', nargout == 0) ;
labels = kron(imdb.images.label(batch), ones(1, 2));

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
