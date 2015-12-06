function imdb = setup_data(varargin)
%    Need to have repo/data/images/{train,val,test}
%     and repo/data/objects/...
%
%    In order to speedup training and testing, it may be a good idea
%    to preprocess the images to have a fixed size (e.g. 256 pixels
%    high) and/or to store the images in RAM disk (provided that
%    sufficient RAM is available). Reading images off disk with a
%    sufficient speed is crucial for fast training.

NUM_TRAINING_IMAGES = 100000;
NUM_VAL_IMAGES = 10000;
NUM_TEST_IMAGES = 10000;

% This needs to be the directory containing our 'images' directory
opts.dataDir = fullfile('..','data') ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

if numel(dir(fullfile(opts.dataDir, 'objects'))) == 0
    error('Make sure that data/objects/... exists!');
end
if numel(dir(fullfile(opts.dataDir, 'images'))) == 0
    error('Make sure that data/images/... exists!');
end

% In vanilla, cats is a 1x1000 cell array of all of the categories
% (classes) ID's, e.g. n02119789, ...
% descrs is a 1x1000 cell array of all the classes names, e.g. 'English
% Setter', 'Siberian Husky', 'kit fox, Vulpes macrotis'

categories = table2cell(readtable(fullfile(fileparts(mfilename('fullpath')), ...
  '..','development_kit', 'data', 'categories.txt'), 'Delimiter',' ', ...
  'ReadVariableNames', false));

% Category names are indexes, descrs are human-readable descriptions. Note
% that we add 1 to each of the indexes since we want them to start at 1.
cats = cellfun(@(x) x+1, categories(:,2));
descrs = categories(:,1);

% 1xNumCategories cell array for name and description
imdb.classes.name = cats' ;
imdb.classes.description = descrs' ;
% This is the top-level directory of image data 
imdb.imageDir = fullfile(opts.dataDir, 'images') ;

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------

fprintf('Searching training images ...\n') ;
names = {} ;
labels = {} ;

% Get image paths and classnames
imagePaths = getAllFiles(fullfile(opts.dataDir, 'images', 'train'));
stripper = stripDirectoryWrapper(fullfile(opts.dataDir, 'images', 'train'));
imagePathsStripped = cellfun(@(s) stripper(s), imagePaths, 'UniformOutput', false);

% Wrapper function for stripDirectory
function fn = stripDirectoryWrapper(directory)
    fn = @(s) stripDirectory(s, directory);
end

% Helper function to get a category and image name from a full path name
function stripped = stripDirectory(s, directory)
    stripped = s(numel(directory)+2:end);
end

names = cellfun(@(s) strcat('train', filesep, s), imagePathsStripped, 'UniformOutput', false);
labels = cellfun(@(s) getLabel(s), imagePathsStripped);

function lab = getLabel(imagePathStripped)
    inds = strfind(imagePathStripped, filesep);
    category = strcat(filesep, imagePathStripped(1:inds(end)-1));
    [~, lab] = ismember(category, descrs);
end

if 0
% if numel(names) ~= NUM_TRAINING_IMAGES;
  warning('Found %d instead of %d training images. Dropping training set.', numel(names), NUM_TRAINING_IMAGES)
  names = {} ;
  labels = [] ;
end

%imdb.images.id is just a unique id for each image, 1 through whatever
%imdb.images.set should all be 1 since these are all training images

imdb.images.id = 1:numel(names) ;
imdb.images.name = names' ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels' ;

% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------

fprintf('Searching validation images ...\n') ;

valLabelsPath = fullfile('..','development_kit', 'data', 'val_fake.txt');
validation = table2cell(readtable(valLabelsPath, 'Delimiter', ' ', ...
    'ReadVariableNames', false));
ims = validation(:,1);
names = sort(ims);
labels = cellfun(@(x) x+1, validation(:,2));

if 0
% if numel(ims) ~= NUM_VAL_IMAGES
  warning('Found %d instead of %d validation images. Dropping validation set.', numel(ims), NUM_VAL_IMAGES)
  names = {} ;
  labels = [] ;
end

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names') ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels') ;

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------

fprintf('Searching test images ...\n') ;

ims = dir(fullfile(opts.dataDir, 'images', 'test', '*.jpg')) ;
names = sort({ims.name}) ;
labels = zeros(1, numel(names)) ;

if 0
% if numel(labels) ~= NUM_TEST_IMAGES
  warning('Found %d instead of %d test images', numel(labels), NUM_TEST_IMAGES)
end

names = strcat(['test' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

end

