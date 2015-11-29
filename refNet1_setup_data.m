function imdb = refNet1_setup_data(varargin)
%    Need to have repo/data/images/{train,val,test}
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
opts.dataDir = fullfile('data') ;
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
  'development_kit', 'data', 'categories.txt'), 'Delimiter',' ', ...
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

if numel(names) ~= NUM_TRAINING_IMAGES;
  warning('Found %d instead of %d training images. Dropping training set.', numel(names), NUM_TRAINING_IMAGES)
  names = {} ;
  labels = [] ;
end


% In vanilla, names is a 1x60658 cell array of directory/imagename.JPEG
% where directory is the directory directly containing the image. I *THINK*
% that the directory name is typically the category ID of the enclosed
% images, e.g. if there is a directory called n02119789, then
% n02119789/image1.JPEG, n02119789/image2.JPG, ..., n02119789/imagek.JPEG
% are all of the training images of class n02119789. There is presumably
% one such directory for each of the classes.
% The variable d loops through each of the directories (each directory has
% multiple images in that class). d.name is the name of the directory, e.g.
% n02119789 (the name of the category). [~,lab] = ... returns into lab the
% index of 'cats' that d.name (the category) is, if it's in that array at
% all. For example, if d.name = n02119789 and n02119789is the 900'th
% element of cats, then lab=900. 
% Hence, 'labels' is a 1x60658 array where each entry is the index/number
% of the category that belongs to the respective image.

%imdb.images.id is just a unique id for each image, 1 through whatever
%imdb.images.set should all be 1 since these are all training images

imdb.images.id = 1:numel(names) ;
imdb.images.name = names' ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels' ;

% The above pattern continues below for validation and test images. It just
% adds those images to the end of imdb.images.id etc. So for example,
% imdb.images.name is [classA/im1.JPG, classA/im2.JPG, ..., classA/imk.JPG,
% classB/... , ... , ... , classZ/imK.JPG, val/valImg1.JPG,
% val/valImg2.JPG, val/valImg3.JPG, ..., test/testImg1.JPG,
% test/testImg2.JPG, ...]
% The reaason that they add 1e7 to the IDs is just so offset them enough so
% that there's no confusion about what IDs belong to what. So the training
% images have IDs from 1 to numel(trainImages), the validation images have
% IDs from 10000000 to 10000000+numel(valImages), the test images have IDs
% from 20000000 to 20000000+numel(testImages). Note also that
% imdb.images.set is 2 for validation, 3 for test, and 1 for train.
% We probably don't have to worry about post-processing for now.

% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------

fprintf('Searching validation images ...\n') ;

valLabelsPath = fullfile('development_kit', 'data', 'val.txt');
validation = table2cell(readtable(valLabelsPath, 'Delimiter', ' ', ...
    'ReadVariableNames', false));
ims = validation(:,1);
names = sort(ims);
labels = cellfun(@(x) x+1, validation(:,2));

if numel(ims) ~= NUM_VAL_IMAGES
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

if numel(labels) ~= NUM_TEST_IMAGES
  warning('Found %d instead of %d test images', numel(labels), NUM_TEST_IMAGES)
end

names = strcat(['test' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;


end

