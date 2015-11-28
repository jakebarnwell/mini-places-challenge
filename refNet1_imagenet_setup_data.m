function imdb = refNet1_imagenet_setup_data(varargin)
% CNN_IMAGENET_SETUP_DATA  Initialize ImageNet ILSVRC CLS-LOC challenge data
%
%    Jake's version
%
%    In order to use the ILSVRC data with these scripts, please
%    unpack it as follows. Create a root folder <DATA>, by default
%
%    data/imagenet12
%
%    (note that this can be a simlink). Use the 'dataDir' option to
%    specify a different path.
%
%    Within this folder, create the following hierarchy:
%
%    <DATA>/images/train/ : content of ILSVRC2012_img_train.tar
%    <DATA>/images/val/ : content of ILSVRC2012_img_val.tar
%    <DATA>/images/test/ : content of ILSVRC2012_img_test.tar
%    <DATA>/ILSVRC2012_devkit : content of ILSVRC2012_devkit.tar
%
%    In order to speedup training and testing, it may be a good idea
%    to preprocess the images to have a fixed size (e.g. 256 pixels
%    high) and/or to store the images in RAM disk (provided that
%    sufficient RAM is available). Reading images off disk with a
%    sufficient speed is crucial for fast training.

opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

d = dir(fullfile(opts.dataDir, 'objects')) ;
d = [d dir(fullfile(opts.dataDir, 'images'))] ;
if numel(d) == 0
    error('Make sure that both data/images/... and data/objects/... exist');
end

categories = readtable(fullfile(fileparts(mfilename('fullpath')), ...
  'development_kit', 'data', 'categories.txt'), 'Delimiter',' ');

cat_cell = table2cell(categories);

% Note that the position of a category in the cell array is it's label
% number
cats = {cat_cell(1:height(categories))} ; % categories but not mapped to numbers
% descrs = {meta.synsets(1:1000).words} ; %TODO put descrs, if we have them, here
% In vanilla, cats is a 1x1000 cell array of all of the categories
% (classes) ID's, e.g. n02119789, ...
% descrs is a 1x1000 cell array of all the classes names, e.g. 'English
% Setter', 'Siberian Husky', 'kit fox, Vulpes macrotis'

imdb.classes.name = cats ;
% imdb.classes.description = descrs ;
imdb.imageDirs = getAllFiles(opts.dataDir); % This should be the full list of all image pathnames

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------

fprintf('searching training images ...\n') ;
names = {} ;
labels = {} ;
% maybe want to iterate a level down
for d = dir(fullfile(opts.dataDir, 'images', 'train', 'n*'))'
  [~,lab] = ismember(d.name, cats) ;
  ims = dir(fullfile(opts.dataDir, 'images', 'train', d.name, '*.JPG')) ;
  names{end+1} = strcat([d.name, filesep], {ims.name}) ;
  labels{end+1} = ones(1, numel(ims)) * lab ;
  fprintf('.') ;
  if mod(numel(names), 50) == 0, fprintf('\n') ; end
  %fprintf('found %s with %d images\n', d.name, numel(ims)) ;
end
names = horzcat(names{:}) ;
labels = horzcat(labels{:}) ;

if numel(names) ~= 100000
  warning('Found %d training images instead of 100,000. Dropping training set.', numel(names)) ;
  names = {} ;
  labels =[] ;
end

names = strcat(['train' filesep], names) ;
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
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;

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

ims = dir(fullfile(opts.dataDir, 'images', 'val', '*.JPG')) ;
names = sort({ims.name}) ;
labels = textread(valLabelsPath, '%d') ;

if numel(ims) ~= 10e3
  warning('Found %d instead of 10,000 validation images. Dropping validation set.', numel(ims))
  names = {} ;
  labels =[] ;
else
  if ~isempty(valBlacklistPath)
    black = textread(valBlacklistPath, '%d') ;
    fprintf('blacklisting %d validation images\n', numel(black)) ;
    keep = setdiff(1:numel(names), black) ;
    names = names(keep) ;
    labels = labels(keep) ;
  end
end

names = strcat(['val' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels') ;

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------

ims = dir(fullfile(opts.dataDir, 'images', 'test', '*.JPG')) ;
names = sort({ims.name}) ;
labels = zeros(1, numel(names)) ;

if numel(labels) ~= 10e3
  warning('Found %d instead of 10,000 test images', numel(labels))
end

names = strcat(['test' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

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
