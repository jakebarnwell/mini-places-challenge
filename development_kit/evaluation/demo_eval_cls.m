%this script demos the usage of evaluation routines
% the result file 'demo.val.pred.txt' on validation data is evaluated
% against the ground truth

fprintf('MINI PLACES SCENE CLASSIFICATION CHALLENGE\n');

pred_file='../../model_averaging_results/jakenet_jamar7_jakenet-aug4flip_jakenet-aug8flip-refnet3_alexnet1_jakenet-flipNoise/val.txt'; %nets/jakenet-flipNoise/val-predictions-21.txt
ground_truth_file='../data/val.txt';
num_predictions_per_image=5;

fprintf('pred_file: %s\n', pred_file);
fprintf('ground_truth_file: %s\n', ground_truth_file);

error_cls = eval_cls(pred_file,ground_truth_file,1:num_predictions_per_image);

disp('# guesses vs cls error');
disp([(1:num_predictions_per_image)',error_cls']);

disp('# guesses vs cls accuracy');
disp([(1:num_predictions_per_image)',1-error_cls']);


