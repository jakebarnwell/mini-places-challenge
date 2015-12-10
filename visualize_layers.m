out = function visualize_layers(net)
weights = net.layers{1}.weights;
filter_weights = weights{1};
biases = weights{2};

% Why not just do imagesc(filter_weights(:,:,:,i)) ?

out = zeros(size(filter_weights));
for f=1:size(filter_weights, 4)
	for k=1:size(filter_weights, 3)
		kernel = filter_weights(:,:,k,f))
		out(
