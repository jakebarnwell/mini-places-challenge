function out = visualize_layers(net, x, y)
addpath('subaxis');
weights = net.layers{1}.weights;
filter_weights = weights{1};

bias = 0.4;
n = 1;
figure; hold on;
for ix=1:x
    for iy=1:y
        if n <= size(filter_weights, 4)
            subaxis(y,x,n, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
            imagesc(bias + filter_weights(:,:,:,(ix-1)*x+iy));
            axis tight; axis off;
            n = n + 1;
        end
    end
end

