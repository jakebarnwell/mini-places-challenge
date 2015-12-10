function out = visualize_image_response(net, x, y, im)
addpath('subaxis');
im = single(im);
weights = net.layers{1}.weights;
filter_weights = weights{1};
biases = weights{2};

clr = [0.1 0.1 0.1; 1 1 1];

n = 1;
figure; hold on;
for ix=1:x
    for iy=1:y
        if n <= size(filter_weights, 4)
            subaxis(y,x,n, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
            
            convd = zeros(size(im));
            for i=1:size(filter_weights, 3)
                f = filter_weights(:,:,i,(ix-1)*x+iy);
                convd(:,:,i) =  conv2(im(:,:,i), f, 'same');
            end
            response = sum(convd, 3) > biases((ix-1)*x+iy);
            imagesc(response);
            colormap(clr);
            axis tight; axis off;
            n = n + 1;
        end
    end
end
