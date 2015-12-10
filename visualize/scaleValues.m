function output = scaleValues(img, low, high)
minVal = min(img(:));
maxVal = max(img(:));

requestedRange = high - low;

img = double(img);

output = (((img - minVal) / (maxVal - minVal)) * requestedRange) + low;
end