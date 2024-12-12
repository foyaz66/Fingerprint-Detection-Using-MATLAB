
clc; clear; close all;
disp('Select the reference fingerprint image:');
[refFile, refPath] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files'});
if isequal(refFile, 0)
    disp('No reference image selected!');
    return;
end
refImg = imread(fullfile(refPath, refFile));

disp('Select the test fingerprint image to match:');
[testFile, testPath] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files'});
if isequal(testFile, 0)
    disp('No test image selected!');
    return;
end
testImg = imread(fullfile(testPath, testFile));

refImgGray = preprocessImage(refImg);
testImgGray = preprocessImage(testImg);

cornersRef = corner(refImgGray, 'Harris');
cornersTest = corner(testImgGray, 'Harris');

patchSize = 11; 
featuresRef = extractFeaturesFromCorners(refImgGray, cornersRef, patchSize);
featuresTest = extractFeaturesFromCorners(testImgGray, cornersTest, patchSize);

[indexPairs, matchedCornersRef, matchedCornersTest] = matchFeaturesCustom(cornersRef, cornersTest, featuresRef, featuresTest);

figure;
imshowpair(refImgGray, testImgGray, 'montage');
hold on;
plot(matchedCornersRef(:, 1), matchedCornersRef(:, 2), 'r*');
plot(matchedCornersTest(:, 1) + size(refImgGray, 2), matchedCornersTest(:, 2), 'g*');
title('Matched Features Between Reference and Test Fingerprints');

similarity = size(indexPairs, 1) / min(size(featuresRef, 1), size(featuresTest, 1)) * 100;
disp(['Similarity score: ', num2str(similarity), '%']);
if similarity > 50
    disp('The fingerprints match!');
else
    disp('The fingerprints do not match!');
end


function processedImg = preprocessImage(img)
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = adapthisteq(img); 
    processedImg = edge(img, 'Canny'); 
end

function features = extractFeaturesFromCorners(img, corners, patchSize)
    halfSize = floor(patchSize / 2);
    [rows, cols] = size(img);
    numCorners = size(corners, 1);
    features = zeros(numCorners, patchSize^2);
    
    for i = 1:numCorners
        x = corners(i, 1);
        y = corners(i, 2);
        if x > halfSize && y > halfSize && x <= (cols - halfSize) && y <= (rows - halfSize)
        patch = img((y-halfSize):(y+halfSize), (x-halfSize):(x+halfSize));
        features(i, :) = patch(:)';
        else
        features(i, :) = NaN; 
        end
    end
    
    features = features(~any(isnan(features), 2), :);
end

function [indexPairs, matchedCorners1, matchedCorners2] = 
matchFeaturesCustom(corners1, corners2, features1, features2)
    numFeatures1 = size(features1, 1);
    numFeatures2 = size(features2, 1);
    distances = zeros(numFeatures1, numFeatures2);
    
    for i = 1:numFeatures1
    for j = 1:numFeatures2
            distances(i, j) = norm(features1(i, :) - features2(j, :));
        end
    end
    
    [minDist, minIndex] = min(distances, [], 2);
    threshold = mean(minDist) + std(minDist); 
    validMatches = minDist < threshold;
    
    indexPairs = [(1:numFeatures1)' minIndex];
    indexPairs = indexPairs(validMatches, :);
    matchedCorners1 = corners1(indexPairs(:, 1), :);
    matchedCorners2 = corners2(indexPairs(:, 2), :);
end
