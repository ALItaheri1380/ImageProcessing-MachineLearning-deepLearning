clc, clear, close all; 

Original_Image = imread('Output.tif'); 
imageWidth = size(Original_Image, 1); 
imageHeight = size(Original_Image, 2); 

cornerFiles = dir('Corner_*.tif'); 
if isempty(cornerFiles) 
    error('No corner files found'); 
end 

sampleCorner = imread(cornerFiles(1).name); 
pieceWidth = size(sampleCorner, 1); 
pieceHeight = size(sampleCorner, 2); 

patchFiles = dir('Patch_*.tif'); 
patch_Images = cell(1, length(patchFiles)); 
for i = 1:length(patchFiles) 
    patch_Images{i} = imread(patchFiles(i).name); 
end 

for row = 0:imageWidth/pieceWidth-1 
    for col = 0:imageHeight/pieceHeight-1 
        mxH = (col + 1) * pieceHeight; 
        mnH = mxH - pieceHeight + 1; 
        
        if row == 0 
            cur_Image = Original_Image(1:pieceWidth, mnH:mxH, :); 
        elseif row == imageWidth/pieceWidth-1 
            cur_Image = Original_Image(imageWidth - pieceWidth + 1:imageWidth, mnH:mxH, :); 
        else 
            cur_Image = Original_Image(row*pieceWidth+1:(row+1)*pieceWidth, mnH:mxH, :); 
        end 

        if any(cur_Image(:) ~= 0) 
            continue; 
        end 

        if row == 0 
            pr_Image = Original_Image(1:pieceHeight, mnH - pieceWidth:mnH - 1, :); 
        elseif col == 0 
            pr_Image = Original_Image((row-1)*pieceWidth+1:row*pieceWidth, 1:pieceHeight, :); 
        else 
            pr_Image = Original_Image(row*pieceWidth+1:(row+1)*pieceWidth, mnH - pieceHeight:mnH - 1, :); 
            adjacentImage = Original_Image((row-1)*pieceWidth+1:row*pieceWidth, mnH:mxH, :); 
        end 

        maxPSNR = -inf; 
        bestPatchIndex = NaN; 

        for k = 1:length(patch_Images) 
            currentPatch = patch_Images{k}; 

            if row == 0 
                psnrValue = psnr(currentPatch(:, 1, :), pr_Image(:, end, :)); 
            elseif col == 0 
                psnrValue = psnr(pr_Image(end, :, :), currentPatch(1, :, :)); 
            else 
                psnrValue = max(psnr(pr_Image(:, end, :), currentPatch(:, 1, :)), ... 
                              psnr(adjacentImage(end, :, :), currentPatch(1, :, :))); 
            end 

            if psnrValue > maxPSNR 
                maxPSNR = psnrValue; 
                bestPatchIndex = k; 
            end 
        end 

        if ~isnan(bestPatchIndex) 
            if row == 0 
                Original_Image(1:pieceWidth, mnH:mxH, :) = patch_Images{bestPatchIndex}; 
            elseif row == imageWidth/pieceWidth-1 
                Original_Image(imageWidth - pieceWidth + 1:imageWidth, mnH:mxH, :) = patch_Images{bestPatchIndex}; 
            else 
                Original_Image(row*pieceWidth+1:(row+1)*pieceWidth, mnH:mxH, :) = patch_Images{bestPatchIndex}; 
            end 
            patch_Images(bestPatchIndex) = []; 
            imshow(Original_Image); 
        end 
    end 
end