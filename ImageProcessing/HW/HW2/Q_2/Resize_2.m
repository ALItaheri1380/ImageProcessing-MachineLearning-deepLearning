function [Image] = Resize_2(I)
    if size(I, 3) == 3
        img_gray = rgb2gray(I);
    else
        img_gray = I;
    end
    
    [X, Y] = meshgrid(1:size(img_gray, 2), 1:size(img_gray, 1));
    [Xq, Yq] = meshgrid(1.5:0.5:size(img_gray, 2), 1.5:0.5:size(img_gray, 1));
    Image = interp2(X, Y, img_gray, Xq, Yq, 'spline');
    Image = padarray(Image, [1, 1], 'replicate');
    
    if size(I, 3) == 3
        Image = cat(3,Image,Image,Image);
    end
end