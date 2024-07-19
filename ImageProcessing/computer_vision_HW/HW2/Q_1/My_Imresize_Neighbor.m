function [image] = My_Imresize_Neighbor(Input_Image, Resizing_Factor)
    n = size(Input_Image, 1);
    m = size(Input_Image, 2);
    numChannels = size(Input_Image, 3);

    new_n = ceil(n * Resizing_Factor);
    new_m = ceil(m * Resizing_Factor);
    image = zeros(new_n, new_m, numChannels);

    for i = 1:new_n
        for j = 1:new_m
           i_ = max(min(round(i/Resizing_Factor), n), 1); 
           j_ = max(min(round(j/Resizing_Factor), m), 1);
           image(i, j, :) = Input_Image(i_, j_,:);
        end
    end
end