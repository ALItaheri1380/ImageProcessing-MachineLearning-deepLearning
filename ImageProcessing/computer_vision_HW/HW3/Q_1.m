clc, clear, close all;

pnt = dir('Q_1\'); 

for k=1:numel(pnt)
    if(pnt(k).isdir) 
        continue;
    end

    score = [];

    Image_name = pnt(k).name;
    Image_path = fullfile('Q_1', Image_name);  
    
    Image = im2double(imread(Image_path));
    noise_value = [0.3, 0.5, 0.7, 0.9];

    for i=1:numel(noise_value)
        noisy_image = imnoise(Image,'salt & pepper',noise_value(i));
        
        Image_1 = My_method_for_Denoise_Image(noisy_image);
        Image_2 = My_method_for_Denoise_Image_2(noisy_image);

        median_method = medfilt2(noisy_image, [8 8]);
        
        Result = (Image_1 + Image_2)/2;
        original_image = im2double(imread(Image_path));
        disp("Amount of noise is = "+num2str(noise_value(i)*100 + "%"));
        disp("psnr of my method in the picture of "+ Image_name(1:find(Image_name == '.') - 1)+ ...
            " is = "+ num2str(psnr(Result,original_image)) + ', and in medfilt2 method is = ' + num2str(psnr(median_method, original_image)));
        disp(' ');
    end
    disp("#############################################");
end

function Image = My_method_for_Denoise_Image(noisy_image)
    noise_pixels = sum(noisy_image(:) == 1 | noisy_image(:) == 0);
    noise_ratio = noise_pixels / numel(noisy_image);
    if noise_ratio < 0.33
        kernel_size = 3;
    elseif noise_ratio < 0.73
        kernel_size = 4;
    else
        kernel_size = 5;
    end

    r_p = floor(kernel_size / 2);
    padded_image = padarray(noisy_image, [r_p, r_p], "replicate");

    for i = 1+r_p : size(padded_image, 1)-r_p
        for j = 1+r_p : size(padded_image, 2)-r_p
            if padded_image(i, j) == 0 || padded_image(i, j) == 1
                kernel_pixels = [];
                distances = [];
                index = 1;
                for i_ = -r_p:r_p
                    for j_ = -r_p:r_p
                        k_x = i + i_;
                        k_y = j + j_;
                        if padded_image(k_x, k_y) ~= 0 && padded_image(k_x, k_y) ~= 1
                            kernel_pixels(index) = padded_image(k_x, k_y);
                            distances(index) = sqrt(i_^2 + j_^2);
                            index = index + 1;
                        end
                    end
                end
                one_matrix = ones(1,numel(distances));
                weights = one_matrix ./ (distances + eps);
                weights = weights / sum(weights);

                result = sum(weights .* kernel_pixels);
                padded_image(i, j) = result;
            end
        end
    end
    Image = padded_image(1+r_p:end-r_p, 1+r_p:end-r_p);
end


function Image = My_method_for_Denoise_Image_2(noisy_image)
    noise_pixels = sum(noisy_image(:) == 1 | noisy_image(:) == 0);
    noise_ratio = noise_pixels / numel(noisy_image);

    if noise_ratio < 0.33
        kernel_size = 3;
    elseif noise_ratio < 0.53
        kernel_size = 4;
    elseif noise_ratio <0.73
        kernel_size = 5;
    else
        kernel_size = 8;
    end

    r_p = floor(kernel_size / 2);
    padded_image = padarray(noisy_image, [r_p r_p], "replicate");

    Image = padded_image;
    for i = 1+r_p : size(padded_image, 1)-r_p
        for j = 1+r_p : size(padded_image, 2)-r_p
            if padded_image(i, j) == 0 || padded_image(i, j) == 1
                neighbors = padded_image(i-r_p:i+r_p, j-r_p:j+r_p);
                valid_neighbors = neighbors(neighbors ~= 0 & neighbors ~= 1);
                
                if isempty(valid_neighbors)
                    Image(i, j) = padded_image(i, j);
                else
                    Image(i, j) = mean(valid_neighbors(:));
                end
            end
        end
    end

    Image = Image(1+r_p:end-r_p, 1+r_p:end-r_p);
end