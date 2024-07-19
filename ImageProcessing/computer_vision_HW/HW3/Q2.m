clc, clear, close all;

Imags = dir('Q_2\');
ptrn_Dgt = 'Ptrn_Digt\';
Result_Dgt = 'Result_Digt\';
Sum_Dgt = 'Sum\';

correct = 0;

for num=1:numel(Imags)
    if(Imags(num).isdir) 
        continue;
    end
    Result_Sum = 0;
    Image_name = Imags(num).name;
    Image_path = fullfile('Q_2', Image_name);
    original_Image = im2double(imread(Image_path));
    gray_Image = im2gray(original_Image);
    Denoised_Image = My_Denoise(gray_Image);
    Copy_Denoised_Image = Denoised_Image;
    x1 = -1; x2 = -1; y1 = -1; y2 = -1;

    for i=1:size(Copy_Denoised_Image,1)
        for j=1:size(Denoised_Image,2)
            if Copy_Denoised_Image(i,j) ~= 1
                x1 = i; x2 = i; y1 = j; y2 = j;

                Stack = [];
                Stack(end+1)=i;
                Stack(end+1)=j;
                
                while ~isempty(Stack)
                    y = Stack(end);
                    Stack = Stack(1:end-1);

                    x = Stack(end);
                    Stack = Stack(1:end-1);

                    x1 = min(x,x1); x2 = max(x,x2);
                    y1 = min(y,y1); y2 = max(y,y2);
                    
                    Copy_Denoised_Image(x,y) = 1;
                    if x>1 && Copy_Denoised_Image(x-1,y) ~= 1
                        Stack(end+1) = x-1; Stack(end+1) = y;
                    end
                    if x<size(Copy_Denoised_Image,1) && Copy_Denoised_Image(x+1,y) ~= 1
                        Stack(end+1) = x+1; Stack(end+1) = y;
                    end
                    if y>1 && Copy_Denoised_Image(x,y-1) ~= 1
                        Stack(end+1) = x; Stack(end+1) = y-1;
                    end
                    if y<size(Copy_Denoised_Image,2) && Copy_Denoised_Image(x,y+1) ~= 1
                        Stack(end+1) = x; Stack(end+1) = y+1;
                    end
                end

                if abs(x1-x2) < 3 || abs(y1-y2) < 3
                   break;
                end
                Digit_crop = Denoised_Image(x1:x2,y1:y2);
                value = -'inf'; index = -1; cntr = 1;
                
                dir_ptrn_Dgt = dir(ptrn_Dgt);                
                for pt=1: numel(dir_ptrn_Dgt)
                   if(dir_ptrn_Dgt(pt).isdir) 
                       continue;
                   end
                   Image_p_name = dir_ptrn_Dgt(pt).name;
                   Image_p_path = fullfile('Ptrn_Digt', Image_p_name);
                   
                   ptrn_Image = im2double(imread(Image_p_path));
                   ptrn_Image = im2gray(ptrn_Image);
                   
                   temp = psnr(ptrn_Image,imresize(Digit_crop,[size(ptrn_Image)]));
                   if temp > value
                       value = temp;
                       index = cntr;
                   end
                   cntr = cntr + 1;
                end
                
                original_crop = original_Image(x1:x2,y1:y2,:);
                red_chanel = original_crop(:,:,1);
                blue_chanel = original_crop(:,:,3);

                if sum(red_chanel(:)) >= sum(blue_chanel(:))
                    sign = 0;
                else
                    sign = 1;
                end

                if sign == 1
                    Result_Sum = Result_Sum + -1*index;
                else
                    Result_Sum = Result_Sum + index;
                end
            end
        end
    end
    Underline = strfind(Imags(num).name,'_');
    dot = strfind(Imags(num).name,'.');
    real = str2double(Imags(num).name(Underline(2) + 1:dot - 1));
    
    if real == Result_Sum
        correct = correct + 1;
        disp("The prediction for Image " + num2str(correct) + " was correct");
    else
        disp("The prediction for Image " + num2str(correct) + " was not correct...!!!");
    end

    if Result_Sum < 0

       res = imresize(imread("Result_Digt\minus.png"),[45 45]);
       if abs(Result_Sum) >= 10
           dig_1 = floor(abs(Result_Sum)/10);
           dig_2 = floor(mod(abs(Result_Sum),10));

           im_1 = convert_dig_to_Gr_image(dig_1);
           im_2 = convert_dig_to_Gr_image(dig_2);

           res = [res im_1 im_2];
       else
           dig = abs(Result_Sum);

           im = convert_dig_to_Gr_image(dig);
           res = [res im];
       end

    elseif Result_Sum == 0
        res = convert_dig_to_Gr_image(0);
    else
        if Result_Sum >= 10
           dig_1 = floor(Result_Sum/10);
           dig_2 = floor(mod(Result_Sum,10));

           im_1 = convert_dig_to_Gr_image(dig_1);
           im_2 = convert_dig_to_Gr_image(dig_2);

           res = [im_1 im_2];
        else
           res = convert_dig_to_Gr_image(Result_Sum);
        end
    end
    
    original_Image(756:800, 300:300+size(res,2)-1, :) = im2double(res);
    imwrite(original_Image, [Sum_Dgt 'Result' Image_name], 'jpg');

    
end

disp("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
disp("Accuracy of my method is = " + num2str(correct) + "%");

function [Image_of_Digit] = convert_dig_to_Gr_image(dig)
    dir_Result_Dgt = dir('Result_Digt\');
    Image_name = dir_Result_Dgt(dig+3).name;
    Image_path = fullfile('Result_Digt',Image_name);
    Image_of_Digit = imresize(imread(Image_path),[45 45]);
end

function Image = My_Denoise(noisy_image)
    kernel_size = 5;
    r_p = floor(kernel_size / 2);
    padded_image = padarray(noisy_image, [r_p r_p], "replicate");

    Image = padded_image;
    for i = 1+r_p : size(padded_image, 1)-r_p
        for j = 1+r_p : size(padded_image, 2)-r_p
            if padded_image(i, j) == 0
                neighbors = padded_image(i-r_p:i+r_p, j-r_p:j+r_p);
                valid_neighbors = neighbors(neighbors ~= 0);
                Image(i, j) = median(valid_neighbors(:));
            end
        end
    end

    Image = Image(1+r_p:end-r_p, 1+r_p:end-r_p);
end