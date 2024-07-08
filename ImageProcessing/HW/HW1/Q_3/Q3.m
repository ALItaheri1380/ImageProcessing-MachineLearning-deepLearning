clc , clear , close all;

Test_1 = im2double(imread("Test_1.ppm"));
Test_2 = im2double(imread("Test_2.ppm"));
Test_3 = im2double(imread("Test_3.ppm"));

figure;
subplot(1,3,1);
imshow(Test_1);

subplot(1,3,2);
imshow(Test_2);

subplot(1,3,3);
imshow(Test_3);

[PSNR_MY_METHOD , PSNR_RGB2GRAY] = cl_psnr(Test_1 , 1);
disp("psnr value of image 1(TEST_1) in my own method = " + PSNR_MY_METHOD + ", PSNR value of image 1(TEST_1) in rgh2gray method = " + PSNR_RGB2GRAY);

[PSNR_MY_METHOD , PSNR_RGB2GRAY] = cl_psnr(Test_2 , 2);
disp("psnr value of image 2(TEST_2) in my own method = " + PSNR_MY_METHOD + ", PSNR value of image 2(TEST_2) in rgh2gray method = " + PSNR_RGB2GRAY);

[PSNR_MY_METHOD , PSNR_RGB2GRAY] = cl_psnr(Test_3 , 3);
disp("psnr value of image 3(TEST_3) in my own method = " + PSNR_MY_METHOD + ", PSNR value of image 3(TEST_3) in rgh2gray method = " + PSNR_RGB2GRAY);

function [PSNR_MY_METHOD , PSNR_RGB2GRAY] = cl_psnr(img , cnt)
    R = img(: , : , 1);
    G = img(: , : , 2);
    B = img(: , : , 3);

    I = 1/3  * R + 1/3 * G + 1/3 * B;
    PSNR_MY_METHOD = psnr(cat(3 , I , I , I) , img);
    PSNR_RGB2GRAY = psnr(cat(3 , rgb2gray(img) , rgb2gray(img) , rgb2gray(img)) , img);

    figure;
    subplot(1,2,1);
    imshow(I , []);

    subplot(1,2,2);
    imshow(rgb2gray(img) ,[]);
    text(-1500,-40,sprintf("psnr value of image in my own method = %d, PSNR value of image in rgh2gray method = %d" , PSNR_MY_METHOD , PSNR_RGB2GRAY) ,'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');

    imwrite(I, sprintf("my_method_result%d.tif", cnt)); 
    imwrite(rgb2gray(img) , sprintf("rgb2gray_result%d.tif", cnt))
    imwrite(img , sprintf("original_image%d.tif", cnt))
end