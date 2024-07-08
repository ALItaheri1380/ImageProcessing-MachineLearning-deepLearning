clc; clear; close all;


Test_1 = im2double(imread("Test_1.jpg")); % 300 × 300
Test_2 = im2double(imread("Test_2.jpg"));% 689 × 689
Test_3 = im2double(imread("Test_3.jpg"));% 1036 × 1036

Resizing_Factor_1 = 0.435; % 300/689
Resizing_Factor_2 = 1.503; % 1036/689

img1 = My_Imresize_Neighbor(Test_2, Resizing_Factor_1);
figure;
imshow(img1,[]);
title('Test 1 Resized by 0.435 using nearest neighbor');

img2 = My_Imresize_Neighbor(Test_2, Resizing_Factor_2);
figure;
imshow(img2,[]);
title('Test 1 Resized by 1.503 using nearest neighbor');

disp(psnr(Test_1,img1));
disp(psnr(Test_3,img2));