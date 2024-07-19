clc, clear, close all;

Resizing_Factor = 2;

Test_1 = im2double(imread("LR_Boat.png"));
Result_1 = im2double(imread("Boat.png"));

Test_2 = im2double(imread("LR_Cameraman.png"));
Result_2 = im2double(imread("Cameraman.png"));

Test_3 = im2double(imread("LR_Peppers.png"));
Result_3 = im2double(imread("Peppers.png"));

Test_4 = im2double(imread("LR_House.png"));
Result_4 = im2double(imread("House.png"));

out_1 = My_Resize(Test_1,2);
out_2 = My_Resize(Test_2,2);
out_3 = My_Resize(Test_3,2);
out_4 = My_Resize(Test_4,2);


disp("psnr of Boat = " + num2str(psnr(out_1,Result_1)));
disp("psnr of Cameraman = " + num2str(psnr(out_2,Result_2)));
disp("psnr of Peppers = " + num2str(psnr(out_3,Result_3)));
disp("psnr of House = " + num2str(psnr(out_4,Result_4)));
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp("average psnr = " + num2str((psnr(out_1,Result_1) + psnr(out_2,Result_2) + psnr(out_3,Result_3) + psnr(out_4,Result_4)) / 4))