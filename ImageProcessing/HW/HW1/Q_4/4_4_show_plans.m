clc, clear, close all;

I = imread("Stego_image.png");
Bit_1 = bitget(I, 1); Bit_2 = bitget(I, 2); Bit_3 = bitget(I, 3); Bit_4 = bitget(I, 4);
Bit_5 = bitget(I, 5); Bit_6 = bitget(I, 6); Bit_7 = bitget(I, 7); Bit_8 = bitget(I, 8);

figure;
subplot(4, 2, 1);
imshow(Bit_1, []);
xlabel('Bit 1');

subplot(4, 2, 2);
imshow(Bit_2, []);
xlabel('Bit 2');

subplot(4, 2, 3);
imshow(Bit_3, []);
xlabel('Bit 3');

subplot(4, 2, 4);
imshow(Bit_4, []);
xlabel('Bit 4');

subplot(4, 2, 5);
imshow(Bit_5, []);
xlabel('Bit 5');

subplot(4, 2, 6);
imshow(Bit_6, []);
xlabel('Bit 6');

subplot(4, 2, 7);
imshow(Bit_7, []);
xlabel('Bit 7');

subplot(4, 2, 8);
imshow(Bit_8, []);
xlabel('Bit 8');

saveas(gcf, 'plans.png');