clc , clear , close all;

coverImage = imread("Cover_Image.png"); % size of this pic is 1080 * 1920

F = fopen("IUT.jpg");
secretImage = fread(F);
fclose(F);
if F == -1
   error('Could not open file.');
end

encodeImage(coverImage , secretImage);

function encodeImage(coverImage, secretImage)
    key = rng;
    
    Random_Pattern = zeros(1,numel(coverImage(:)));
    Random_Pattern(1:size(secretImage,1)*8 + 16) = round(rand(1,size(secretImage,1)*8 + 16));
    Random_Pattern = reshape(Random_Pattern , size(coverImage));

    temp_cover = zeros(1,numel(coverImage(:)));
    temp_cover(1:16) = int2bit(size(secretImage , 1) , 16); 
    temp_cover(17:numel(secretImage(:))*8 + 16) = int2bit(secretImage(1:end) , 8);

    bit_1 = bitget(coverImage , 1);

    temp_cover(numel(secretImage(:))*8 + 17 : end) = bit_1(numel(secretImage(:))*8 + 17:end);
    temp_cover = reshape(temp_cover , size(coverImage));
    
    stegoimage = xor(temp_cover,Random_Pattern);
    stegoimage = bitset(coverImage , 1 , stegoimage);
    
    imshow([coverImage  stegoimage] , [])
    text(1650, -50, ['PSNR: ', num2str(psnr(stegoimage , coverImage))], 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');
        
    imwrite(stegoimage ,"Stego_image.png")
    save("key","key");
end