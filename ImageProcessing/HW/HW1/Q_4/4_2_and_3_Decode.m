clc , clear , close all;

load key
rng(key);

stegoimage = imread("Stego_image.png");

decodeimage(stegoimage, key);

function decodeimage(stegoimage, key)

    bit_1 = bitget(stegoimage , 1);
    bit_1 = bit_1(:);
    
    Random_Pattern = zeros(1,numel(stegoimage(:)));
    Random_Pattern(1:end) = round(rand(1,numel(stegoimage(:))));
    
    Random_pat_size = Random_Pattern(:);
    rsize = Random_pat_size(1:16);
    img_size = bit2int(xor(bit_1(1:16) , rsize) , 16);
    Random_Pattern(img_size*8 + 17 : end) = 0;
    Random_Pattern = reshape(Random_Pattern , size(stegoimage));
    
    scrt = xor(Random_Pattern , bitget(stegoimage , 1));
    scrt = scrt(:);
    
    scrt2 = scrt(17:img_size*8 + 16);
    secret_Image = bit2int(scrt2 , 8);
    
    outfile = 'secret_massage.jpg';
    
    [fid, msg] = fopen(outfile, 'wb');
    if fid < 0
        error('Failed to create file "%s" because "%s"', outfile, msg);
    end
    
    fwrite(fid, secret_Image, 'uint8');
    fclose(fid);
    
    figure();
    imshow(outfile);
    text(160, 0, ['PSNR: ', num2str(psnr(imread('secret_massage.jpg') , imread("IUT.jpg")))], 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');
end