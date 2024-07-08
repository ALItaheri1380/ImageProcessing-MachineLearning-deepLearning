clc, clear, close all;

File_Name = input("FileName: ", "s");
Plot_Image(File_Name);

function Plot_Image(f_name)

    F = fopen(f_name);
    if F == -1
        error('Could not open file.');
    end

    magic_number = fscanf(F, '%s', 1);
    width = fscanf(F, '%d', 1);
    height = fscanf(F, '%d', 1);
    
    fclose(F);
    
    file = fopen(f_name);
    fContent = fread(file);
   
    if strcmp(magic_number, "P6")
        plot_ppm(fContent , width , height);

    else
        error('Invalid file format.');
    end
  
    fclose(file);
end

function plot_ppm(fContent , width , height)
    fContent = fContent(length(fContent) - (width * height * 3) + 1 : end);
    Image = zeros(height, width, 3, 'uint8');
        
    k = 1;
    for i = 1:height
       for j = 1:width
           for x = 1 : 3
              Image(i, j, x) = fContent(k);
              k = k + 1;
           end
       end
    end
    pic = imread("Q_1.ppm");
    imshow([Image  pic] , []);

    disp("psnr is = " + psnr(pic , Image));
end



