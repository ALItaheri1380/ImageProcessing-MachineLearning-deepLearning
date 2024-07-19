function [Image] = My_Resize(Input_Image,Resizing_Factor)
     tic
     p = 1.06;
     Image = zeros(size(Input_Image,1)*2,size(Input_Image,2)*2,size(Input_Image,3));

     img_1 = Resize_1(Input_Image, Resizing_Factor, p);
     img_2 = Resize_2(Input_Image);

     Image(1:2:end,1:2:end) = img_1(1:2:end,1:2:end);
    
     if size(Input_Image, 3) == 3
        img_gray1 = rgb2gray(Input_Image);
     else
        img_gray1 = Input_Image;
     end
     edges = edge(img_gray1, "canny");

    
     for i=1:size(img_1,1)
        for j=1:size(img_1,2) 
           if all([rem(i,2),rem(j,2)])
             continue;
           end
            ax = max(min(floor((i-1)/Resizing_Factor) + 1, size(Input_Image,1)), 1); ay=max(min(floor((j-1)/Resizing_Factor) + 1,size(Input_Image,2)), 1);
            bx = max(min(floor((i-1)/Resizing_Factor) + 1, size(Input_Image,1)), 1); by=max(min(ceil((j-1)/Resizing_Factor) + 1, size(Input_Image,2)), 1);
            cx = max(min(ceil((i-1)/Resizing_Factor) + 1, size(Input_Image,1)), 1); cy=max(min(floor((j-1)/Resizing_Factor) + 1, size(Input_Image,2)), 1);
            dx = max(min(ceil((i-1)/Resizing_Factor) + 1, size(Input_Image,1)), 1); dy=max(min(ceil((j-1)/Resizing_Factor) + 1, size(Input_Image,2)), 1);
            
           if any([edges(ax,ay)==1, edges(bx,by)==1, edges(cx,cy)==1, edges(dx,dy)==1])
              if all([edges(ax,ay)==1, edges(bx,by)==1, edges(cx,cy)==0, edges(dx,dy)==0])
                  Image(i,j,:) = (0.4*img_1(i,j,:) + 0.6*img_2(i,j,:));
              elseif all([edges(ax,ay)==0, edges(bx,by)==0, edges(cx,cy)==1, edges(dx,dy)==1])
                  Image(i,j,:) = (0.6*img_1(i,j,:) + 0.4*img_2(i,j,:));
              elseif all([edges(ax,ay)==0, edges(bx,by)==1, edges(cx,cy)==1, edges(dx,dy)==1])
                  Image(i,j,:) = (0.4*img_1(i,j,:) + 0.6*img_2(i,j,:));
              elseif all([edges(ax,ay)==1, edges(bx,by)==0, edges(cx,cy)==0, edges(dx,dy)==0])
                  Image(i,j,:) = (0.4*img_1(i,j,:) + 0.6*img_2(i,j,:));
              elseif all([edges(ax,ay)==1, edges(bx,by)==0, edges(cx,cy)==1, edges(dx,dy)==1])
                  Image(i,j,:) = (0.25*img_1(i,j,:) + 0.75*img_2(i,j,:));
              elseif all([edges(ax,ay)==0, edges(bx,by)==1, edges(cx,cy)==0, edges(dx,dy)==0])
                  Image(i,j,:) = (0.45*img_1(i,j,:) + 0.55*img_2(i,j,:));
              elseif all([edges(ax,ay)==1, edges(bx,by)==1, edges(cx,cy)==0, edges(dx,dy)==1])
                  Image(i,j,:) = (0.25*img_1(i,j,:) + 0.75*img_2(i,j,:));
              elseif all([edges(ax,ay)==0, edges(bx,by)==0, edges(cx,cy)==1, edges(dx,dy)==0])
                  Image(i,j,:) = (0.5*img_1(i,j,:) + 0.5*img_2(i,j,:));
              elseif all([edges(ax,ay)==1, edges(bx,by)==1, edges(cx,cy)==1, edges(dx,dy)==0])
                  Image(i,j,:) = (0.3*img_1(i,j,:) + 0.7*img_2(i,j,:));
              else
                  Image(i,j,:) = (0.15*img_1(i,j,:) + 0.85*img_2(i,j,:));
              end
           else
              Image(i,j,:) = (0.6*img_1(i,j,:) + 0.4*img_2(i,j,:));
           end 
       end 
     end
     toc
end