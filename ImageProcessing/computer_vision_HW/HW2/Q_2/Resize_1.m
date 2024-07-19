function [Image] = Resize_1(I, Resizing_Factor, p)
    n = size(I, 1);
    m = size(I, 2);

    numChannels = size(I, 3);
    new_n = ceil(n * Resizing_Factor);
    new_m = ceil(m * Resizing_Factor);

    Image = zeros(new_n, new_m, numChannels);

    Image(1:2:end,1:2:end) = I(:,:);

    if size(I, 3) == 3
        img_gray = rgb2gray(I);
    else
        img_gray = I;
    end
    edges = edge(img_gray, "sobel");
    
    for i=1:new_n
        for j=1:new_m   
            if all([rem(i,2)==1,rem(j,2)==1])
                continue;
            end 
            ax = max(min(floor((i-1)/Resizing_Factor) + 1, n), 1); ay=max(min(floor((j-1)/Resizing_Factor) + 1,m), 1);
            bx = max(min(floor((i-1)/Resizing_Factor) + 1, n), 1); by=max(min(ceil((j-1)/Resizing_Factor) + 1, m), 1);
            cx = max(min(ceil((i-1)/Resizing_Factor) + 1, n), 1); cy=max(min(floor((j-1)/Resizing_Factor) + 1, m), 1);
            dx = max(min(ceil((i-1)/Resizing_Factor) + 1, n), 1); dy=max(min(ceil((j-1)/Resizing_Factor) + 1, m), 1);
    
            a = I(ax,ay , :);
            b = I(bx,by , :);
            c = I(cx, cy, :);
            d = I(dx,dy , :);
    
            x=((i-1)/Resizing_Factor) - floor((i-1)/Resizing_Factor);
            y=((j-1)/Resizing_Factor) - floor((j-1)/Resizing_Factor);  
                            
            ca = 1-(abs(x)^p + abs(y)^p)^(1/p);
            cb = 1-(abs(x)^p + abs(1-y)^p)^(1/p);
            cc = 1-(abs(1-x)^p + abs(y)^p)^(1/p);
            cd = 1-(abs(1-x)^p + abs(1-y)^p)^(1/p);
            
            if any([edges(ax,ay)==1, edges(bx,by)==1, edges(cx,cy)==1, edges(dx,dy)==1])
                if rem(i,2)==1
                    if any([all([edges(ax,ay)==1,edges(bx,by)==0])])
                        if(a-b)>0.3
                           Image(i,j,:) = b*1.56;
                        elseif (b-a)>0.3
                            Image(i,j,:) = (1.3*ca*a + cb*b) /(1.3*ca+cb); 
                        else
                           Image(i,j,:) = (0.9*ca*a + cb*b) /(0.9*ca+cb); 
                        end
                    elseif any([all([edges(ax,ay)==0,edges(bx,by)==1])])
                        if(a-b)>0.4
                            Image(i,j,:) = (0.9*ca*a + cb*b) /(0.9*ca+cb);
                        elseif (b-a)>0.3
                            Image(i,j,:) = (0.9*ca*a + cb*b) /(0.9*ca+cb);
                        else
                            Image(i,j,:) = (1.1*ca*a + cb*b) /(1.1*ca+cb);
                        end
                    else
                        Image(i,j,:) = (ca*a + cb*b + cc*c + cd*d) /(ca+cb+cc+cd);
                    end
                
                else
                    if all([edges(ax,ay)==1, edges(bx,by)==1, edges(cx,cy)==0, edges(dx,dy)==0])
                         Image(i,j,:) = (ca*a + 0.6*cb*b + 1.3*cc*c + cd*d) /(ca+0.6*cb+1.3*cc+cd);
                    elseif all([edges(ax,ay)==0, edges(bx,by)==0, edges(cx,cy)==1, edges(dx,dy)==1])
                         Image(i,j,:) = (1.4*ca*a + 1.1*cb*b + cc*c + cd*d) /(1.4*ca+1.1*cb+cc+cd); 
                    elseif all([edges(ax,ay)==0, edges(bx,by)==1, edges(cx,cy)==1, edges(dx,dy)==1])
                         Image(i,j,:) = (.7*ca*a + 1.5*cb*b + cc*c + .9*cd*d) /(.7*ca+1.5*cb+cc+.9*cd);
                    elseif all([edges(ax,ay)==1, edges(bx,by)==0, edges(cx,cy)==0, edges(dx,dy)==0])
                        Image(i,j,:) = (.6*ca*a + 1.1*cb*b + cc*c + cd*d) /(.6*ca+1.1*cb+cc+cd);
                    elseif all([edges(ax,ay)==1, edges(bx,by)==0, edges(cx,cy)==1, edges(dx,dy)==1])
                        Image(i,j,:) = (ca*a + cb*b + .9*cc*c + cd*d) /(ca+cb+.9*cc+cd);
                    elseif all([edges(ax,ay)==0, edges(bx,by)==1, edges(cx,cy)==0, edges(dx,dy)==0])
                        Image(i,j,:) = (1.6*ca*a + .5*cb*b + cc*c + cd*d) /(1.6*ca+.5*cb+cc+cd);
                    elseif all([edges(ax,ay)==1, edges(bx,by)==1, edges(cx,cy)==0, edges(dx,dy)==1])
                        Image(i,j,:) = (ca*a + cb*b + .9*cc*c + cd*d) /(ca+cb+.9*cc+cd);
                    elseif all([edges(ax,ay)==0, edges(bx,by)==0, edges(cx,cy)==1, edges(dx,dy)==0])
                        Image(i,j,:) = (ca*a + cb*b + .6*cc*c + cd*d) /(ca+cb+.6*cc+cd);
                    elseif all([edges(ax,ay)==1, edges(bx,by)==1, edges(cx,cy)==1, edges(dx,dy)==0])
                        Image(i,j,:) = (.9*ca*a + cb*b + 1.2*cc*c + cd*d) /(.9*ca+cb+1.2*cc+cd);
                    else
                         Image(i,j,:) = (ca*a + cb*b + cc*c + cd*d) /(ca+cb+cc+cd);
                    end
                end
            
            else
                Image(i,j,:) = (ca*a + cb*b + cc*c + cd*d) /(ca+cb+cc+cd);
            end
        end      
    end
end