I = rgb2gray(imread('clock.jpg'));
I = im2double(I);
imshow(I)

%Find the edges in the image using the edge function. 
BW = edge(I,'sobel');
imshow(BW);

%Compute the Hough transform of the binary image returned by edge.
[H,theta,rho] = hough(BW);

%Display the transform, H, returned by the hough function.
figure;
imshow(imadjust(rescale(H)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)

%Find the peaks in the Hough transform matrix, H, using the houghpeaks function.
P = houghpeaks(H,40,'threshold',ceil(0.1*max(H(:))));

%Superimpose a plot on the image of the transform that identifies the peaks.
x = theta(P(:,2));
y = rho(P(:,1));
plot(x,y,'s','color','black');

%%
%Find lines in the image using the houghlines function. 
lines = houghlines(BW,theta,rho,P,'FillGap',5,'MinLength',10);

%Create a plot that displays the original image with the lines superimposed on it.
figure, imshow("clock.jpg"), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','white');

%    % Plot beginnings and ends of lines
%    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','white');