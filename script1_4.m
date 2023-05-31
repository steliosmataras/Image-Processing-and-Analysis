clc; 
clear;
%matlab script for exercise 4
%divided into sections 
%1) image reading, showing and scaling
%2) original images histogram plotting
%3) log transformed image showing and histogram plotting
%4) global histogram imagfe showing and histogram plotting
%5) local histogram using a self made local histogram kernel 
%loading and preprocessing of images
a_1 = imread('dark_road_1.jpg');
a_2 = imread("dark_road_2.jpg");
a_3 = imread("dark_road_3.jpg");

% Converting the image class into "double"
b_1 = im2double(a_1);
b_2 = im2double(a_2);
b_3 = im2double(a_3);

% Converting the double  into "uint8"
b_1 = uint8(255*b_1);
b_2 = uint8(255*b_2);
b_3 = uint8(255*b_3);
% reading the image size
[M_1,N_1] = size(b_1);
[M_2,N_2] = size(b_2);
[M_3,N_3] = size(b_3);
figure;
imshow(b_1);title('original image dark_road_1');

figure;
imshow(b_2);title('original image dark_road_2');


figure;
imshow(b_3);title('original image dark_road_3');
%%
%plotting original images and respective histograms
figure(1);
subplot(2, 3, 1), imshow(b_1,[]);title('original image dark road 1');
subplot(2, 3, 2), imshow(b_2,[]);title('original image dark road 2');
subplot(2, 3, 3), imshow(b_3,[]);title('original image dark road 3');
subplot(2,3,4)
[counts,binLocations] = imhist(b_1,256);
stem(binLocations,counts,"Marker",".")
subplot(2,3,5)
[counts,binLocations] = imhist(b_2,256);
stem(binLocations,counts,"Marker",".")
subplot(2,3,6)
[counts,binLocations] = imhist(b_3,256);
stem(binLocations,counts,"Marker",".")
print(gcf, '-dpng', 'images_4/erwtima4_1_Histograms_original.png');

%%
%plotting logarithm transformed images and respective histograms
figure(1);
imshow(log(1+255*im2double(b_1)),[]);title('log transformed image 1');
print(gcf, '-dpng', 'images_4/erwtima4_logarithm_transformed_image1.png');
figure(2);
imshow(log(1+255*im2double(b_2)),[]);title('log transformed image 2');
print(gcf, '-dpng', 'images_4/erwtima4_logarithm_transformed_image2.png');
figure(3);
imshow(log(1+255*im2double(b_3)),[]);title('log transformed image  3');
print(gcf, '-dpng', 'images_4/erwtima4_logarithm_transformed_image3.png');
subplot(1,3,1)
[counts,binLocations] = imhist(uint8(log(1+255*im2double(b_1))),256);
stem(binLocations,counts,"Marker",".")
subplot(1,3,2)
[counts,binLocations] = imhist(uint8(log(1+255*im2double(b_2))),256);
stem(binLocations,counts,"Marker",".")
subplot(1,3,3)
[counts,binLocations] = imhist(uint8(log(1+255*im2double(b_3))),256);
stem(binLocations,counts,"Marker",".")
print(gcf, '-dpng', 'images_4/erwtima4_1_Histograms.png');
%%
[g_enhanced_1_withoutlog,T1] = histeq(b_1);
[g_enhanced_2_withoutlog,T2] = histeq(b_2);
[g_enhanced_3_withoutlog,T3] = histeq(b_3);
figure(10);
plot(255*T1,'blue');
hold on 
plot(255*T2,'black');
plot(255*T3,'red');
hold off 
print(gcf, '-dpng', 'images_4/transfer_functions_plot.png');
legend('image 1 transfer function','image 2 transfer function','image 3 transfer function');
title('Transfer function plot')
figure(1);
imshow(g_enhanced_1_withoutlog,[]);title('globalhist_transformed image 1');
print(gcf, '-dpng', 'images_4/erwtima4_globalhist_transformed_image1.png');
figure(2);
imshow(g_enhanced_2_withoutlog,[]);title('globalhist_transformed image 2');
print(gcf, '-dpng', 'images_4/erwtima4_globalhist_transformed_image2.png');
figure(3);
imshow(g_enhanced_3_withoutlog,[]);title('globalhist_transformed image  3');
print(gcf, '-dpng', 'images_4/erwtima4_globalhist_transformed_image3.png');
figure(4);
title(4,"Histogram plot of transformed images");
subplot(1,3,1)
[counts,binLocations] = imhist(uint8(255*im2double(g_enhanced_1_withoutlog)),256);
stem(binLocations,counts,"Marker",".")
subplot(1,3,2)
[counts,binLocations] = imhist(uint8(255*im2double(g_enhanced_2_withoutlog)),256);
stem(binLocations,counts,"Marker",".")
subplot(1,3,3)
[counts,binLocations] = imhist(uint8(255*im2double(g_enhanced_3_withoutlog)),256);
stem(binLocations,counts,"Marker",".")

print(gcf, '-dpng', 'images_4/global_histogram_equilization_withoutlog.png');

%%
%local histogram equalization
b_1 = imread('dark_road_1.jpg');
b_2 = imread("dark_road_2.jpg");
b_3 = imread("dark_road_3.jpg");

I_1 = uint8(255*im2double(b_1));
I_2 = uint8(255*im2double(b_2));
I_3 = uint8(255*im2double(b_3));

neighborhood = 75;
%fun = @(x) local(x);
%B_1 = nlfilter(b_1,[neighborhood neighborhood],fun);
[M1,N1] = size(I_1);
[M2,N2] = size(I_2);
[M3,N3] = size(I_3);
step  = floor(neighborhood/2);

L=256;%number of intensity levels
% b_2_padded  = uint8(zeros(p,q));%allocating space for padded image
% icounter=1;
% jcounter=1;
% for i = step+1:p-step
%
%     for j = step+1:q-step
%         b_2_padded(i,j) = b_2(icounter,jcounter);
%         jcounter=jcounter+1;
%     end
%     icounter=icounter+1;
%     jcounter=1;
% end
%b_2_padded = padarray(b_2,[step step],0,'both');
%b_3_padded = padarray(b_2,[step step],'replicate','both');
%Finished padding the image
I_1_dummy = zeros(M1,N1);
I_2_dummy = zeros(M2,N2);
I_3_dummy = zeros(M3,N3);
%b_3_padded_dummy = zeros(p,q);
%Locally equalizing each pixel
for i = 1:M1%step+1:p-step
    for j = 1:N1%step+1:q-step
        I_1_dummy(i,j) = round((L-1)*local_hist_kernel(I_1,i,j,neighborhood));

        %b_3_padded_dummy(i,j) = round((L-1)*local_hist_kernel(b_3_padded,i,j,neighborhood));
    end
end


for i = 1:M2%step+1:p-step
    for j = 1:N2%step+1:q-step
        I_2_dummy(i,j) = round((L-1)*local_hist_kernel(I_2,i,j,neighborhood));

        %b_3_padded_dummy(i,j) = round((L-1)*local_hist_kernel(b_3_padded,i,j,neighborhood));
    end
end
for i = 1:M3%step+1:p-step
    for j = 1:N3%step+1:q-step
        I_3_dummy(i,j) = round((L-1)*local_hist_kernel(I_3,i,j,neighborhood));

        %b_3_padded_dummy(i,j) = round((L-1)*local_hist_kernel(b_3_padded,i,j,neighborhood));
    end
end
% B_2 = b_2_padded_dummy(step+1:p-step,step+1:q-step);%dropping all zeros from the padded image
% B_3 = b_3_padded_dummy(step+1:p-step,step+1:q-step);

figure;
subplot(2,3,1)
imshow(I_1_dummy,[]);title('original image dark_road_1 locally equalized');
print(gcf, '-dpng', 'images_4/erwtima4_1_Histograms_locallyequalized.png');
subplot(2,3,2)
imshow(I_2_dummy,[]);title('original image dark_road_2 locally equalized');
print(gcf, '-dpng', 'images_4/erwtima4_2_Histograms_locallyequalized.png');
subplot(2,3,3)
imshow(I_3_dummy,[]);title('original image dark_road_3 locally equalized');
print(gcf, '-dpng', 'images_4/erwtima4_3_Histograms_locallyequalized.png');



subplot(2,3,4)
imhist(uint8(I_1_dummy));
subplot(2,3,5)
imhist(uint8(I_2_dummy));
subplot(2,3,6)
imhist(uint8(I_3_dummy));

print(gcf, '-dpng', 'images_4/erwtima4_3_Histograms_locallyequalized.png');
%subplot(2,3,2)
%imhist(B_2);
%subplot(2,3,3)
%imhist(B_3);

%showing images after local histogram equalization

%%
figure;
imhist(uint8(I_1_dummy));
print(gcf, '-dpng', 'images_4/erwtima4_1_locallyequalized_histogram.png');
figure;
imhist(uint8(I_2_dummy));
print(gcf, '-dpng', 'images_4/erwtima4_2_locallyequalized_histogram.png');

figure;
imhist(uint8(I_3_dummy));
print(gcf, '-dpng', 'images_4/erwtima4_3_locallyequalized_histogram.png');


