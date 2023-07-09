% Denoising  Implementation
%-----------------------------------------------
%Part1 gaussian white noise
% Step1: Given an input image f(x,y) of size M x N, obtain the pading
% parameters P and Q. Typically, we select P = 2M and Q = 2N
flower = imread('flower.png');
% Converting the image class into "double"
flower = im2double(flower);%255*
% reading the image size
[m1,n1] = size(flower);
dimensions=[m1 n1];
%creating noisy images
%first create white gaussian noise of variance enough for 15 db SNR afterwards
snr=15;
imagepower = sum(flower(:).^2)/length(flower(:));%is it defined diferently in signal processing
variance_of_noise = imagepower/(10^(snr/10));
white_gaussian_noise=sqrt(variance_of_noise)*randn(dimensions);
noisy_gaussian_flower=flower+white_gaussian_noise;
figure(1);
imshow(uint8(255.*flower),[]);
title("Original flower image");
print('images_3/original_flower','-dpng');
figure(2);
imshow(uint8(255.*noisy_gaussian_flower),[]);
title("Flower image with additive white gaussian noise,0.0077 variance ");
print('images_3/gaussian_noisy_flower','-dpng');
%first create salt and pepper noise of variance enough for 15 db SNR afterwards
percentage=0.25;
noisy_saltppepper_flower=imnoise(flower,'salt & pepper',percentage);
figure(3);
imshow(uint8(255.*flower),[]);
title("Original flower image");
figure(4);
imshow(uint8(255.*noisy_saltppepper_flower),[]);
title("Flower image with salt and pepper noise");
print('images_3/saltandpepper_noisy_flower','-dpng');

%%
%---------------------------------------------- filter denoising
%Preallocating arrays to store results of all size kernels for speed
moving_average_results_gaussian = zeros(9,m1,n1);
median_results_saltpepper = zeros(9,m1,n1);
moving_average_results_saltpepper = zeros(9,m1,n1);
median_results_gaussian = zeros(9,m1,n1);
%Step 3 filtering with all types of kernels of size 3x3 to 11x11
for i=3:11
    %-----------------------------------------------
    %median fliter denoising
    median_results_saltpepper(i-2,:,:) = medfilt2(noisy_saltppepper_flower,[i i]); %median filtering image with windows of size 3x3 to 11x11
    median_results_gaussian(i-2,:,:) = medfilt2(noisy_gaussian_flower,[i i]); %median filtering image with windows of size 3x3 to 11x11
    %Moving average filter denoising
    h = fspecial('average',i);
    moving_average_results_gaussian(i-2,:,:) = imfilter(noisy_gaussian_flower,h,'conv');
    moving_average_results_saltpepper(i-2,:,:) = imfilter(noisy_saltppepper_flower,h,'conv');
end
%Displaying results for:
%%
%median filtering 
%salt and pepper denoising
imshow(noisy_saltppepper_flower,[]);
for i=3:11
    figure;
    temp = squeeze(median_results_saltpepper(i-2,:,:));
    imshow(temp,[])
end
%%
%white gaussian denoising
imshow(noisy_gaussian_flower,[]);
for i=3:11
    figure;
    temp = squeeze(median_results_gaussian(i-2,:,:));
    imshow(temp,[])
end
%%
%moving average filtering 
%white gaussian denoising
imshow(noisy_gaussian_flower,[]);
for i=3:11
    figure;
    temp = squeeze(moving_average_results_gaussian(i-2,:,:));
    imshow(temp,[])
end
%%
%salt and pepper denoising
imshow(noisy_saltppepper_flower,[]);
for i=3:11
    figure;
    temp = squeeze(moving_average_results_saltpepper(i-2,:,:));
    imshow(temp,[])
end
%%
%%figure;
subplot(2, 2, 1)
imshow(noisy_saltppepper_flower,[]);
title("SandP 25% percentage")
subplot(2, 2, 2)
imshow(squeeze(median_results_saltpepper(4,:,:)),[]);
title("SandP best median filter 6x6")
subplot(2, 2, 3)
imshow(squeeze(moving_average_results_saltpepper(5,:,:)),[]);
title("SandP best averaging filter 7x7")
subplot(2, 2, 4)
imshow(squeeze(median_results_saltpepper(1,:,:)),[]);
title("SandP  median filter 3x3")
print(gcf, '-dpng', 'images_3/saltandpepper.png');

figure;
subplot(2, 2, 1)
imshow(flower,[]);
title("Initial Image")
subplot(2, 2, 2)
imshow(noisy_gaussian_flower,[]);
title("Gaussian noise so as SNR is 15db")

subplot(2, 2, 3)
imshow(squeeze(median_results_gaussian(3,:,:)),[]);
title("Gaussian Noise best median filter 5x5")
subplot(2, 2, 4)
imshow(squeeze(moving_average_results_gaussian(4,:,:)),[]);
title("Gaussian Noise best filter 6x6")
print(gcf, '-dpng', 'images_3/gaussian.png');

