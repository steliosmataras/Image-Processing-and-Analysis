function [scalar] = local_histogram_kernel(I,x,y,L)
%UNTITLED9 Summary of this function goes here
[m,n] = size(I);
step = floor(L/2);
array = zeros(1,256);
        for i =max(x-step,1):min(x+step,m)%computing histogram around the neighborhood
            for j=max(y-step,1):min(y+step,n)
                array(1+I(i,j)) = array(1+I(i,j))+1;

            end

        end

array = array./(L*L); %computing propabilities


 % computing cdf only for the cell that we need
 array(1+I(x,y)) = sum(array(1:(1+I(x,y))));


scalar = array(1+I(x,y)); % rounding to the nearest integer as per the book
