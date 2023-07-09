#This is a module containing the total for the HOG descriptor as well as the documentation for the functions. The reasoning is that the module can work as an implementation libary for the HOG descriptor, and the documentation can be used as a reference for the functions.

#The first step of calculation in many feature detectors in image pre-processing is to ensure normalized color and gamma values. As Dalal and Triggs point out, however, this step can be omitted in HOG descriptor computation, as the ensuing descriptor normalization essentially achieves the same result. Image pre-processing thus provides little impact on performance. Instead, the first step of calculation is the computation of the gradient values.


import numpy as np
import argparse
import matplotlib.pyplot as plt
import math 
from skimage import filters
#imports for testing purposes
from skimage.feature import hog
from numpy import floor

class HOG:
    feature_vector = np.array([])
    def __init__(self, image, cell_size=8, block_size=2, bins=9, factor=1e-7):
        self.image = image
        self.cell_size = cell_size #for each individual histogram
        self.block_size = block_size #histograms of 8*8 cells to be accounted for in the overlapping window
        self.bins = bins
        self.angle_unit = 180 // self.bins #floor division
        self.epsilon = factor #for numerical stability in normalization
 
    def extract(self):
        '''
            Function to extract the HOG features from the image
            input: None
            output: feature_vector:ndarray
        ''' 
        gradient_magnitude, gradient_angle = self.sobel_edge_detection()
        histogram_matrix = self.get_histogram_matrix(gradient_magnitude, gradient_angle)
        feature_vector  = self.block_normalization(histogram_matrix) #this is the feature vector for the image
        return feature_vector
                                         
    def sobel_edge_detection(self, verbose=False):
        # Edge Detection Kernel
        #fixed selection of kernel, as the kernel is not the focus of the project
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        #new_image_x = self.convolve2D(self.image, kernel, strides=1)
    
        if verbose:
            plt.imshow(new_image_x, cmap='gray')
            plt.title("Horizontal Edge")
            plt.show()
    
        #new_image_y = self.convolve2D(self.image, kernel.transpose(), strides=1)
    
        if verbose:
            plt.imshow(new_image_y, cmap='gray')
            plt.title("Vertical Edge")
            plt.show()
        #testing
        new_image_y,new_image_x=np.gradient(self.image)
        # Calculate Gradient Magnitude
        #gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
        gradient_magnitude = np.hypot(new_image_x, new_image_y)
        """Rescaling magnitude to 0-255
        gradient_magnitude *= 255.0 / gradient_magnitude.max() """
        #Calculate Gradient Angle
        gradient_angle = np.arctan2(new_image_y, new_image_x)
        #converting radians to degrees
        gradient_angle = np.rad2deg(gradient_angle) % 180 #theoretically, the angle should be between 0 and 180, but just in case
        if verbose:
            plt.imshow(gradient_magnitude, cmap='gray')
            plt.title("Gradient Magnitude")
            plt.show()

        if verbose:
            plt.imshow(gradient_magnitude, cmap='gray')
            plt.title("Gradient Magnitude")
            plt.show()
  
        return gradient_magnitude, gradient_angle

    def convolve2D(self,image:np.ndarray, kernel:np.ndarray, strides=1):
        # Standard convolution using corelation function with flipped kernel
        #needs standard convolution reworking
        
        kernel = np.flipud(np.fliplr(kernel))

        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]
        padding_x = (xKernShape//2)
        padding_y = (yKernShape//2) 
        # Shape of Output of standard Convolution
        xOutput = int(((xImgShape) / strides))
        yOutput = int(((yImgShape ) / strides))
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        imagePadded = np.zeros((image.shape[0] + padding_x*2, image.shape[1] + padding_y*2))
        imagePadded[int(padding_x):int(-1 * padding_x), int(padding_y):int(-1 * padding_y)] = image #probably works as intended, but needs testing
        

        # Iterate through image
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides implementation of interleaving
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides, implementation of interleaving
                        if x % strides == 0:
                            output[x, y] = (kernel * imagePadded[x: 2*padding_x+x+1 , y: 2*padding_y+y+1]).sum()
                    except:
                        break

        return output

    def get_histogram(self, gradient_magnitude:np.ndarray, gradient_angle:np.ndarray)->np.ndarray:
        '''
            Function to calculate the histogram of the image, for a single cell in the image(As defined by the cell size i.e 8x8).
            input: gradient_magnitude:ndarray, gradient_angle:ndarray
            output: histogram:ndarray

            **note it has been shown that unsigned gradients work better than signed gradients, so we will be using unsigned gradients for the histogram calculation
        '''
        histogram = np.zeros(self.bins)
        for i in range(gradient_magnitude.shape[0]):
            for j in range(gradient_magnitude.shape[1]):
                angle = abs(gradient_angle[i][j]) # angle should be between 0 and 180, and theoretically there shouldnt be any problems with negative angles, but just in case
                #indexes of lower and upper bins(bins that need to be affected by the gradient magnitude)
                bin_index_lower = int(floor((angle / self.angle_unit)-(1/2)))%self.bins #in order for the bin to wrap when next bin is below 0 degrees
                bin_index_upper = int((bin_index_lower + 1 )) % self.bins # in order for the bin to wrap when next bin is above 180 degrees
                #centers of bins
                lower_centre = (bin_index_lower + 1/2)*self.angle_unit
                upper_centre = (bin_index_upper + 1/2)*self.angle_unit
                #calculating the percentage of the gradient magnitude that needs to be added to the lower and upper bins
                bin_lower_vote = gradient_magnitude[i][j]*((upper_centre - angle)/self.angle_unit) #the percentage of the gradient magnitude that needs to be added to the lower bin)
                bin_upper_vote = gradient_magnitude[i][j]*((angle - lower_centre)/self.angle_unit) #the percentage of the gradient magnitude that needs to be added to the upper bin)
                histogram[bin_index_lower] += bin_lower_vote
                histogram[bin_index_upper] += bin_upper_vote
        #print(np.linalg.norm(histogram, ord=2)==0)
        #/ gradient_magnitude.shape[0]*gradient_magnitude.shape[1] #normalizing the histogram
        return histogram 

                

    def get_histogram_matrix(self, gradient_magnitude:np.ndarray, gradient_angle:np.ndarray)->np.ndarray:
        '''     
           Function to calculate the histogram matrix of the image, for every cell in the image(As defined by the cell size i.e 8x8).
           The histogram matrix is a 3D matrix, with the first two dimensions representing the number of cells in the image, and the third dimension representing the number of bins in the histogram.
           input: gradient_magnitude:ndarray, gradient_angle:ndarray
           output: histogram_matrix:ndarray
        
        '''
        histogram_matrix = np.zeros((math.ceil(int(gradient_magnitude.shape[0] / self.cell_size)), math.ceil(int(gradient_magnitude.shape[1] / self.cell_size)), self.bins))
        for i in range(0,(histogram_matrix.shape[0])):
            if (i+1)*self.cell_size < histogram_matrix.shape[0]:
                for j in range(0,(histogram_matrix.shape[1])):
                    if (j+1)*self.cell_size < histogram_matrix.shape[1]:
                        histogram_matrix[i][j][:] = self.get_histogram(gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size, j * self.cell_size:(j + 1) * self.cell_size], gradient_angle[i * self.cell_size:(i + 1) * self.cell_size, j * self.cell_size:(j + 1) * self.cell_size])
                    else:
                        histogram_matrix[i][j][:] = self.get_histogram(gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size, j * self.cell_size:int(gradient_magnitude.shape[1])], gradient_angle[i * self.cell_size:(i + 1) * self.cell_size, j * self.cell_size:int(gradient_magnitude.shape[1])])
            else:
                for j in range(0,(histogram_matrix.shape[1])):
                    if (j+1)*self.cell_size < histogram_matrix.shape[1]:
                        histogram_matrix[i][j][:] = self.get_histogram(gradient_magnitude[i * self.cell_size:int(gradient_magnitude.shape[0]), j * self.cell_size:(j + 1) * self.cell_size], gradient_angle[i * self.cell_size:int(gradient_magnitude.shape[0]), j * self.cell_size:(j + 1) * self.cell_size])
                    else:
                        histogram_matrix[i][j][:] = self.get_histogram(gradient_magnitude[i * self.cell_size:int(gradient_magnitude.shape[0]), j * self.cell_size:int(gradient_magnitude.shape[1])], gradient_angle[i * self.cell_size:int(gradient_magnitude.shape[0]), j * self.cell_size:int(gradient_magnitude.shape[1])])
        return histogram_matrix
    def block_normalization(self, histogram_matrix:np.ndarray)->np.ndarray:
        '''
            Function to normalize the histogram matrix, in order to account for lighting changes in the image. Works like CNN convolution operation
            input: histogram_matrix:ndarray
            output: feature_vector:ndarray
        '''
        
        #feature_vector = np.empty(len(range(0,(histogram_matrix.shape[0]-self.block_size)+1)) * len(range(0,(histogram_matrix.shape[1]-self.block_size)+1))*(self.block_size**2) * self.bins)
        normalized_blocks = np.zeros((len(range(0,(histogram_matrix.shape[0]-self.block_size)+1)), len(range(0,(histogram_matrix.shape[1]-self.block_size )+1)), self.block_size, self.block_size, self.bins))
        feature_vector = np.array([])
        for i in range(0,(histogram_matrix.shape[0]-self.block_size)+1):
                for j in range(0,(histogram_matrix.shape[1]-self.block_size )+1):
                    local_block_vector = np.array([])
                    """ for cells_row in range(0, self.block_size):
                            for cells_column in range(0, self.block_size):
                                local_block_vector = np.append(local_block_vector,histogram_matrix[i+cells_row,j+cells_column,:]) """
                    local_block_vector = histogram_matrix[i:i+self.block_size,j:j+self.block_size,:]
                    normalized_blocks[i][j][:] = local_block_vector/(np.sqrt(np.sum(local_block_vector ** 2)+self.epsilon))         
                    #feature_vector = np.append(feature_vector, local_block_vector)
                    feature_vector = normalized_blocks.ravel()

        feature_vector = feature_vector/(np.sqrt(np.sum(feature_vector ** 2)+self.epsilon)) #normalizing the feature vector
        feature_vector = np.clip(feature_vector, None, 0.2) #clipping the feature vector
        feature_vector = feature_vector/(np.sqrt(np.sum(feature_vector ** 2)+self.epsilon)) #normalizing the feature vector
        return feature_vector
""" if __name__ == '__main__':
    # Grayscale Image
    #image = cv2.imread('butterfly.png', 0)
    # Convolve and Save Output
    #my_hog = HOG(image)
    #output1,output2 = hog.sobel_edge_detection(verbose=True)
    #cv2.imwrite('2D_standard_Convolved_butterfly_.jpg', output1)
    
    #just for testing purposes
    output = filters.sobel(image)
    plt.imshow(output,cmap= plt.cm.gray)
    plt.show()
    
    myfd = my_hog.extract()
    fdskimage,hogplotimage = hog(image,orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm= 'L2',visualize=True)
    error_between_hog_implementations = np.linalg.norm(fdskimage - myfd, ord=2)/np.linalg.norm(fdskimage, ord=2)
    print(error_between_hog_implementations) """

    