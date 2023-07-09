#Module for the implementation of the second part of the project in order to compare the CNN approach with the SVM one.
#The CNN approach is implemented using Tensor-Flow library and corresponding Keras API.

#Importing the libraries
import os
import numpy as np


#Importing specific methods and classes
import Utils #my personal implemented functions for the First and Second Part of the project
from CNN import CNN #my personal implemented CNN architecture


#----------------------------------------------------------------------------------------------------#
#Importing yaml file for configuration
config_path = os.path.join(os.getcwd(), "config.yml")
config = Utils.configuration_loader(config_path)
#----------------------------------------------------------------------------------------------------#
#Step 0: Load the data, from the MNIST dataset, using the keras library
#Importing the dataset
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#Visualizing the representative images from 10 different classes in the dataset and more specically 0-9 numbers
Utils.visualize_dataset(X_train, y_train, config["labels"])
#----------------------------------------------------------------------------------------------------#
#Step 1: Preprocessing of the data
#Reshaping the data in order to feed the CNN architecture
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000,28,28,1)
#Normalizing the images to the range of [0,1]
X_train = X_train/255.0
X_test = X_test/255.0
#----------------------------------------------------------------------------------------------------#
#Step 2: One hot encoding of the labels for the classification task
#One hot encoding of the labels
y_train = Utils.one_hot_encoding(y_train)
y_test = Utils.one_hot_encoding(y_test)
#----------------------------------------------------------------------------------------------------#
#Step 3: Creation of the CNN architecture
#Creation of the CNN architecture, using the parameters taken from the yaml configuration file
myconvolutionalneuralnetwork = CNN(config["CNN"]["constructor_parameters"])
myconvolutionalneuralnetwork.create_model_()
#----------------------------------------------------------------------------------------------------#
#Step 4: Training of the CNN architecture
#Training of the CNN architecture, using the parameters taken from the yaml configuration file
myconvolutionalneuralnetwork.compile_model_()
history,accuracy_list,loss_list = myconvolutionalneuralnetwork.fit_model_(X_train, y_train,X_test, y_test)
#----------------------------------------------------------------------------------------------------#
#Step 5: Evaluation of the CNN architecture
#Evaluation of the CNN architecture, using the test set and my own confusion matrix implementation
y_predicted = myconvolutionalneuralnetwork.evaluate_model_(X_test, y_test,config["labels"])
#----------------------------------------------------------------------------------------------------#
#Step 6: Plotting of the loss and accuracy curves as well as the confusion matrix
#Plotting of the loss and accuracy curves


#Plotting of the confusion matrix
#Creation of the confusion matrix
my_confusion_matrix = Utils.confusion_matrix(Utils.one_hot_decoding(y_test,config["labels"]), y_predicted)
Utils.confusion_matrix_display(my_confusion_matrix, labels=config["labels"])

#---------------------------------------------------------------------------------------------#
#Plotting the loss and accuracy curves
Utils.plot_metric_total_set(accuracy_list, "accuracy")
Utils.plot_metric_total_set(loss_list, "loss")

#---------------------------------------------------------------------------------------------#
#Plotting the loss and accuracy curves for the validation set and training set
Utils.plot_metric_total_validation_train_set(history, "accuracy")
Utils.plot_metric_total_validation_train_set(history, "loss")
