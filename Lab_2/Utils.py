#This is a module containing all utilitary functions used in both parts of the project.

#Importing the libraries'
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from tensorflow import one_hot

#Own implementation of Confusion Matrix function
def confusion_matrix(y_true:np.array, y_pred:np.array)-> np.array:
    #y_true: ground truth labels
    #y_pred: predicted labels
    #returns: confusion matrix
    #----------------------------------------------------------------------------------------------------#
    #Step 1: Get the number of classes
    num_classes = len(np.unique(y_true))
    #----------------------------------------------------------------------------------------------------#
    #Step 2: Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    #----------------------------------------------------------------------------------------------------#
    #Step 3: Fill the confusion matrix
    for i in range(len(y_true)):#Confusion matrix of sklearn has true labels as rows and predicted labels as collumns
        confusion_matrix[y_true[i], y_pred[i]] += 1
    #----------------------------------------------------------------------------------------------------#
    return confusion_matrix.astype(int)

def confusion_matrix_display(confusion_matrix:np.array, labels:list)-> None:
    #confusion_matrix: confusion matrix
    #labels: list of labels
    #returns: None
    #----------------------------------------------------------------------------------------------------#
    #Step 1: Display the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    #----------------------------------------------------------------------------------------------------#

    return None

def configuration_loader(config_path:str)-> dict:
    #config_path: path to the configuration file
    #returns: configuration dictionary
    #----------------------------------------------------------------------------------------------------#
    #Step 1: Load the configuration file
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    #----------------------------------------------------------------------------------------------------#
    return config

def one_hot_encoding(label_array:np.array)-> np.array:
    #labels: labels to be encoded
    #returns: encoded labels in a 2-D matrice of shape (num_samples, num_classes)
    encodedlabel_matrice = one_hot(label_array.astype(np.int32), depth = 10)
    return encodedlabel_matrice
def one_hot_decoding(encoded_label_matrix:np.array,labels)-> np.array:
    #labels: onehot encoded matrice of predictions or true labels to be decoded
    #returns: vector of decoded labels from onehot encoded matrice of predictions
    #----------------------------------------------------------------------------------------------------#
    array_of_labels = np.array([int(i) for i in labels])
    y_vector = np.zeros(encoded_label_matrix.shape[0])
    for i in range(encoded_label_matrix.shape[0]):
        y_vector[i] = np.argmax(encoded_label_matrix[i,:], axis = -1)

    return y_vector.astype(int)

def visualize_dataset(X_train:np.array, y_train:np.array, labels:list)-> None:
    """
    Function for visualizing the representative images from 10 different classes in the dataset and more specically 0-9 numbers
    input: X_train: train set of images
           y_train: train set of labels
           labels: list of labels
    output: None
    
    """
    class_ = 0
    rows = 2
    columns = 5
    fig = plt.figure()
    for i in range(X_train.shape[0]):
        if class_ == 10:
            break
        if y_train[i] == class_:
            
            ax = fig.add_subplot(rows,columns,class_+1)
            ax.imshow(X_train[i,:,:], cmap='gray')
            ax.set_title(f"Number {labels[y_train[i]]}")
            class_ += 1
    plt.show()
def plot_metric_total_set(history, metric):
    '''
        Function to plot Results of Cnn training
    '''
    plt.figure(figsize=(10, 8))
    train_metrics = history
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.title('Test '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["Test_"+metric])
    plt.show( block = False)
    plt.savefig(f'test_{metric}.png')

def plot_metric_total_validation_train_set(history, metric):
    '''
        Function to plot Results of Cnn training
    '''
    plt.figure(figsize=(10, 8))
    train_metrics = history.history[metric]
    valid_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, valid_metrics)
    plt.title('Train set and Validation set '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["Train_"+metric, "Validation_"+metric])
    plt.show( block = False)
    plt.savefig(f'trainandvalidation_{metric}.png')