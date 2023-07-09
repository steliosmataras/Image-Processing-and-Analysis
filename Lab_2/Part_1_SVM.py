#Module for the implementation of the second part of the project in order to compare the CNN approach with the SVM one.
#The SVM approach is implemented using the scikit-learn library.
#As of 07/07/2023, the code does  contain the bonus part of the project, thus HOG implementation in the skimage.feature library is used.

#Importing the libraries
import os
import numpy as np  
from HOG import HOG #my personal implementation of HOG
#importing specific methods and classes
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
import Utils #my personal implemented functions for the First and Second Part of the project
#----------------------------------------------------------------------------------------------------#
#Importing yaml file for configuration
config_path = os.path.join(os.getcwd(), "config.yml")
config = Utils.configuration_loader(config_path)
#----------------------------------------------------------------------------------------------------#
#Step 0: Load the data, from the MNIST dataset, using the keras library
#Importing the dataset
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#----------------------------------------------------------------------------------------------------#
if config["HOG"]["own_implementation"] != True:
    #Step 1: Extract HOG features for data(using skimage.feature library) 
    #Step 1.1: Extract HOG features for train data
    hog_featurevectors_xtrain = []
    for image in range(X_train.shape[0]):
        fdlocal_xtrain,hog_image = hog(X_train[image,:,:], orientations=config["HOG"]["orientations"], pixels_per_cell=(config["HOG"]["pixels_per_cell"],config["HOG"]["pixels_per_cell"]),cells_per_block=(config["HOG"]["cells_per_block"], config["HOG"]["cells_per_block"]),block_norm= 'L2',visualize=True)
        hog_featurevectors_xtrain.append(fdlocal_xtrain) #Each feture vector will be size 
    Xtrain = np.array(hog_featurevectors_xtrain)#1-D numpy array 
    #Step 1.2: extract HOG features for test data
    hog_featurevectors_xtest = []
    for image in range(X_test.shape[0]):
        fdlocal_xtest,hog_image = hog(X_test[image,:,:], orientations=config["HOG"]["orientations"], pixels_per_cell=(config["HOG"]["pixels_per_cell"],config["HOG"]["pixels_per_cell"]),cells_per_block=(config["HOG"]["cells_per_block"], config["HOG"]["cells_per_block"]),block_norm= 'L2',visualize=True)
        hog_featurevectors_xtest.append(fdlocal_xtest) #Each feture vector will be size 
    Xtest = np.array(hog_featurevectors_xtest)#1-D numpy array 
#----------------------------------------------------------------------------------------------------#
else:
    #Step 2:(optional) Extract HOG features for train data(using my personal implementation of HOG in the HOG.py module)
    #Step 2.1: Extract HOG features for train data
    myhog_featurevectors_xtrain = []
    for image in range(X_train.shape[0]):
        my_hog = HOG(X_train[image,:,:], bins=config["HOG"]["orientations"], cell_size=config["HOG"]["pixels_per_cell"],block_size=config["HOG"]["cells_per_block"])
        my_fdlocal_xtrain = my_hog.extract()
        myhog_featurevectors_xtrain.append(my_fdlocal_xtrain)  
    Xtrain_from_myhog = np.array(myhog_featurevectors_xtrain)#1-D numpy array 
    #Step 2.1: extract HOG features for test data
    myhog_featurevectors_xtest = []
    for image in range(X_test.shape[0]):
        my_hog = HOG(X_test[image,:,:], bins=config["HOG"]["orientations"], cell_size=config["HOG"]["pixels_per_cell"],block_size=config["HOG"]["cells_per_block"])
        my_fdlocal_xtest = my_hog.extract()
        myhog_featurevectors_xtest.append(my_fdlocal_xtest) 
    Xtest_from_myhog = np.array(myhog_featurevectors_xtest)#1-D numpy array 
#----------------------------------------------------------------------------------------------------#
#Step 3: Apply the SVM classifier, using HOG features.
general_confusion_m_myimplementation = None
my_confusion_m_myimplementation = None
if config["HOG"]["own_implementation"] != True: 
    #Step 3.1: Apply the SVM classifier, using HOG features, from skimage.feature library implementation. 
    svm = SVC(kernel=config["classifier"]["SVC"]["kernel"] , C = config["classifier"]["SVC"]['C'] ,gamma=config["classifier"]["SVC"]['gamma'])
    svm.fit(Xtrain, y_train )
    general_predictions = svm.predict(Xtest)
    print('Confusion Matrix:')
    """general_confusion_m = confusion_matrix(y_test, general_predictions)
    print(general_confusion_m) """
    general_confusion_m_myimplementation = Utils.confusion_matrix(y_test, general_predictions)
    print(general_confusion_m_myimplementation)
else:  
#Step 3.2:(optional) Apply the SVM classifier, using HOG features, from my own implementation.  
    svm = SVC(kernel=config["classifier"]["SVC"]["kernel"] , C = config["classifier"]["SVC"]['C'] ,gamma=config["classifier"]["SVC"]['gamma'])
    svm.fit(Xtrain_from_myhog, y_train )
    my_predictions = svm.predict(Xtest_from_myhog)
    print('Confusion Matrix:')
    """ my_confusion_m = confusion_matrix(y_test, my_predictions)
    print(my_confusion_m) """
    my_confusion_m_myimplementation = Utils.confusion_matrix(y_test, my_predictions)
    print(my_confusion_m_myimplementation)
#----------------------------------------------------------------------------------------------------#
#Step 4: Display the confusion matrix from initial SVM classifier(non-optimized).
if config["HOG"]["own_implementation"] != True: 
#Step 4.1: Display the confusion matrix, using HOG features, from skimage.feature library implementation.
    Utils.confusion_matrix_display(general_confusion_m_myimplementation, labels=config["labels"])
else:  
#Step 4.2:(optional) Display the confusion matrix, using HOG features, from my own implementation.  
    Utils.confusion_matrix_display(my_confusion_m_myimplementation, labels=config["labels"])

#----------------------------------------------------------------------------------------------------#
#Step 6: Fine tune the SVM classifier using GridSearchCV, from scikit-learn library, and select the best parameters for the final confusion matrix.
if config["fine_tune"]["Permit"]:
    optimizedclassifier_general_confusion_m_myimplementation = None
    optimizedclassifier_my_confusion_m_myimplementation = None
    grid = GridSearchCV(estimator=SVC(), param_grid=config["fine_tune"]["param_grid"], verbose=config["fine_tune"]["verbose"],refit=True,cv = config["fine_tune"]["cv"])#cv is for number of folds
    if config["HOG"]["own_implementation"] != True:
        #Executing grid search to find the best parameter values a multitute of kernels and hyperparameter values are used as seen in the.yml file
        grid.fit(X_train, y_train )
        #C is penalty parameter, we try to create a grid with values that differ at least one order of metric
        if config["fine_tune"]["check_results"]:   
            #Show best parameters and best score
            print(grid.best_params_,grid.best_score_)
            #Show the full spectrum of the best estimator
            print(grid.best_estimator_)

        skimagehog_grid_predictions = grid.predict(X_test)
        optimizedclassifier_general_confusion_m_myimplementation = Utils.confusion_matrix(y_test, skimagehog_grid_predictions)

    else:
        #Executing grid search to find the best parameter values a multitute of kernels and hyperparameter values are used as seen in the.yml file
        grid.fit(Xtrain_from_myhog, y_train )
        #C is penalty parameter, we try to create a grid with values that differ at least one order of metric
        if config["fine_tune"]["check_results"]:   
            #Show best parameters and best score
            print(grid.best_params_,grid.best_score_)
            #Show the full spectrum of the best estimator
            print(grid.best_estimator_)

        myhog_grid_predictions = grid.predict(Xtest_from_myhog)
        optimizedclassifier_my_confusion_m_myimplementation = Utils.confusion_matrix(y_test, myhog_grid_predictions)


    #----------------------------------------------------------------------------------------------------#
    #Step 6.1: Display the confusion matrix from optimized SVM classifier.
    if config["HOG"]["own_implementation"] != True: 
    #Step 6.1.1: Display the confusion matrix, using HOG features, from skimage.feature library implementation.
        Utils.confusion_matrix_display(optimizedclassifier_general_confusion_m_myimplementation, labels=config["labels"])
    else:  
    #Step 6.1.2:(optional) Display the confusion matrix, using HOG features, from my own implementation.  
        Utils.confusion_matrix_display(optimizedclassifier_my_confusion_m_myimplementation, labels=config["labels"])

    #----------------------------------------------------------------------------------------------------#
#Classification report 
if config["HOG"]["own_implementation"] != True:
    print(classification_report(y_test, general_predictions))
else:
    print(classification_report(y_test, my_predictions))
#----------------------------------------------------------------------------------------------------#

