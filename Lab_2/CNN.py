#Python module containing CNN architecture implementation 

#Importing the libraries
from tensorflow.keras import Sequential , optimizers
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import Callback #callback class for inheritance in the plotsCallback class
from Utils import one_hot_decoding #my personal implemented function of one hot decoding 
accuracy_list = []
loss_list = []
class CNN():
    model = None
    def __init__(self,parameters_dict) -> None:
        #parameters_dict is a dictionary containing all the parameters needed for the CNN architecture
        self.kernel_size = parameters_dict["kernelsize"]
        self.shape = (parameters_dict["imageshape"][0],parameters_dict["imageshape"][1],parameters_dict["imageshape"][2])
        self.learning_rate =parameters_dict["learning_rate"] 
        self.num_epochs = parameters_dict["epochnum"]
        self.batch_size = parameters_dict["minibatchsize"]
        #convolutional layer parameters
        self.firstlayer_filters = parameters_dict["firstlayer"]
        self.secondlayer_filters = parameters_dict["secondlayer"]
        #activation functions
        activations_list  = parameters_dict["activations"]
        self.conv_activation_1 = activations_list[0]
        self.conv_activation_2 = activations_list[1]
        self.dense_activation_1 = activations_list[2]
        self.dense_activation_2 = activations_list[3]
        self.output_layer_activation = activations_list[4]
        #loss function
        self.loss_function = parameters_dict["loss_function"]
        #optimizer
        self.optimizer = parameters_dict["optimizer"]

    def create_model_(self):
        """
        Method for the creation of the CNN architecture. The architecture is composed by two convolutional layers and three dense layers as per the project's pdf instruction. Additional layers of BatchNormalization and AveragePooling are added in order to improve the performance of the model. All the parameters needed for the creation of the model are taken from the yaml configuration file.
        input: None
        output: None


        """
        model = Sequential()
        #Convolutional layers for creation of features maps and extraction of features
        model.add(Conv2D(filters=self.firstlayer_filters, kernel_size=(self.kernel_size,self.kernel_size),strides = (1,1), activation=self.conv_activation_1, input_shape=self.shape))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(filters=self.secondlayer_filters, kernel_size=(self.kernel_size,self.kernel_size),strides = (1,1), activation=self.conv_activation_2))
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
        #Flattening convolutional layers output in order to feed the dense layers for the final classification process
        model.add(Flatten())
        #Dense layers for classification
        model.add(Dense(units=120, activation=self.dense_activation_1))
        model.add(Dense(units=84, activation=self.dense_activation_2))
        model.add(Dense(units=10, activation=self.output_layer_activation))
        self.model = model
        return None
    def compile_model_(self):
        """
        Method for the compilation of the model, based on optimizer selection. The parameters needed for the compilation are taken from the yaml configuration file.
        input: None
        output: None

        """
        optimizer = None
        #optimizer selection
        if self.optimizer == "Adam":
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = optimizers.SGD(learning_rate=self.learning_rate)
        #compilation of the model
        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['accuracy'])
        #training of the model
        return None
    def fit_model_(self, X_train, y_train,X_test,y_test):
        """
        Method for the training of the model. The parameters needed for the training are taken from the yaml configuration file.
        input: X_train, y_train
        output: training_history -> object containing the history of the training process

        """
        #callback class for plotting the loss and accuracy curves
        class plotsCallback(Callback):
            """
            Class for the creation of the callback object, important for plots of loss and accuracy(as per pdf instructions).
            """
            def on_epoch_end(self, epoch, logs={}):
                if(logs.get('accuracy')>0.995):
                    print("\nReached 99.5% accuracy so cancelling training!")
                    self.model.stop_training = True
                loss,accuracy = self.model.evaluate(X_test, y_test,verbose = 1)
                accuracy_list.append(accuracy*100)
                loss_list.append(loss)

        mycallback = plotsCallback()
        training_history = self.model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=self.batch_size,callbacks = [mycallback],use_multiprocessing=True,workers=4,validation_split=0.1)

        return training_history , accuracy_list, loss_list
    def evaluate_model_(self, X_test, y_test,labels = None,verbose = 1):
        """
        Method for the evaluation of the model. The parameters needed for the evaluation are taken from the yaml configuration file.
        input: X_test, y_test
        output: None

        """
        _,model_accuracy=self.model.evaluate(X_test, y_test,verbose = verbose,use_multiprocessing=True,workers=4)
        print("Accuracy of the model is: %.2f" % (model_accuracy*100))
        predictions = self.model.predict(X_test)
        my_encoded_predictions = one_hot_decoding(predictions,labels)
        return my_encoded_predictions
    
    
