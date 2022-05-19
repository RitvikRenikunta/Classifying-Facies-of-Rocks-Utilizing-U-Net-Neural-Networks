import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
import keras_tuner as kt
from matplotlib import pyplot as plt
import numpy as np

from data_preprocessing import DataPreprocess

class U_Net:
    def __init__(self, training_seismic_location, training_labels_location, dropout_rate = 0.5, learning_rate = 1e-4, input_size = (176, 64, 3), layers = 6):
        self.training_seismic, self.training_labels, self.training_seismic_resized, self.training_labels_resized = self.pre_process_data(training_seismic_location, training_labels_location)
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layers = layers
        self.architecture() # will set self.model

    def pre_process_data(self, seismic_location, labels_location):
        Data = DataPreprocess(seismic_location, labels_location)
        return (Data.seismic, Data.labels, Data.seismic_resized, Data.labels_resized)

    def architecture(self):
        inputs = Input(self.input_size)

        #Downsampling

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(self.dropout_rate)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(self.dropout_rate)(conv5)

        #Upsampling 

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        activation = Activation('softmax')(conv9)

        self.model = Model(inputs, activation)

        self.model.compile(optimizer = Adam(learning_rate = self.learning_rate), loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics = ['accuracy'])

    def train(self):
        self.results = self.model.fit(self.training_seismic_resized, self.training_labels_resized, batch_size=32, epochs=20, verbose = 1, validation_split=0.05)
    
    def predict(self, seismic_resized, labels_resized):
        return self.model.predict(seismic_resized)

    def display_accuracy_loss(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        plt.plot(x, self.results.history['accuracy'], label = 'training')
        plt.plot(x, self.results.history['val_accuracy'], label = 'validation')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()

        plt.plot(x, self.results.history['loss'], label = 'training')
        plt.plot(x, self.results.history['val_loss'], label = 'validation')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

# Hyperparameter Tuning

# Building U-net architecture using hyperparameter tuned variables
def build_unet(hp, input_size = (176, 64, 3)):
    inputs = Input(input_size)
    hp_units_learning_rate = hp.Choice('Learning rate', values = [1e-2, 1e-3, 1e-4, 1e-5])
    hp_units_dropout = hp.Choice('Dropout rate', values = [0.5, 0.6, 0.7, 0.8])

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(rate = hp_units_dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(rate = hp_units_dropout)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
   
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    activation = Activation('softmax')(conv9)

    model = Model(inputs, activation)

    model.compile(optimizer = Adam(learning_rate = hp_units_learning_rate), loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics = ['accuracy'])

    return model

def bayesian_hyperaparemeter_tuning(seismic_resized, labels_resized):
    bayesian_tuner = kt.BayesianOptimization(build_unet, objective='val_loss', max_trials = 5, project_name='UNET-Bayesian')

    bayesian_tuner.search(seismic_resized, labels_resized, epochs=15, validation_split=0.1)

    best_hps=bayesian_tuner.get_best_hyperparameters(num_trials=1)
    print("Bayesian Optimization:")
    print(f"Learning rate: {best_hps[0].get('Learning rate')}")
    print(f"Dropout rate: {best_hps[0].get('Dropout rate')}")

def hyperband_hyperaparemeter_tuning(seismic_resized, labels_resized):
    hyperband_tuner = kt.Hyperband(build_unet, objective='val_loss', project_name='UNET-Hyperband')

    hyperband_tuner.search(seismic_resized, labels_resized, epochs=15, validation_split=0.1)

    best_hps=hyperband_tuner.get_best_hyperparameters(num_trials=1)
    print("Hyperband Optimization:")
    print(f"Learning rate: {best_hps[0].get('Learning rate')}")
    print(f"Dropout rate: {best_hps[0].get('Dropout rate')}")

# Model Training (based on hyperparameter tuning)
def train_with_optimization(seismic_resized, labels_resized):
    bayesian_unet = U_Net(seismic_resized, labels_resized, dropout_rate = 0.5, learning_rate = 1e-5)
    bayesian_unet.train()

    hyperband_unet = U_Net(seismic_resized, labels_resized, dropout_rate = 0.8, learning_rate = 1e-4)
    hyperband_unet.train() 

    return (bayesian_unet, hyperband_unet)

if __name__ == '__main__':
    model = U_Net('data/train/train_seismic.npy', 'data/train/train_labels.npy')
    model.train()

    # Predicting from training set
    training_predictions = model.predict(model.train_seismic_resized, model.train_labels_resized)
    plt.imshow(np.argmax(training_predictions[10], -1)) # can change index
    plt.imshow(model.train_labels_resized[10])

    # Predicting from testing set
    test_seismic, test_labels, test_seismic_resized, test_labels_resized = model.pre_process_data('data/test_once/test1_seismic.npy', 'data/test_once/test1_labels.npy')
    testing_predictions = model.predict(test_seismic_resized, test_labels_resized)
    plt.imshow(np.argmax(testing_predictions[10], -1)) # can change index
    plt.imshow(test_labels_resized[10])

    # Plot accuracy and loss vs time
    model.display_accuracy_loss()

    # Training optimized models
    bayesian_model, hyperband_model = train_with_optimization(model.train_seismic_resized, model.train_labels_resized)
    
    # Bayesian and Hyperband Model Training Results
    bayesian_model.display_accuracy_loss()
    hyperband_model.display_accuracy_loss()

    bayesian_predictions = bayesian_model.predict(model.train_seismic_resized, model.train_labels_resized)
    plt.imshow(np.argmax(bayesian_predictions[10], -1)) # can change index
    plt.imshow(test_labels_resized[10])

    hyperband_predictions = hyperband_model.predict(model.train_seismic_resized, model.train_labels_resized)
    plt.imshow(np.argmax(hyperband_predictions[10], -1)) # can change index
    plt.imshow(test_labels_resized[10])