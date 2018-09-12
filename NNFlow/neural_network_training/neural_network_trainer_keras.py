from __future__ import absolute_import, division, print_function

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score

from NNFlow.data_frame.data_frame                             import DataFrame              as NNFlowDataFrame






class NeuralNetworkTrainer(object):


    def __init__(self):

        # Limit gpu usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def train(self,
              save_path,
              model_id,
              hidden_layers,
              activation_function_name,
              dropout_keep_probability,
              l2_regularization_beta,
              early_stopping_intervall,
              path_to_training_data_set,
              path_to_validation_data_set,
              path_to_test_data_set,
              batch_size_training,
              ):


        print('\n' + '=======================')
        print(       'TRAINING NEURAL NETWORK')
        print(       '=======================' + '\n')

        # Check given paths
        if not os.path.isdir(save_path):
            sys.exit("Directory '" + save_path + "' doesn't exist." + "\n")

        if not os.path.isfile(path_to_training_data_set):
            sys.exit("File '" + path_to_training_data_set + "' doesn't exist." + "\n")

        if not os.path.isfile(path_to_validation_data_set):
            sys.exit("File '" + path_to_validation_data_set + "' doesn't exist." + "\n")

        directory_model_properties = os.path.join(save_path, 'model_properties')
        directory_plots            = os.path.join(save_path, 'plots')

        # Create directories
        if not os.path.isdir(directory_model_properties):
            os.mkdir(directory_model_properties)
        if not os.path.isdir(directory_plots):
            os.mkdir(directory_plots)

        # Load data.
        training_data_set   = NNFlowDataFrame(path_to_training_data_set)
        validation_data_set = NNFlowDataFrame(path_to_validation_data_set)
        test_data_set = NNFlowDataFrame(path_to_test_data_set)

        # Drop the given parameter in the dataset

        number_of_input_neurons  = training_data_set.get_number_of_input_neurons()
        number_of_output_neurons = training_data_set.get_number_of_output_neurons()

        # Get weights and labels
        train_label, train_weights = training_data_set.get_labels_event_weights()
        vali_label, vali_weights= validation_data_set.get_labels_event_weights()
        test_label, test_weights= test_data_set.get_labels_event_weights()

        # Get scaled data. data is scaled between 0 and 1
        train_data =training_data_set.get_scaled_data()
        vali_data =validation_data_set.get_scaled_data()
        test_data =test_data_set.get_scaled_data()

        # Build the model with the given parameters
        model = self._build_model(layers=hidden_layers,
                                  activation_function=activation_function_name,
                                  number_of_input_neurons=number_of_input_neurons,
                                  number_of_output_neurons=number_of_output_neurons,
                                  dropout=dropout_keep_probability,
                                  l2_regularization=l2_regularization_beta)

        # Create the optimier. Here the Adamoptimizer is used but can be changed to a different optimizer
        optimizer = tf.train.AdamOptimizer(1e-3)

        # Compile the model with mean squared error as the loss function
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['binary_accuracy'])
        model.summary()

        # Define earlystopping to reduce overfitting
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=early_stopping_intervall)
        ival = IntervalEvaluation(validation_data=(vali_data, vali_label), interval=1)
        # time_callback is used to get the time per epoch
        time_callback = TimeHistory()

        # Define max number of epochs
        nEpochs = 200
        print(train_data.shape)
        print(train_label.shape)
        # Train the NN
        history = model.fit(x=train_data,
                            y=train_label,
                            batch_size=batch_size_training,
                            epochs=nEpochs,
                            verbose=1,
                            callbacks=[earlyStopping,time_callback,ival],
                            validation_data=(vali_data,vali_label,vali_weights),
                            sample_weight = train_weights)

        # Get the time spend per epoch
        times = time_callback.times

        # Determine the number of epochs the NN trained
        if earlyStopping.stopped_epoch ==0:
            Epoch_end = nEpochs
        else:
            Epoch_end=earlyStopping.stopped_epoch

        # Plot the history of the loss function
        #self._plot_history(history= history,path=directory_plots)
        print('\n')
        print("Training finished, evaluating model on test data set")
        print('\n')
        # Evaluate the trained model on the test data sample
        [loss,mae] = model.evaluate(x=test_data,
                                    y=test_label,
                                    verbose=1,
                                    sample_weight=test_weights)
        print("Testing set Mean Abs Error: ${:7.5f}".format(mae))
        print('\n')

        # Create 2D plot for the predicted parameter

        # Save cpkt file for this NN
        sess = keras.backend.get_session()
        saver = tf.train.Saver()
        save_path = saver.save(sess, directory_model_properties )

        # Print and save some info about the NN
        network_and_training_properties_string = self._get_network_and_training_properties_string(model_id                     = model_id,
                                                                                                  number_of_input_neurons      = number_of_input_neurons,
                                                                                                  hidden_layers                = hidden_layers,
                                                                                                  activation_function_name     = activation_function_name,
                                                                                                  dropout_keep_probability     = dropout_keep_probability,
                                                                                                  l2_regularization_beta       = l2_regularization_beta,
                                                                                                  early_stopping_interval      = early_stopping_intervall,
                                                                                                  batch_size_training          = batch_size_training,
                                                                                                  early_stopping_epoch         = Epoch_end,
                                                                                                  mean_training_time_per_epoch=np.mean(times),
                                                                                                  )

        # Save additional Info of NN
        with open(os.path.join(directory_model_properties, 'NN_Info.txt'), 'w') as NN_Info_output_file:
            NN_Info_output_file.write(network_and_training_properties_string)

        print(network_and_training_properties_string, end='')

        with open(os.path.join(directory_model_properties, 'inputVariables.txt'), 'w') as outputfile_input_variables:
            for variable in training_data_set.get_input_variables():
                outputfile_input_variables.write(variable + '\n')


        with open(os.path.join(directory_model_properties, 'preselection.txt'), 'w') as outputfile_preselection:
            outputfile_preselection.write(training_data_set.get_preselection() + '\n')

        print('\n' + '========')
        print(       'FINISHED')
        print(       '========' + '\n')



    def _build_model(self,
                     layers,
                     activation_function,
                     number_of_input_neurons,
                     number_of_output_neurons,
                     dropout,
                     l2_regularization,
                     network_type = 'binary'):
        # Builds the layers to a model

        # Define first layer, which takes all input features
        model = keras.Sequential()
        model.add(keras.layers.Dense(layers[0],
                                     activation=activation_function,
                                     input_shape=(number_of_input_neurons,),
                                     kernel_regularizer=keras.regularizers.l2(l2_regularization)))
        # Check if dropout is given, add the dropout layer if so
        if dropout !=1:
            model.add(keras.layers.Dropout(dropout))

        # Build the rest of the model
        for i in range(len(layers)-1):
            model.add(keras.layers.Dense(layers[i+1],
                                         activation=activation_function,
                                         kernel_regularizer=keras.regularizers.l2(l2_regularization)))
            if dropout != 1:
                model.add(keras.layers.Dropout(dropout))
        # Build last layer, regression only works with 1 parameter
        if network_type =='binary':
            model.add(keras.layers.Dense(number_of_output_neurons,
                                         activation='sigmoid',
                                         kernel_regularizer=keras.regularizers.l2(l2_regularization)))
        else:
            model.add(keras.layers.Dense(number_of_output_neurons,
                                         activation='softmax',
                                         kernel_regularizer=keras.regularizers.l2(l2_regularization)))
        return model




    def _get_network_and_training_properties_string(self,
                                                    model_id,
                                                    number_of_input_neurons,
                                                    hidden_layers,
                                                    activation_function_name,
                                                    dropout_keep_probability,
                                                    l2_regularization_beta,
                                                    early_stopping_interval,
                                                    batch_size_training,
                                                    early_stopping_epoch,
                                                    mean_training_time_per_epoch
                                                    ):
        
        # Prints model properties and returns those properties to be saved in a file
        column_width = 60


        network_and_training_properties = str()


        network_and_training_properties += '{:{width}} {}\n'.format('Model ID:', model_id, width=column_width)
        network_and_training_properties += '\n'

        network_and_training_properties += '{:{width}} {}\n'.format('Network type:', 'regression', width=column_width)
        network_and_training_properties += '\n'

        network_and_training_properties += '{:{width}} {}\n'.format('Number of input variables:', number_of_input_neurons, width=column_width)
        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {}\n'.format('Hidden layers:',       hidden_layers,            width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('Activation function:', activation_function_name, width=column_width)
        network_and_training_properties += '\n'


        network_and_training_properties += '{:{width}} {}\n'.format('Keep probability (dropout):', dropout_keep_probability, width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('L2 regularization:',          l2_regularization_beta,   width=column_width)
        network_and_training_properties += '{:{width}} {}\n'.format('Early stopping interval:',    early_stopping_interval,  width=column_width)
        network_and_training_properties += '\n'

        network_and_training_properties += '{:{width}} {}\n'.format('Batch size:', batch_size_training, width=column_width)
        network_and_training_properties += '\n'
        network_and_training_properties += '{:{width}} {}\n'.format('Stopping epoch:', early_stopping_epoch,
                                                                    width=column_width)
        network_and_training_properties += '{:{width}} {:.2f} s\n'.format('Mean training time per epoch:', mean_training_time_per_epoch, width=column_width)
        network_and_training_properties += '\n'


        return network_and_training_properties



    def _plot_history(self, history,path):

        # Plots the history of the loss function
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
                 label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
                 label='Val loss')
        plt.legend()
        plt.savefig(path + "/Plot_history_loss")

        # Save data to file to change plots later
        data_dict= {"Epochs": history.epoch,
                    "Train Loss": list(history.history['mean_absolute_error']),
                    'Val loss': list(history.history['val_mean_absolute_error'])}
        np.savez(path+"/history_data_file.np",data_dict)

from keras.callbacks import Callback

class TimeHistory(keras.callbacks.Callback):
    # Class to track the time spend for each epoch
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print(score)



# ==================================================================
# IMPORTANT: Make a copy of this file before you insert your values!
# ==================================================================

import os
import datetime

import NNFlow

# ----------------------------------------------------------------------------------------------------


workdir_base = '/storage/9/jschindler/'
name_subdir = 'NN_v1'


number_of_hidden_layers = 3
number_of_neurons_per_layer = 200
hidden_layers = [number_of_neurons_per_layer for i in range(number_of_hidden_layers)]

### Available activation functions: 'elu', 'relu', 'tanh', 'sigmoid', 'softplus'
activation_function_name = 'elu'

early_stopping_intervall = 10

### Parameter for dropout
dropout_keep_probability = 1.

### Parameter for L2 regularization
l2_regularization_beta = 0

batch_size_training = 500
optimizer = None

parameter = 'Evt_blr_ETH'
# ----------------------------------------------------------------------------------------------------


model_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

save_path = os.path.join(workdir_base, name_subdir, 'tth/model_' + model_id)
model_name = name_subdir + '_' + model_id

path_to_training_data_set = os.path.join(workdir_base, 'data_sets_2/training_data_set.hdf')
path_to_validation_data_set = os.path.join(workdir_base, 'data_sets_2/validation_data_set.hdf')
path_to_test_data_set =  os.path.join(workdir_base, 'data_sets_2/test_data_set.hdf')
# ----------------------------------------------------------------------------------------------------
if not os.path.isdir(save_path):
    if os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(save_path)
# ----------------------------------------------------------------------------------------------------
train_dict = {'save_path': save_path,
              'model_id': model_id,
              'hidden_layers': hidden_layers,
              'activation_function_name': activation_function_name,
              'dropout_keep_probability': dropout_keep_probability,
              'l2_regularization_beta': l2_regularization_beta,
              'early_stopping_intervall': early_stopping_intervall,
              'path_to_training_data_set': path_to_training_data_set,
              'path_to_validation_data_set': path_to_validation_data_set,
              'path_to_test_data_set' : path_to_test_data_set,
              'batch_size_training': batch_size_training,
              }

t = NeuralNetworkTrainer()
t.train(**train_dict)