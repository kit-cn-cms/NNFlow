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

from NNFlow.data_frame.data_frame                             import DataFrame              as NNFlowDataFrame






class NeuralNetworkTrainerRegression(object):


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
              parameter,
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
        training_data_set.drop_nuisance_parameter(parameter)
        validation_data_set.drop_nuisance_parameter(parameter)
        test_data_set.drop_nuisance_parameter(parameter)

        number_of_input_neurons  = training_data_set.get_number_of_input_neurons()
        number_of_output_neurons = training_data_set.get_number_of_output_neurons()

        # Get labels for the NN. get_adversary_labels() returns the labels for the dropped parameter
        train_label=training_data_set.get_adversary_labels()
        vali_label =validation_data_set.get_adversary_labels()
        test_label =test_data_set.get_adversary_labels()

        # Get weights
        train_weights = training_data_set.get_weights()
        vali_weights= validation_data_set.get_weights()
        test_weights= test_data_set.get_weights()

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
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae'])
        model.summary()

        # Define earlystopping to reduce overfitting
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=early_stopping_intervall)

        # time_callback is used to get the time per epoch
        time_callback = TimeHistory()

        # Define max number of epochs
        nEpochs = 200

        # Train the NN
        history = model.fit(x=train_data,
                            y=train_label,
                            batch_size=batch_size_training,
                            epochs=nEpochs,
                            verbose=1,
                            callbacks=[earlyStopping,time_callback],
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
        self._plot_history(history= history,
                           parameter=parameter,
                           path=directory_plots)
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
        test_predictions = model.predict(test_data).flatten()
        self._plot_prediction_true_value(test_label=test_label,
                                         test_predictions=test_predictions,
                                         parameter=parameter,
                                         path= directory_plots)

        # Save cpkt file for this NN
        sess = keras.backend.get_session()
        saver = tf.train.Saver()
        save_path = saver.save(sess, directory_model_properties )

        # Print and save some info about the NN
        network_and_training_properties_string = self._get_network_and_training_properties_string(model_id                     = model_id,
                                                                                                  parameter                    = parameter,
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
                     l2_regularization):
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
        # TODO: Change regression to predict 2 parameters
        if number_of_output_neurons ==1:
            model.add(keras.layers.Dense(number_of_output_neurons,
                                         kernel_regularizer=keras.regularizers.l2(l2_regularization)))
        return model




    def _get_network_and_training_properties_string(self,
                                                    model_id,
                                                    parameter,
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

        network_and_training_properties += '{:{width}} {}\n'.format('Regression Parameter:', parameter, width=column_width)
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


    def _plot_prediction_true_value(self, test_label, test_predictions, parameter,path):

        # Create 2D plot of true value over predicted value
        plt.clf()
        plt.hist2d(test_label,test_predictions, bins=80, cmin=1)
        plt.colorbar()
        plt.xlabel('True Values ')
        plt.ylabel('Predictions ')
        plt.title(parameter)
        plt.savefig(path+"/Plot_2D_"+parameter)

        # Save plot data to file to change plots later
        data_dict = {'label': list(test_label),
                     'prediction': list(test_predictions)}
        np.savez(path +"/"+ parameter + "2D_data_file.np", data_dict)


    def _plot_history(self, history,parameter,path):

        # Plots the history of the loss function
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
                 label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
                 label='Val loss')
        plt.legend()
        plt.savefig(path + "/Plot_history_loss" + parameter)

        # Save data to file to change plots later
        data_dict= {"Epochs": history.epoch,
                    "Train Loss": list(history.history['mean_absolute_error']),
                    'Val loss': list(history.history['val_mean_absolute_error'])}
        np.savez(path+"/"+parameter+"history_data_file.np",data_dict)


class TimeHistory(keras.callbacks.Callback):
    # Class to track the time spend for each epoch
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


