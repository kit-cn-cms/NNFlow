from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .model_analyser import ModelAnalyser



class BinaryModelAnalyser(ModelAnalyser):


    def __init__(self,
                 path_to_model,
                 batch_size_classification = 200000,
                 session_config            = None,
                 ):


        ModelAnalyser.__init__(self, path_to_model, batch_size_classification, session_config)


        self._network_type = 'binary'
        self._number_of_output_neurons = 1




    def plot_output_distribution(self,
                                 path_to_data,
                                 save_path,
                                 file_name
                                 ):


        labels, predictions, event_weights = self._get_labels_predictions_event_weights(path_to_data)

        df_labels        = pd.DataFrame(labels,        columns=['label'])
        df_predictions   = pd.DataFrame(predictions,   columns=['prediction'])
        df_event_weights = pd.DataFrame(event_weights, columns=['event_weight'])

        df = pd.concat([df_labels, df_predictions, df_event_weights], axis=1)


        signal_predictions   = df.query('label==1')['prediction'].values
        signal_event_weights = df.query('label==1')['event_weight'].values

        background_predictions   = df.query('label==0')['prediction'].values
        background_event_weights = df.query('label==0')['event_weight'].values


        plt.clf()

        bin_edges = np.linspace(0, 1, 30)

        plt.hist(signal_predictions,     bins=bin_edges, weights=signal_event_weights,     histtype='step', lw=1.5, label='Signal',     normed='True', color='#1f77b4')
        plt.hist(background_predictions, bins=bin_edges, weights=background_event_weights, histtype='step', lw=1.5, label='Background', normed='True', color='#d62728')

        plt.legend(loc='upper left')
        plt.xlabel('Network Output')
        plt.ylabel('Events (normalized)')

        plt.savefig(os.path.join(save_path, file_name + '.pdf'))

        plt.clf()




    def plot_training_validation_output_distribution(self,
                                                     path_to_training_data,
                                                     path_to_validation_data,
                                                     save_directory,
                                                     output_file_name,
                                                     ):


        training_labels,   training_predictions,   training_event_weights   = self._get_labels_predictions_event_weights(path_to_training_data)
        validation_labels, validation_predictions, validation_event_weights = self._get_labels_predictions_event_weights(path_to_validation_data)


        df_training_labels        = pd.DataFrame(training_labels,        columns=['label'])
        df_training_predictions   = pd.DataFrame(training_predictions,   columns=['prediction'])
        df_training_event_weights = pd.DataFrame(training_event_weights, columns=['event_weight'])

        df_training = pd.concat([df_training_labels, df_training_predictions, df_training_event_weights], axis=1)


        training_signal_predictions   = df_training.query('label==1')['prediction'].values
        training_signal_event_weights = df_training.query('label==1')['event_weight'].values

        training_background_predictions   = df_training.query('label==0')['prediction'].values
        training_background_event_weights = df_training.query('label==0')['event_weight'].values


        df_validation_labels        = pd.DataFrame(validation_labels,        columns=['label'])
        df_validation_predictions   = pd.DataFrame(validation_predictions,   columns=['prediction'])
        df_validation_event_weights = pd.DataFrame(validation_event_weights, columns=['event_weight'])

        df_validation = pd.concat([df_validation_labels, df_validation_predictions, df_validation_event_weights], axis=1)


        validation_signal_predictions   = df_validation.query('label==1')['prediction'].values
        validation_signal_event_weights = df_validation.query('label==1')['event_weight'].values

        validation_background_predictions   = df_validation.query('label==0')['prediction'].values
        validation_background_event_weights = df_validation.query('label==0')['event_weight'].values


        plt.clf()

        bin_edges   = np.linspace(0, 1, 30)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

        validation_signal_histogram     = np.histogram(validation_signal_predictions,     weights=validation_signal_event_weights,     bins=bin_edges, normed=True)[0]
        validation_background_histogram = np.histogram(validation_background_predictions, weights=validation_background_event_weights, bins=bin_edges, normed=True)[0]

        plt.hist(training_signal_predictions,     bins=bin_edges, weights=training_signal_event_weights,     histtype='step', lw=1.5, label='Signal (Training)',     normed='True', color='#1f77b4')
        plt.hist(training_background_predictions, bins=bin_edges, weights=training_background_event_weights, histtype='step', lw=1.5, label='Background (Training)', normed='True', color='#d62728')

        plt.plot(bin_centres, validation_signal_histogram,     ls='', marker='o', markersize=3, color='#1f77b4', label='Signal (Validation)')
        plt.plot(bin_centres, validation_background_histogram, ls='', marker='o', markersize=3, color='#d62728', label='Background (Validation)')
 
        plt.legend(loc='upper left')
        plt.xlabel('Network Output')
        plt.ylabel('Events (normalized)')

        plt.savefig(os.path.join(save_directory, output_file_name + '.pdf'))

        plt.clf()
