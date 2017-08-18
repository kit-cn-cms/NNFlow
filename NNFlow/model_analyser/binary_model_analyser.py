from __future__ import absolute_import, division, print_function

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sklearn.metrics

from NNFlow.model_analyser.model_analyser import ModelAnalyser as NNFlowModelAnalyser




class BinaryModelAnalyser(NNFlowModelAnalyser):


    def __init__(self,
                 path_to_model,
                 batch_size_classification = 200000,
                 session_config            = None,
                 ):


        ModelAnalyser.__init__(self, path_to_model, batch_size_classification, session_config)


        self._network_type = 'binary'
        self._number_of_output_neurons = 1




    def get_roc_auc(self,
                    path_to_data,
                    ):
    

        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_data)

        roc_auc = sklearn.metrics.roc_auc_score(y_true = labels, y_score = network_output, sample_weight = event_weights)


        return roc_auc




    def plot_output_distribution(self,
                                 path_to_data,
                                 save_path,
                                 file_name
                                 ):


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_data)

        df_labels         = pd.DataFrame(labels,         columns=['label'])
        df_network_output = pd.DataFrame(network_output, columns=['network_output'])
        df_event_weights  = pd.DataFrame(event_weights,  columns=['event_weight'])

        df = pd.concat([df_labels, df_network_output, df_event_weights], axis=1)


        signal_network_output = df.query('label==1')['network_output'].values
        signal_event_weights  = df.query('label==1')['event_weight'].values

        background_network_output = df.query('label==0')['network_output'].values
        background_event_weights  = df.query('label==0')['event_weight'].values


        plt.clf()

        bin_edges = np.linspace(0, 1, 30)

        plt.hist(signal_network_output,     bins=bin_edges, weights=signal_event_weights,     histtype='step', lw=1.5, label='Signal',     normed='True', color='#1f77b4')
        plt.hist(background_network_output, bins=bin_edges, weights=background_event_weights, histtype='step', lw=1.5, label='Background', normed='True', color='#d62728')

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


        training_labels,   training_network_output,   training_event_weights   = self.get_labels_network_output_event_weights(path_to_training_data)
        validation_labels, validation_network_output, validation_event_weights = self.get_labels_network_output_event_weights(path_to_validation_data)


        df_training_labels         = pd.DataFrame(training_labels,         columns=['label'])
        df_training_network_output = pd.DataFrame(training_network_output, columns=['network_output'])
        df_training_event_weights  = pd.DataFrame(training_event_weights,  columns=['event_weight'])

        df_training = pd.concat([df_training_labels, df_training_network_output, df_training_event_weights], axis=1)


        training_signal_network_output = df_training.query('label==1')['network_output'].values
        training_signal_event_weights  = df_training.query('label==1')['event_weight'].values

        training_background_network_output = df_training.query('label==0')['network_output'].values
        training_background_event_weights  = df_training.query('label==0')['event_weight'].values


        df_validation_labels         = pd.DataFrame(validation_labels,         columns=['label'])
        df_validation_network_output = pd.DataFrame(validation_network_output, columns=['network_output'])
        df_validation_event_weights  = pd.DataFrame(validation_event_weights,  columns=['event_weight'])

        df_validation = pd.concat([df_validation_labels, df_validation_network_output, df_validation_event_weights], axis=1)


        validation_signal_network_output = df_validation.query('label==1')['network_output'].values
        validation_signal_event_weights  = df_validation.query('label==1')['event_weight'].values

        validation_background_network_output = df_validation.query('label==0')['network_output'].values
        validation_background_event_weights  = df_validation.query('label==0')['event_weight'].values


        plt.clf()

        bin_edges   = np.linspace(0, 1, 30)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

        validation_signal_histogram     = np.histogram(validation_signal_network_output,     weights=validation_signal_event_weights,     bins=bin_edges, normed=True)[0]
        validation_background_histogram = np.histogram(validation_background_network_output, weights=validation_background_event_weights, bins=bin_edges, normed=True)[0]

        plt.hist(training_signal_network_output,     bins=bin_edges, weights=training_signal_event_weights,     histtype='step', lw=1.5, label='Signal (Training)',     normed='True', color='#1f77b4')
        plt.hist(training_background_network_output, bins=bin_edges, weights=training_background_event_weights, histtype='step', lw=1.5, label='Background (Training)', normed='True', color='#d62728')

        plt.plot(bin_centres, validation_signal_histogram,     ls='', marker='o', markersize=3, color='#1f77b4', label='Signal (Validation)')
        plt.plot(bin_centres, validation_background_histogram, ls='', marker='o', markersize=3, color='#d62728', label='Background (Validation)')
 
        plt.legend(loc='upper left')
        plt.xlabel('Network Output')
        plt.ylabel('Events (normalized)')

        plt.savefig(os.path.join(save_directory, output_file_name + '.pdf'))

        plt.clf()
