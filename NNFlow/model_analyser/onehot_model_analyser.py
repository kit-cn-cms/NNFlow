from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow as tf

from .model_analyser import ModelAnalyser
from NNFlow.onehot_output_processor.onehot_output_processor import AdvancedOneHotOutputProcessor




class OneHotModelAnalyser(ModelAnalyser):


    def __init__(self,
                 path_to_model,
                 batch_size_classification = 200000,
                 session_config            = None,
                 ):


        ModelAnalyser.__init__(self, path_to_model, batch_size_classification, session_config)


        self._network_type = 'one-hot'


        config = self._session_config.get_config()
        graph = tf.Graph()
        with tf.Session(config=config, graph=graph) as sess:
            saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            saver.restore(sess, self._path_to_model)
            self._names_output_neurons = graph.get_tensor_by_name('names_output_neurons:0').eval()


        self._number_of_output_neurons = len(self._names_output_neurons)


        self.onehot_output_processor = AdvancedOneHotOutputProcessor(self._names_output_neurons)




    def get_accuracy(self,
                     path_to_data,
                     ):


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_data)

        array_predicted_true = self.onehot_output_processor.get_predicted_true_matrix(labels, network_output, event_weights)

        accuracy = np.diagonal(array_predicted_true).sum() / array_predicted_true.sum()


        return accuracy




    def get_mean_roc_auc(self,
                         path_to_data,
                         ):


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_data)

        mean_roc_auc = self.onehot_output_processor.get_mean_roc_auc(labels, network_output, event_weights)

        return mean_roc_auc




    def get_signal_over_background(self,
                                   path_to_data,
                                   cross_sections,
                                   ):


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_data)

        array_predicted_true = self.onehot_output_processor.get_predicted_true_matrix(labels, network_output, event_weights, cross_sections)


        signal_over_background = OrderedDict()

        for j in range(self._number_of_output_neurons):
            s = np.diagonal(array_predicted_true)[j]
            b = array_predicted_true[:, j].sum() - s

            signal_over_background[self._names_output_neurons[j]] = s/b


        return signal_over_background




    def plot_heatmap(self,
                     save_path,
                     filename_outputfile,
                     path_to_input_file,
                     cross_sections = 'equal',
                     ):


        plt.clf()


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_input_file)
        array_predicted_true = self.onehot_output_processor.get_predicted_true_matrix(labels, network_output, event_weights, cross_sections)


        cmap = matplotlib.cm.RdYlBu_r
        cmap.set_bad(color='white')

        minimum = np.min(array_predicted_true) / (np.pi**2.0 * np.exp(1.0)**2.0)
        maximum = np.max(array_predicted_true) * np.pi**2.0 * np.exp(1.0)**2.0

        x = np.linspace(0, self._number_of_output_neurons, self._number_of_output_neurons+1)
        y = np.linspace(0, self._number_of_output_neurons, self._number_of_output_neurons+1)

        xn, yn = np.meshgrid(x,y)

        plt.pcolormesh(xn, yn, array_predicted_true, cmap=cmap, norm=colors.LogNorm(vmin=max(minimum, 1e-6), vmax=maximum))
        plt.colorbar()

        plt.xlim(0, self._number_of_output_neurons)
        plt.ylim(0, self._number_of_output_neurons)

        plt.xlabel("Predicted")
        plt.ylabel("True")

        for yit in range(array_predicted_true.shape[0]):
            for xit in range(array_predicted_true.shape[1]):
                plt.text(xit + 0.5, yit + 0.5, '{:.1f}'.format(array_predicted_true[yit, xit]), horizontalalignment='center', verticalalignment='center')

        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(self._names_output_neurons)
        ax.set_yticklabels(self._names_output_neurons)

        plt.tight_layout()

        plt.savefig(os.path.join(save_path, filename_outputfile + '.pdf'))


        plt.clf()
