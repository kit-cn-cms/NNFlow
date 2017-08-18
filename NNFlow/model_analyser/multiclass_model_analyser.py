from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import tensorflow as tf

from NNFlow.model_analyser.model_analyser                     import ModelAnalyser          as NNFlowModelAnalyser
from NNFlow.softmax_output_processor.softmax_output_processor import SoftmaxOutputProcessor as NNFlowSoftmaxOutputProcessor




class MulticlassModelAnalyser(NNFlowModelAnalyser):


    def __init__(self,
                 path_to_model,
                 batch_size_classification = 200000,
                 session_config            = None,
                 ):


        NNFlowModelAnalyser.__init__(self, path_to_model, batch_size_classification, session_config)


        self._network_type = 'multiclass'


        tf_config = self._session_config.get_tf_config()
        tf_graph = tf.Graph()
        with tf.Session(config=tf_config, graph=tf_graph) as tf_session:
            tf_saver = tf.train.import_meta_graph(self._path_to_model + '.meta')
            tf_saver.restore(tf_session, self._path_to_model)
            self._output_labels = tf_graph.get_tensor_by_name('output_labels:0').eval()


        self._number_of_output_neurons = len(self._output_labels)


        self.softmax_output_processor = NNFlowSoftmaxOutputProcessor()




    def get_accuracy(self,
                     path_to_data,
                     ):


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_data)

        confusion_matrix = self.softmax_output_processor.get_confusion_matrix(labels, network_output, event_weights)

        accuracy = np.diagonal(confusion_matrix).sum() / confusion_matrix.sum()


        return accuracy




    def get_mean_roc_auc(self,
                         path_to_data,
                         ):


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_data)

        mean_roc_auc = self.softmax_output_processor.get_mean_roc_auc(labels, network_output, event_weights)

        return mean_roc_auc




    def get_signal_over_background(self,
                                   path_to_data,
                                   cross_sections,
                                   ):


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_data)

        confusion_matrix = self.softmax_output_processor.get_confusion_matrix(labels, network_output, event_weights, cross_sections)


        signal_over_background = OrderedDict()

        for j in range(self._number_of_output_neurons):
            s = np.diagonal(confusion_matrix)[j]
            b = confusion_matrix[:, j].sum() - s

            signal_over_background[self._output_labels[j]] = s/b


        return signal_over_background




    def plot_confusion_matrix(self,
                              save_path,
                              filename_outputfile,
                              path_to_input_file,
                              cross_sections = 'equal',
                              ):


        plt.clf()


        labels, network_output, event_weights = self.get_labels_network_output_event_weights(path_to_input_file)
        confusion_matrix = self.softmax_output_processor.get_confusion_matrix(labels, network_output, event_weights, cross_sections)

        y_shape, x_shape = confusion_matrix.shape


        cmap = matplotlib.cm.RdYlBu_r
        cmap.set_bad(color='white')

        minimum = np.min(confusion_matrix) / (np.pi**2.0 * np.exp(1.0)**2.0)
        maximum = np.max(confusion_matrix) * np.pi**2.0 * np.exp(1.0)**2.0

        x = np.linspace(0, x_shape, x_shape+1)
        y = np.linspace(0, y_shape, y_shape+1)

        xn, yn = np.meshgrid(x,y)

        plt.pcolormesh(xn, yn, confusion_matrix, cmap=cmap, norm=colors.LogNorm(vmin=max(minimum, 1e-6), vmax=maximum))
        plt.colorbar()

        plt.xlim(0, x_shape)
        plt.ylim(0, y_shape)

        plt.xlabel("Predicted")
        plt.ylabel("True")

        for yit in range(confusion_matrix.shape[0]):
            for xit in range(confusion_matrix.shape[1]):
                plt.text(xit + 0.5, yit + 0.5, '{:.1f}'.format(confusion_matrix[yit, xit]), horizontalalignment='center', verticalalignment='center')

        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)

        ax.set_xticklabels(list(self._output_labels) + [str(i) for i in range(x_shape-y_shape)])
        ax.set_yticklabels(     self._output_labels)

        ax.set_aspect('equal')

        plt.tight_layout()

        plt.savefig(os.path.join(save_path, filename_outputfile + '.pdf'))


        plt.clf()
