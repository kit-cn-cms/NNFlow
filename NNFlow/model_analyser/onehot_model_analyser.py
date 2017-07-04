from __future__ import absolute_import, division, print_function

import os

import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .model_analyser import ModelAnalyser




class OneHotModelAnalyser(ModelAnalyser):


    def plot_heatmap(self,
                     save_path,
                     filename_outputfile,
                     path_to_input_file,
                     processes,
                     ):


        array_predicted_true = self._get_predicted_true_matrix(path_to_input_file)
        array_predicted_true *= 10000


        cmap = matplotlib.cm.RdYlBu_r
        cmap.set_bad(color='white')

        minimum = np.min(array) / (np.pi**2.0 * np.exp(1.0)**2.0)
        maximum = np.max(array) * np.pi**2.0 * np.exp(1.0)**2.0

        x = np.linspace(0, self._number_of_output_neurons, self._number_of_output_neurons+1)
        y = np.linspace(0, self._number_of_output_neurons, self._number_of_output_neurons+1)

        xn, yn = np.meshgrid(x,y)

        plt.pcolormesh(xn, yn, array, cmap=cmap, norm=colors.LogNorm(vmin=max(minimum, 1e-6), vmax=maximum))
        plt.colorbar()

        plt.xlim(0, self._number_of_output_neurons)
        plt.ylim(0, self._number_of_output_neurons)

        plt.xlabel("Predicted")
        plt.ylabel("True")

        for yit in range(array.shape[0]):
            for xit in range(array.shape[1]):
                plt.text(xit + 0.5, yit + 0.5, '{:.1f}'.format(array[yit, xit]), horizontalalignment='center', verticalalignment='center')

        ax = plt.gca()
        ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
        ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
        ax.set_xticklabels(processes)
        ax.set_yticklabels(processes)

        plt.savefig(os.path.join(save_path, filename_outputfile + '.pdf'))




    def _get_predicted_true_matrix(self,
                                   path_to_input_file
                                   ):


        labels, predictions, event_weights = self._get_labels_predictions_event_weights(path_to_input_file)

        array_predicted_true = np.zeros((self._number_of_output_neurons, self._number_of_output_neurons), dtype=np.float32)


        index_true        = np.argmax(labels, axis=1)
        index_predictions = np.argmax(predictions, axis=1)


        for i in range(index_true.shape[0]):
            array_predicted_true[index_true[i]][index_predictions[i]] += event_weights[i]


        return array_predicted_true
