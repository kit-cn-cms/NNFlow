from __future__ import absolute_import, division, print_function

import numpy as np

from sklearn.metrics import roc_auc_score




class SoftmaxOutputProcessor(object):


    def __init__(self):


        self.set_mode_max_output()




    def get_confusion_matrix(self,
                             labels,
                             network_output,
                             event_weights,
                             cross_sections = None,
                             ):


        number_of_events         = labels.shape[0]
        number_of_output_neurons = labels.shape[1]

        if self._additional_predicted_bin:
            confusion_matrix = np.zeros((number_of_output_neurons, number_of_output_neurons+1), dtype=np.float32)
        else:
            confusion_matrix = np.zeros((number_of_output_neurons, number_of_output_neurons), dtype=np.float32)

        for i in range(number_of_events):
            index_true      = np.argmax(labels[i])
            index_predicted = self.get_prediction(network_output[i])

            confusion_matrix[index_true][index_predicted] += event_weights[i]


        if cross_sections is not None:
            for j in range(number_of_output_neurons):
                confusion_matrix[j] /= confusion_matrix[j].sum()

            if isinstance(cross_sections, basestring):
                if cross_sections == 'equal':
                    confusion_matrix *= number_of_output_neurons
                    confusion_matrix *= 100

            else:
                for j in range(number_of_output_neurons):
                    confusion_matrix[j] *= cross_sections[j]


        return confusion_matrix




    def get_mean_roc_auc(self,
                         labels,
                         network_output,
                         event_weights
                         ):


        number_of_output_neurons = labels.shape[1]

        roc_auc_list = list()

        for j in range(number_of_output_neurons):
            y_true  = labels[:, j]
            y_score = network_output[:, j]

            roc_auc = roc_auc_score(y_true=y_true, y_score=y_score, sample_weight=event_weights)

            roc_auc_list.append(roc_auc)


        mean_roc_auc = np.mean(roc_auc_list)


        return mean_roc_auc




    def get_prediction(self,
                       network_output,
                       ):


        if self._mode == 'max_output':
            prediction = np.argmax(network_output)

        elif self._mode == 'custom_function':
            prediction = self._predict_function(network_output)


        return prediction




    def set_mode_max_output(self):

        self._mode = 'max_output'




    def set_mode_custom_function(self,
                                 function,
                                 additional_predicted_bin=False,
                                 ):


        self._mode = 'custom_function'

        self._predict_function         = function
        self._additional_predicted_bin = additional_predicted_bin
