from __future__ import absolute_import, division, print_function

import numpy as np

import sklearn.metrics




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

        confusion_matrix = np.zeros((number_of_output_neurons, number_of_output_neurons + self._number_of_additional_bins), dtype=np.float32)

        for i in range(number_of_events):
            index_true      = np.argmax(labels[i])
            index_predicted = self._predict_function(network_output[i])

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

            roc_auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score, sample_weight=event_weights)

            roc_auc_list.append(roc_auc)


        mean_roc_auc = np.mean(roc_auc_list)


        return mean_roc_auc




    def get_one_vs_others_roc_aucs(self,
                                   labels,
                                   network_output,
                                   event_weights
                                   ):


        number_of_output_neurons = labels.shape[1]

        roc_auc_list = list()

        for j in range(number_of_output_neurons):
            y_true  = labels[:, j]
            y_score = network_output[:, j]

            roc_auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score, sample_weight=event_weights)

            roc_auc_list.append(roc_auc)


        return roc_auc_list




    def set_mode_max_output(self):

        self._mode = 'max_output'

        self._predict_function          = np.argmax
        self._number_of_additional_bins = 0




    def set_mode_custom_function(self,
                                 function,
                                 number_of_additional_bins=0,
                                 ):


        self._mode = 'custom_function'

        self._predict_function          = function
        self._number_of_additional_bins = number_of_additional_bins
