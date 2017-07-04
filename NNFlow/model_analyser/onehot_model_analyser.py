from __future__ import absolute_import, division, print_function

import numpy as np

from .model_analyser import ModelAnalyser


class OneHotModelAnalyser(ModelAnalyser):


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
