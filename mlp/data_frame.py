from __future__ import absolute_import, division, print_function

import numpy as np


class DataFrame(object):


    def __init__(self,
                 path_to_input_file,
                 number_of_output_neurons
                 ):


        array = np.load(path_to_input_file)


        self._data          = array[:, number_of_output_neurons:-1]
        self._event_weights = array[:, -1]

        if number_of_output_neurons == 1:
            self._labels = array[:, 0]
        else:
            self._labels = array[:, :number_of_output_neurons]


        self._number_of_events = self.data.shape[0]




    def batches(self,
                batch_size
                ):


        permutation = np.random.permutation(self._number_of_events)

        current_index = 0
        while (current_index + batch_size <= self._number_of_events):
            batch_indices = permutation[current_index : current_index+batch_size]

            current_index += batch_size

            yield (self._data         [batch_indices],
                   self._labels       [batch_indices],
                   self._event_weights[batch_indices]
                   )




    def get_data(self):


        return self._data




    def get_data_labels_weights(self):


        return self._data, self._labels, self._event_weights
