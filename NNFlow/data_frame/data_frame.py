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


        self._number_of_events    = self._data.shape[0]
        self._number_of_variables = self._data.shape[1]




    def get_data_labels_event_weights_as_batches(self,
                                                 batch_size,
                                                 sort_events_randomly,
                                                 include_smaller_last_batch
                                                 ):


        if sort_events_randomly:
            permutation = np.random.permutation(self._number_of_events)

        else:
            permutation = np.array(range(self._number_of_events))


        current_index = 0
        while (current_index + batch_size <= self._number_of_events):
            batch_indices = permutation[current_index : current_index+batch_size]

            current_index += batch_size

            yield (self._data         [batch_indices],
                   self._labels       [batch_indices],
                   self._event_weights[batch_indices]
                   )


        if include_smaller_last_batch:
            if current_index < self._number_of_events:
                batch_indices = permutation[current_index :]
                
                yield (self._data         [batch_indices],
                       self._labels       [batch_indices],
                       self._event_weights[batch_indices]
                       )




    def get_data(self):


        return self._data




    def get_data_labels_event_weights(self):


        return self._data, self._labels, self._event_weights




    def get_labels_event_weights(self):


        return self._labels, self._event_weights




    def get_number_of_variables(self):


        return self._number_of_variables
