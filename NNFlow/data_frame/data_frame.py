from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd


class DataFrame(object):


    def __init__(self,
                 path_to_input_file,
                 ):


        with pd.HDFStore(path_to_input_file, mode='r') as store_input:
            array                 = store_input.select('data').values
            self._network_type    = store_input.select('network_type').iloc[0]
            self._preselection    = store_input.select('preselection').iloc[0]
            self._input_variables = store_input.select('inputVariables').values

            if self._network_type == 'multiclass':
                self._output_labels = store_input.select('outputLabels').values

        self._number_of_input_neurons = len(self._input_variables)

        if self._network_type == 'multiclass':
            self._number_of_output_neurons = len(self._output_labels)
        elif self._network_type == 'binary':
            self._number_of_output_neurons = 1


        self._data          = array[:, self._number_of_output_neurons:-1]
        self._event_weights = array[:, -1]

        if self._number_of_output_neurons == 1:
            self._labels = array[:, 0]
        else:
            self._labels = array[:, :self._number_of_output_neurons]


        self._number_of_events = self._data.shape[0]




    def get_data(self):


        return self._data




    def get_data_labels_event_weights(self):


        return self._data, self._labels, self._event_weights




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




    def get_input_variables(self):


        return self._input_variables




    def get_labels_event_weights(self):


        return self._labels, self._event_weights




    def get_network_type(self):


        return self._network_type




    def get_number_of_output_neurons(self):


        return self._number_of_output_neurons




    def get_number_of_input_neurons(self):


        return self._number_of_input_neurons




    def get_output_labels(self):


        return self._output_labels




    def get_preselection(self):


        return self._preselection
