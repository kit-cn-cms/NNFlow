from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd


class DataFrame(object):


    def __init__(self,
                 path_to_input_file,
                 network_type,
                 ):


        with pd.HDFStore(path_to_input_file, mode='r') as store_input:
            array           = store_input.select('data').values
            self._variables = store_input.select('variables').values

            if network_type == 'one-hot':
                self._processes = store_input.select('processes').values

        self._number_of_input_neurons = len(self._variables)

        if network_type == 'one-hot':
            self._number_of_output_neurons = len(self._processes)
        elif network_type == 'binary':
            self._number_of_output_neurons = 1


        self._data          = array[:, self._number_of_output_neurons:-1]
        self._event_weights = array[:, -1]

        if self._number_of_output_neurons == 1:
            self._labels = array[:, 0]
        else:
            self._labels = array[:, :self._number_of_output_neurons]


        self._number_of_events    = self._data.shape[0]




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




    def get_number_of_output_neurons(self):


        return self._number_of_output_neurons




    def get_number_of_input_neurons(self):


        return self._number_of_input_neurons




    def get_processes(self):


        return self._processes




    def get_variables(self):


        return self._variables
