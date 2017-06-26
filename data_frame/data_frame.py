from __future__ import absolute_import, division, print_function

import numpy as np


class DataFrame(object):


    def __init__(self,
                 path_to_input_file,
                 number_of_output_neurons
                 ):


        array = np.load(path_to_input_file)


        self.data                    = array[:, number_of_output_neurons:-1]
        self.event_weights           = array[:, -1]

        if number_of_output_neurons == 1:
            self.labels = array[:, 0]
        else:
            self.labels = array[:, :number_of_output_neurons]


        self.number_of_events = self.data.shape[0]




    def batches(self,
                batch_size
                ):


        permutation = np.random.permutation(self.number_of_events)

        current_index = 0
        while (current_index + batch_size <= self.number_of_events):
            batch_indices = permutation[current_index : current_index+batch_size]

            current_index += batch_size

            yield (self.data         [batch_indices],
                   self.labels       [batch_indices],
                   self.event_weights[batch_indices]
                   )




    def get_data_labels_weights(self):


        return self.data, self.labels, self.event_weights
