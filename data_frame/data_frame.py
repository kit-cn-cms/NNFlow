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
            self.event_labels = array[:, 0]
        else:
            self.event_labels = array[:, :number_of_output_neurons]


        self.number_of_events        = self.data.shape[0]


        self._next_id     = 0
        self._permutation = np.array(range(self.number_of_events))




    def get_data_labels_weights(self):


        return self.data, self.event_labels, self.event_weights




    def next_batch(self,
                   batch_size
                   ):


        if self._next_id + batch_size >= self.number_of_events:
            self.shuffle()

        batch_indices = self._permutation[self._next_id : self._next_id+batch_size]

        self._next_id += batch_size

        return (self.data   [batch_indices],
                self.labels [batch_indices],
                self.weights[batch_indices]
                )




    def shuffle(self):
        

        self._next_id     = 0
        self._permutation = np.random.permutation(self.number_of_events)
